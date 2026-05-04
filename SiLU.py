import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
from tqdm import tqdm

# =====================================================================
# 模块一：100% 物理级比特对齐模拟器 (SiLU 底座精准测试)
# =====================================================================

class HardwareSimulator:
    def __init__(self, int_bits=6, frac_bits=11, is_sat=True, arc_frac_bits=6, ROM_BITS=8):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.is_sat = is_sat

        self.resolution = 2.0 ** (-frac_bits)
        self.max_val = (2.0 ** (int_bits - 1)) - self.resolution
        self.min_val = -(2.0 ** (int_bits - 1))
        self.LOG2_E = self.ap_fixed(1.4426950408889634)

        # ==========================================================
        # 🚀 硬件架构参数 1：统一 PLA 微型查表 (默认 64 段, 可配 256)
        # ==========================================================
        self.ROM_BITS = ROM_BITS
        self.LUT_SIZE = 2 ** self.ROM_BITS
        self._init_pla_tables()

        # ==========================================================
        # 🚀 硬件架构参数 2：GELU 专用的 ARC_LUT 残差表
        # ==========================================================
        self.arc_frac_bits = arc_frac_bits
        self.arc_step = 2.0 ** (-arc_frac_bits)
        self.arc_size = int(128 / self.arc_step)
        self.ARC_LUT = np.zeros(self.arc_size)

        half_size = self.arc_size // 2
        for i in range(-half_size, half_size):
            x = float(i) * self.arc_step
            if x == 0:
                silu = 0.0;
                gelu = 0.0
            else:
                silu = x / (1.0 + math.exp(-x))
                gelu = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
            # 这里必须用 ap_fixed 将残差也锁定在硬件位宽下
            self.ARC_LUT[i + half_size] = self.ap_fixed(gelu - silu)

    def _init_pla_tables(self):
        self.lse_step = 32.0 / self.LUT_SIZE

        self.exp2_base = np.zeros(self.LUT_SIZE)
        self.exp2_slope = np.zeros(self.LUT_SIZE)
        self.log2_base = np.zeros(self.LUT_SIZE)
        self.log2_slope = np.zeros(self.LUT_SIZE)
        self.lse_base = np.zeros(self.LUT_SIZE)
        self.lse_slope = np.zeros(self.LUT_SIZE)

        for i in range(self.LUT_SIZE):
            x1_e = i / self.LUT_SIZE
            x2_e = (i + 1) / self.LUT_SIZE
            self.exp2_base[i] = 2.0 ** x1_e
            self.exp2_slope[i] = (2.0 ** x2_e - 2.0 ** x1_e) * self.LUT_SIZE

            x1_l = i / self.LUT_SIZE
            x2_l = (i + 1) / self.LUT_SIZE
            self.log2_base[i] = np.log2(1.0 + x1_l)
            self.log2_slope[i] = (np.log2(1.0 + x2_l) - np.log2(1.0 + x1_l)) * self.LUT_SIZE

            x1_s = i * self.lse_step
            x2_s = (i + 1) * self.lse_step
            y1_s = np.log2(1.0 + 2.0 ** (-x1_s))
            y2_s = np.log2(1.0 + 2.0 ** (-x2_s))
            self.lse_base[i] = y1_s
            self.lse_slope[i] = (y2_s - y1_s) / self.lse_step

    def ap_fixed(self, val):
        val_quant = np.round(val / self.resolution) * self.resolution
        if self.is_sat:
            return np.clip(val_quant, self.min_val, self.max_val)
        return val_quant

    # ==========================================================
    # 🌟 核心引擎替换：完全基于 向量化掩码 + DSP48E2 MAC 补偿
    # ==========================================================
    def hw_ifs_exp2(self, x):
        x = self.ap_fixed(x)
        res = np.zeros_like(x)

        mask = x >= -31.0
        x_valid = x[mask]
        if len(x_valid) == 0: return res

        I = np.floor(x_valid).astype(np.int64)
        F = self.ap_fixed(x_valid - I)

        idx = np.floor(F * self.LUT_SIZE).astype(np.int64)
        idx = np.clip(idx, 0, self.LUT_SIZE - 1)
        delta_x = self.ap_fixed(F - (idx / self.LUT_SIZE))

        base = self.ap_fixed(self.exp2_base[idx])
        slope = self.ap_fixed(self.exp2_slope[idx])

        # 模拟 DSP 补偿
        pla_res = self.ap_fixed(base + slope * delta_x)
        res[mask] = self.ap_fixed(pla_res * (2.0 ** I))
        return res

    def hw_linear_to_log_32(self, x):
        res = np.full_like(x, self.min_val)
        mask = x > 0
        x_valid = x[mask]
        if len(x_valid) == 0: return res

        E = np.floor(np.log2(x_valid)).astype(np.int64)
        f = self.ap_fixed((x_valid / (2.0 ** E)) - 1.0)

        idx = np.floor(f * self.LUT_SIZE).astype(np.int64)
        idx = np.clip(idx, 0, self.LUT_SIZE - 1)
        delta_x = self.ap_fixed(f - (idx / self.LUT_SIZE))

        base = self.ap_fixed(self.log2_base[idx])
        slope = self.ap_fixed(self.log2_slope[idx])

        # 模拟 DSP 补偿
        pla_res = self.ap_fixed(base + slope * delta_x)
        res[mask] = self.ap_fixed(E + pla_res)
        return res

    def hw_lse_engine(self, a, b):
        a, b = self.ap_fixed(a), self.ap_fixed(b)
        max_val = np.maximum(a, b)
        abs_delta = self.ap_fixed(np.abs(a - b))

        res = self.ap_fixed(max_val)
        mask = abs_delta < 31.0
        d_valid = abs_delta[mask]
        if len(d_valid) == 0: return res

        idx = np.floor(d_valid / self.lse_step).astype(np.int64)
        idx = np.clip(idx, 0, self.LUT_SIZE - 1)

        # 保持内部残差的无损相减，送入 DSP
        delta_x = self.ap_fixed(d_valid - (idx * self.lse_step))

        base = self.ap_fixed(self.lse_base[idx])
        slope = self.ap_fixed(self.lse_slope[idx])

        # 模拟 DSP 补偿
        pla_res_full = base + slope * delta_x
        res[mask] = self.ap_fixed(max_val[mask] + pla_res_full)
        return res

    def run_unle_silu(self, v_val):
        """
        🚀 SiLU 核心物理通路复刻
        消除了静态量化缩放常数，纯粹验证 LNS 对数域数学转换精度
        """
        v_val = self.ap_fixed(v_val)
        abs_val = np.abs(v_val)

        # 1. 线性转对数: log2(|x|)
        log2_x_int = self.hw_linear_to_log_32(abs_val)
        exponent_term = self.ap_fixed(v_val * self.LOG2_E)

        # 2. 对数域求分母 LSE(0, -x*log2e) -> log2(1 + e^-x)
        log2_denominator = self.hw_lse_engine(np.zeros_like(exponent_term), -exponent_term)

        # 3. 对数域除法变减法
        log2_y_silu = self.ap_fixed(log2_x_int - log2_denominator)

        # 4. 指数还原
        raw_exp_silu = self.hw_ifs_exp2(log2_y_silu)

        # 5. 恢复符号位
        silu_base = np.where(v_val >= 0, raw_exp_silu, -raw_exp_silu)

        return silu_base


from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# 模块二：Qwen2.5 劫持模块 (增加 tqdm 进度条支持)
# =====================================================================
class HWMockSiLU(nn.Module):
    def __init__(self, hw_sim, layer_idx):
        super().__init__()
        self.hw_sim = hw_sim
        self.layer_idx = layer_idx  # 记录当前是第几层，用于进度条显示

    def forward(self, hidden_states):
        # 🌟 Qwen2 原生使用 bfloat16，NumPy 不支持，必须强制中转为 float32
        hs_np = hidden_states.detach().cpu().to(torch.float32).numpy()
        hw_out = np.zeros_like(hs_np)

        # 提取维度信息 (通常为 [batch_size, seq_len, intermediate_size])
        batch_size, seq_len, dim = hs_np.shape

        # 🌟 引入 tqdm 进度条，按 Token 逐个送入硬件模拟器
        for b in range(batch_size):
            for s in tqdm(range(seq_len), desc=f"仿真 Layer {self.layer_idx:02d} 硬件 SiLU", leave=False):
                # 提取单个 Token 的特征向量，过硬件算子
                token_features = hs_np[b, s, :]
                hw_out[b, s, :] = self.hw_sim.run_unle_silu(token_features)

        # 将结果转回 Tensor，并严格恢复为输入时的数据类型 (bfloat16) 和设备
        return torch.tensor(hw_out, dtype=hidden_states.dtype, device=hidden_states.device)


# =====================================================================
# 模块三：Qwen2.5 原生 SiLU 长序列端到端评测
# =====================================================================
def run_hardware_search():
    import warnings
    warnings.filterwarnings("ignore")

    print("加载原生使用 SiLU 的现代大模型 (Qwen2.5-0.5B)...")
    model_id = "Qwen/Qwen2.5-0.5B"

    # Qwen2 系列必须使用 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 🌟 对齐 GELU 的超长序列压测配置
    base_text = "The rapid development of artificial intelligence has led to unprecedented changes in modern society. "
    test_text = base_text * 68  # 构造长文本
    inputs = tokenizer(test_text, return_tensors="pt")

    print(f"\n[验证序列长度]: {inputs['input_ids'].shape[1]} Tokens")
    print("=" * 60)

    # 1. 抓取无损的 FP32 Ground Truth
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model_fp32.eval()

    with torch.no_grad():
        outputs_fp32 = model_fp32(**inputs, use_cache=False)
        logits_fp32 = outputs_fp32.logits

    # 🌟 严格对照实验：传统 16-bit 架构 vs 面向 DSP48E2 满载的 19-bit 极限架构
    configs = [
        (8, 8, True),  # 传统对照组：ap_fixed<16, 8> (16-bit 边缘方案)
        (8, 11, True),  # 论文核心组：ap_fixed<19, 8> (完美填满 DSP 18-bit 输入端口)
    ]

    for int_b, frac_b, is_sat in configs:
        print(f"\n🚀 烧录硬件架构: ap_fixed<{int_b + frac_b}, {int_b}> | 测试原生 SiLU")

        # 初始化硬件级定点约束模拟器
        hw_sim = HardwareSimulator(int_bits=int_b, frac_bits=frac_b, is_sat=is_sat)

        # ==========================================================
        # 🔬 架构师自证环节：纯净算子级 (Operator-Level) 精度测试
        # ==========================================================
        x_test_np = np.linspace(-10.0, 10.0, 50000)
        x_test_pt = torch.tensor(x_test_np, dtype=torch.float32)
        fp32_silu_out = nn.SiLU()(x_test_pt).numpy()
        hw_silu_out = hw_sim.run_unle_silu(x_test_np)
        op_mse = np.mean((fp32_silu_out - hw_silu_out) ** 2)
        print(f"  -> 🌟 [核心突破] 算子级纯净 MSE : {op_mse:.4e}")
        # ==========================================================

        model_hw = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model_hw.eval()

        # 🌟 核心：劫持 Qwen2 原生架构的 SiLU (注意层路径的变化，并传入 layer_idx)
        for layer_idx, layer in enumerate(model_hw.model.layers):
            # Qwen2 使用 SwiGLU，激活函数绑定在 mlp.act_fn 上
            layer.mlp.act_fn = HWMockSiLU(hw_sim, layer_idx)

        # 执行端到端硬件仿真推理
        try:
            with torch.no_grad():
                outputs_hw = model_hw(**inputs, use_cache=False)
                logits_hw = outputs_hw.logits

            # ==========================================================
            # 🌟 引入统一的工业级端到端评估探针 (启用 Float64 核级精度)
            # ==========================================================
            # ⚠️ 面对 1.55 亿级别的张量，BFloat16 或 Float32 的平方累加会引发大数吃小数
            # 必须全面提升至 Float64 才能算出绝对真实的 Cos Sim 和 KL Div
            logits_fp32_f64 = logits_fp32.to(torch.float64)
            logits_hw_f64 = logits_hw.to(torch.float64)

            mse_loss = torch.nn.functional.mse_loss(logits_fp32_f64, logits_hw_f64).item()

            cos_sim = torch.nn.functional.cosine_similarity(
                logits_fp32_f64.flatten(),
                logits_hw_f64.flatten(),
                dim=0
            ).item()

            log_probs_hw = torch.nn.functional.log_softmax(logits_hw_f64, dim=-1)
            probs_fp32 = torch.nn.functional.softmax(logits_fp32_f64, dim=-1)
            kl_div = torch.nn.functional.kl_div(
                log_probs_hw,
                probs_fp32,
                reduction='batchmean'
            ).item()

            pred_fp32 = torch.argmax(logits_fp32_f64[0, -1, :]).item()
            pred_hw = torch.argmax(logits_hw_f64[0, -1, :]).item()

            print("\n  📊 [工业级端到端评估报告]")
            print("-" * 50)
            print(f"  -> 端到端 MSE 误差    : {mse_loss:.4e} (微观绝对误差)")
            print(f"  -> 余弦相似度 (Cos Sim) : {cos_sim:.6f} (工业红线: > 0.9900, 理论极限: 1.0)")
            print(f"  -> 概率 KL 散度 (KL Div): {kl_div:.6f} (工业红线: < 0.0500, 理论极限: 0.0)")
            print("-" * 50)

            print(f"  -> FP32 原始预测词 : '{tokenizer.decode([pred_fp32])}'")
            print(f"  -> NPU  硬件预测词 : '{tokenizer.decode([pred_hw])}'")

            if cos_sim > 0.99:
                print("  -> ✅ 终极验证通过：SiLU 硬件特征与浮点模型达到 99% 以上语义同构，SOTA 达成！")
            else:
                print("  -> ⚠️ 警告：余弦相似度低于 99%，特征分布已发生偏离。")

        except Exception as e:
            print(f"仿真过程出现异常: {e}")

if __name__ == "__main__":
    run_hardware_search()