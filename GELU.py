import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
from tqdm import tqdm


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
        # 🚀 硬件架构参数 1：统一 PLA 微型查表 (默认 256 段, 可配 256)
        # ==========================================================
        self.ROM_BITS = ROM_BITS
        self.LUT_SIZE = 2 ** self.ROM_BITS
        self._init_pla_tables()

        # ==========================================================
        # 🚀 硬件架构参数 2：GELU 专用的 ARC_LUT 残差表
        # ==========================================================
        # self.arc_frac_bits = arc_frac_bits
        # self.arc_step = 2.0 ** (-arc_frac_bits)
        # self.arc_size = int(128 / self.arc_step)
        # self.ARC_LUT = np.zeros(self.arc_size)
        #
        # half_size = self.arc_size // 2
        # for i in range(-half_size, half_size):
        #     x = float(i) * self.arc_step
        #     if x == 0:
        #         silu = 0.0;
        #         gelu = 0.0
        #     else:
        #         silu = x / (1.0 + math.exp(-x))
        #         gelu = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        #     # 这里必须用 ap_fixed 将残差也锁定在硬件位宽下
        #     self.ARC_LUT[i + half_size] = self.ap_fixed(gelu - silu)
        # ==========================================================
        # 🚀 硬件架构参数 2：GELU 专用的 ARC 1阶线性插值双表 (PLA)
        # ==========================================================
        self.arc_frac_bits = arc_frac_bits
        self.arc_step = 2.0 ** (-arc_frac_bits)
        self.arc_size = int(128 / self.arc_step)  # 注意：如果 arc_frac_bits=6, 表深是 8192

        self.arc_base = np.zeros(self.arc_size)
        self.arc_slope = np.zeros(self.arc_size)

        half_size = self.arc_size // 2
        for i in range(self.arc_size):
            # 获取当前区间起点 x1 和终点 x2
            x1 = (i - half_size) * self.arc_step
            x2 = (i + 1 - half_size) * self.arc_step

            # 定义一个内部求真实残差的闭包函数
            def calc_true_arc(x):
                if x == 0:
                    return 0.0
                silu = x / (1.0 + math.exp(-x))
                gelu = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
                return gelu - silu

            y1 = calc_true_arc(x1)
            y2 = calc_true_arc(x2)

            # 存储基准值
            self.arc_base[i] = self.ap_fixed(y1)
            # 存储斜率: dy / dx
            self.arc_slope[i] = self.ap_fixed((y2 - y1) / self.arc_step)

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

    #
    # def get_arc_val(self, v_val):
    #     idx = np.floor(v_val * (2 ** self.arc_frac_bits)).astype(int)
    #     idx = np.clip(idx + self.arc_size // 2, 0, self.arc_size - 1)
    #     return self.ARC_LUT[idx]
    def get_arc_val(self, v_val):
        # 1. 提取浮点索引 (不加偏置前)
        raw_idx = np.floor(v_val / self.arc_step).astype(np.int64)

        # 2. 加上中心偏移量 (防止负数索引) 并边界钳位
        idx = raw_idx + (self.arc_size // 2)
        idx = np.clip(idx, 0, self.arc_size - 1)

        # 3. 提取残差 delta_x
        # 计算当前索引对应的物理坐标起点 x_start
        x_start = (idx - (self.arc_size // 2)) * self.arc_step
        # 输入值减去起点坐标，得到微小残差
        delta_x = self.ap_fixed(v_val - x_start)

        # 4. 查双表
        base = self.ap_fixed(self.arc_base[idx])
        slope = self.ap_fixed(self.arc_slope[idx])

        # 5. 模拟 DSP48E2 MAC 补偿计算
        pla_res = self.ap_fixed(base + slope * delta_x)

        return pla_res

    def run_unle_pointwise(self, v_val, is_gelu=True):
        v_val = self.ap_fixed(v_val)
        abs_val = np.abs(v_val)

        log2_x_int = self.hw_linear_to_log_32(abs_val)
        exponent_term = self.ap_fixed(v_val * self.LOG2_E)
        log2_denominator = self.hw_lse_engine(np.zeros_like(exponent_term), -exponent_term)

        log2_y_silu = self.ap_fixed(log2_x_int - log2_denominator)
        raw_exp_silu = self.hw_ifs_exp2(log2_y_silu)

        silu_base = np.where(v_val >= 0, raw_exp_silu, -raw_exp_silu)

        if not is_gelu:
            return silu_base

        arc_val = self.get_arc_val(v_val)
        gelu_res = silu_base + arc_val
        return gelu_res
class HWMockGELU(nn.Module):
    def __init__(self, hw_sim, layer_idx):
        super().__init__()
        self.hw_sim = hw_sim
        self.layer_idx = layer_idx  # 记录当前是第几层，用于进度条显示

    def forward(self, hidden_states):
        # hidden_states shape: [batch, seq_len, dim]
        hs_np = hidden_states.detach().cpu().numpy()
        hw_out = np.zeros_like(hs_np)
        batch_size, seq_len, dim = hs_np.shape

        # 🌟 引入 tqdm 进度条，按 Token 逐个送入硬件模拟器
        for b in range(batch_size):
            for s in tqdm(range(seq_len), desc=f"仿真 Layer {self.layer_idx:02d} 硬件 GELU", leave=False):
                # 提取单个 Token 的特征向量，过硬件算子
                token_features = hs_np[b, s, :]
                hw_out[b, s, :] = self.hw_sim.run_unle_pointwise(token_features, is_gelu=True)

        return torch.tensor(hw_out, dtype=hidden_states.dtype, device=hidden_states.device)


def run_hardware_search():
    import warnings
    warnings.filterwarnings("ignore")

    print("加载 GPT-2 模型...")
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # 🌟 1. 对齐 Softmax 的超长序列压测配置
    base_text = "The rapid development of artificial intelligence has led to unprecedented changes in modern society. "
    test_text = base_text *68
    inputs = tokenizer(test_text, return_tensors="pt")

    print(f"\n[验证序列长度]: {inputs['input_ids'].shape[1]} Tokens")
    print("=" * 60)

    model_fp32 = GPT2LMHeadModel.from_pretrained(model_id)

    # 🌟 必须追加的物理对齐：把 HuggingFace 默认的 tanh 近似 GELU
    # 强制替换为精确的 erf GELU，否则你的硬件在和错误的答案对答案！
    for block in model_fp32.transformer.h:
        block.mlp.act = nn.GELU(approximate='none')

    model_fp32.eval()
    with torch.no_grad():
        outputs_fp32 = model_fp32(**inputs, use_cache=False)
        logits_fp32 = outputs_fp32.logits

    # 🌟 我们直接对比 "原版HLS" vs "修正后的HLS"
    configs = [
        (7, 10, True, 6),  # 原版 HLS：强制抹除所有小数 (>> 8)
        (8, 11, True, 6),  # 修正版 HLS：保留 4 位小数查表 (>> 4)
    ]

    for int_b, frac_b, is_sat, arc_f in configs:
        print(f"\n🚀 烧录硬件架构: ap_fixed<{int_b + frac_b}, {int_b}> | ARC表保留小数位: {arc_f} Bit")

        # 初始化当前位宽的硬件模拟器
        hw_sim = HardwareSimulator(int_bits=int_b, frac_bits=frac_b, is_sat=is_sat, arc_frac_bits=arc_f)

        # ==========================================================
        # 🔬 架构师自证环节：算子级微观精度 (Operator-Level)
        # ==========================================================
        x_test_np = np.linspace(-10.0, 10.0, 50000)
        x_test_pt = torch.tensor(x_test_np, dtype=torch.float32)
        fp32_gelu_out = nn.GELU(approximate='none')(x_test_pt).numpy()
        hw_gelu_out = hw_sim.run_unle_pointwise(x_test_np, is_gelu=True)
        op_mse = np.mean((fp32_gelu_out - hw_gelu_out) ** 2)
        print(f"  -> 🌟 算子级 (Operator) 纯净 MSE : {op_mse:.4e}")
        # ==========================================================

        # 载入用于硬件替换的独立模型
        model_hw = GPT2LMHeadModel.from_pretrained(model_id)
        model_hw.eval()

        # 🌟 将模型每一层的 MLP 激活函数替换为带有进度条的硬件劫持模块
        for layer_idx, block in enumerate(model_hw.transformer.h):
            block.mlp.act = HWMockGELU(hw_sim, layer_idx)

        # 执行端到端硬件仿真推理
        try:
            with torch.no_grad():
                outputs_hw = model_hw(**inputs, use_cache=False)
                logits_hw = outputs_hw.logits

            # ==========================================================
            # 🌟 引入与 Softmax 完全一致的工业级端到端评估探针
            # ==========================================================
            mse_loss = torch.nn.functional.mse_loss(logits_fp32, logits_hw).item()

            cos_sim = torch.nn.functional.cosine_similarity(
                logits_fp32.flatten(),
                logits_hw.flatten(),
                dim=0
            ).item()

            log_probs_hw = torch.nn.functional.log_softmax(logits_hw, dim=-1)
            probs_fp32 = torch.nn.functional.softmax(logits_fp32, dim=-1)
            kl_div = torch.nn.functional.kl_div(
                log_probs_hw,
                probs_fp32,
                reduction='batchmean'
            ).item()

            pred_fp32 = torch.argmax(logits_fp32[0, -1, :]).item()
            pred_hw = torch.argmax(logits_hw[0, -1, :]).item()

            # print("\n  📊 [工业级端到端评估报告]")
            # print("-" * 50)
            # print(f"  -> 绝对均方误差 (MSE)   : {mse_loss:.4f} (仅供参考)")
            # print(f"  -> 余弦相似度 (Cos Sim) : {cos_sim:.6f} (工业红线: > 0.9900)")
            # print(f"  -> 概率 KL 散度 (KL Div): {kl_div:.6f} (工业红线: < 0.0500)")
            # print("-" * 50)
            print("\n  📊 [工业级端到端评估报告]")
            print("-" * 50)
            # 恢复你熟悉的科学计数法和名称
            print(f"  -> 端到端 MSE 误差    : {mse_loss:.4e} (微观绝对误差)")
            print(f"  -> 余弦相似度 (Cos Sim) : {cos_sim:.6f} (工业红线: > 0.9900)")
            print(f"  -> 概率 KL 散度 (KL Div): {kl_div:.6f} (工业红线: < 0.0500)")
            print("-" * 50)
            print(f"  -> FP32 原始预测词 : '{tokenizer.decode([pred_fp32])}'")
            print(f"  -> NPU  硬件预测词 : '{tokenizer.decode([pred_hw])}'")

            if cos_sim > 0.99:
                print("  -> ✅ 终极验证通过：GELU 硬件特征与浮点模型达到 99% 以上语义同构，SOTA 达成！")
            else:
                print("  -> ⚠️ 警告：余弦相似度低于 99%，特征分布已发生偏离。")

        except Exception as e:
             print(f"仿真过程出现异常: {e}")

if __name__ == "__main__":
    run_hardware_search()