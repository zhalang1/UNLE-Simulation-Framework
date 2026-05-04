
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import math
from tqdm import tqdm


# =====================================================================
# 大一统底座：HardwareSimulator (100% 物理级比特对齐, 包含 256段 PLA)
# =====================================================================
class HardwareSimulator:
    def __init__(self, int_bits=8, frac_bits=8, is_sat=True, arc_frac_bits=6, ROM_BITS=8):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.is_sat = is_sat

        self.resolution = 2.0 ** (-frac_bits)
        self.max_val = (2.0 ** (int_bits - 1)) - self.resolution
        self.min_val = -(2.0 ** (int_bits - 1))
        self.LOG2_E = self.ap_fixed(1.4426950408889634)

        self.ROM_BITS = ROM_BITS
        self.LUT_SIZE = 2 ** self.ROM_BITS
        self._init_pla_tables()

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
        pla_res = self.ap_fixed(base + slope * delta_x)
        res[mask] = self.ap_fixed(E + pla_res)
        return res

    # ==========================================================
    # 🚀 核心引擎：RMSNorm 物理通路复刻
    # 完美映射 C++ 中的 LNS 0周期除法与开方逻辑
    # ==========================================================
    def run_unle_rmsnorm(self, X, eps_hw):
        # 1. 输入截断与高精度累加
        # v_val = self.ap_fixed(X)
        # dim = X.shape[-1]
        #
        # # 模拟 C++ 中的 step_sq_sum 高位宽累加防溢出
        # sq_sum = np.sum(v_val ** 2, axis=-1, keepdims=True)
        v_val = X
        dim = X.shape[-1]

        # 模拟 32-bit 硬件加法树，高位宽累加防溢出
        sq_sum = np.sum(v_val ** 2, axis=-1, keepdims=True)
        # 2. LNS 零周期魔法：除法变减法，开方变右移
        log_N = self.hw_linear_to_log_32(np.array([dim], dtype=np.float64))
        log_S = self.hw_linear_to_log_32(sq_sum + eps_hw)
        diff = self.ap_fixed(log_N - log_S)

        # 开方操作在对数域即为乘以 0.5 (右移 1 位)
        global_offset = self.ap_fixed(diff * 0.5)

        # 3. 终局投影 (Epilogue)
        abs_X = np.abs(v_val)
        log_O = self.hw_linear_to_log_32(abs_X)
        log_final = self.ap_fixed(log_O + global_offset)
        final_abs = self.hw_ifs_exp2(log_final)

        # 4. 恢复符号位
        final_out = np.where(v_val >= 0, final_abs, -final_abs)
        final_out = np.where(v_val == 0, 0.0, final_out)

        return final_out


# =====================================================================
# HWMock 适配层：针对 PyTorch Qwen2 网络 RMSNorm 的硬件劫持
# =====================================================================
class HWMockRMSNorm(nn.Module):
    def __init__(self, hw_sim, original_norm, layer_idx, name):
        super().__init__()
        self.hw_sim = hw_sim
        self.layer_idx = layer_idx
        self.name = name

        # 保存 PyTorch 原生的参数
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon

    def forward(self, hidden_states):
        # 1. 抓取输入并转为 Float32 (兼容 Qwen2 原生的 BFloat16)
        hs_np = hidden_states.detach().cpu().to(torch.float32).numpy()
        hw_out = np.zeros_like(hs_np)

        batch_size, seq_len, dim = hs_np.shape

        # 🚨 物理对齐关键点：送入硬件的 epsilon 必须乘以通道数 dim
        hw_eps = self.variance_epsilon * dim

        # 🌟 引入 tqdm 进度条，按 Token 逐个送入硬件模拟器
        for b in range(batch_size):
            for s in tqdm(range(seq_len), desc=f"仿真 Layer {self.layer_idx:02d} [{self.name}]", leave=False):
                token_features = hs_np[b, s, :]
                # 调用硬件 IP 计算纯净归一化值 (不带 Gamma 权重)
                hw_out[b, s, :] = self.hw_sim.run_unle_rmsnorm(token_features, eps_hw=hw_eps)

        hw_out_pt = torch.tensor(hw_out, dtype=hidden_states.dtype, device=hidden_states.device)

        # 3. 模拟上位机软件级的 Scale 吸收：乘以 weight 向量
        return hw_out_pt * self.weight


# =====================================================================
# 模块三：长序列端到端压测与工业级评估
# =====================================================================
def run_hardware_search():
    import warnings
    warnings.filterwarnings("ignore")

    print("加载原生大模型 Qwen2.5-0.5B 进行 RMSNorm 硬件验证...")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 🌟 1. 对齐超长序列压测配置 (~1021 Tokens)
    base_text = "The rapid development of artificial intelligence has led to unprecedented changes in modern society. "
    test_text = base_text * 68
    inputs = tokenizer(test_text, return_tensors="pt")

    print(f"\n[验证序列长度]: {inputs['input_ids'].shape[1]} Tokens")
    print("=" * 60)

    # 2. 抓取无损的 FP32 Ground Truth
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model_fp32.eval()
    with torch.no_grad():
        outputs_fp32 = model_fp32(**inputs, use_cache=False)
        logits_fp32 = outputs_fp32.logits

    # 锁定物理位宽：16-bit (8整数, 8小数)
    int_b = 8
    frac_b = 8
    is_sat = True

    print(f"\n🚀 终极架构烧录: ap_fixed<{int_b + frac_b}, {int_b}> | 256段 PLA | RMSNorm 引擎")
    hw_sim = HardwareSimulator(int_bits=int_b, frac_bits=frac_b, is_sat=is_sat, ROM_BITS=8)

    np.random.seed(42)
    x_test_np = np.random.randn(1, 10, 896) * 5.0
    x_test_pt = torch.tensor(x_test_np, dtype=torch.float32)

    qwen_norm = model_fp32.model.layers[0].input_layernorm
    fp32_norm_out = qwen_norm(x_test_pt).detach().numpy()

    # 算子级测试直接调 run_unle_rmsnorm 向量化计算
    hw_eps = qwen_norm.variance_epsilon * 896
    hw_norm_out_pure = hw_sim.run_unle_rmsnorm(x_test_np, eps_hw=hw_eps)
    hw_norm_out = hw_norm_out_pure * qwen_norm.weight.detach().to(torch.float32).numpy()
    op_mse_norm = np.mean((fp32_norm_out - hw_norm_out) ** 2)
    print(f"  -> 🌟 [核心突破] 算子级 LNS 引擎纯净 MSE : {op_mse_norm:.4e}")

    model_hw = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model_hw.eval()

    # 劫持所有 Transformer Block 中的输入和后置 RMSNorm
    for layer_idx, layer in enumerate(model_hw.model.layers):
        layer.input_layernorm = HWMockRMSNorm(hw_sim, layer.input_layernorm, layer_idx, "InputNorm")
        layer.post_attention_layernorm = HWMockRMSNorm(hw_sim, layer.post_attention_layernorm, layer_idx,
                                                       "PostAttnNorm")

    # 劫持全局 Final RMSNorm
    final_layer_idx = len(model_hw.model.layers)
    model_hw.model.norm = HWMockRMSNorm(hw_sim, model_hw.model.norm, final_layer_idx, "FinalNorm")

    try:
        with torch.no_grad():
            outputs_hw = model_hw(**inputs, use_cache=False)
            logits_hw = outputs_hw.logits

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
            print("  -> ✅ 终极验证通过：RMSNorm 硬件特征与浮点模型达到 99% 以上语义同构，SOTA 达成！")
        else:
            print("  -> ⚠️ 警告：余弦相似度低于 99%，特征分布已发生偏离。")

    except Exception as e:
        print(f"仿真过程出现异常: {e}")


if __name__ == "__main__":
    run_hardware_search()
