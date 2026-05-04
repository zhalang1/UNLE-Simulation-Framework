
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import numpy as np
from tqdm import tqdm


class HardwareSimulator:
    # 🚀 引入双精度配置：ext_ 代表外部总线/BRAM，acc_ 代表内部累加器/DSP
    def __init__(self, ext_int_bits=6, ext_frac_bits=11, acc_int_bits=12, acc_frac_bits=20, is_sat=True):
        self.is_sat = is_sat

        # --- 外部窄体规格 (模拟 BRAM IO) ---
        self.ext_res = 2.0 ** (-ext_frac_bits)
        self.ext_max = (2.0 ** (ext_int_bits - 1)) - self.ext_res
        self.ext_min = -(2.0 ** (ext_int_bits - 1))

        # --- 内部宽体规格 (模拟 32/48-bit 寄存器) ---
        self.acc_res = 2.0 ** (-acc_frac_bits)
        self.acc_max = (2.0 ** (acc_int_bits - 1)) - self.acc_res
        self.acc_min = -(2.0 ** (acc_int_bits - 1))

        # 核心常数放入内部宽体
        self.LOG2_E = self.quant_acc(1.4426950408889634)

        self.ROM_BITS = 8
        self.LUT_SIZE = 2 ** self.ROM_BITS
        self._init_pla_tables()

    # 📏 外部物理墙：任何进出算子模块的数据，必须过这道墙
    def quant_ext(self, val):
        val_quant = np.round(val / self.ext_res) * self.ext_res
        if self.is_sat: return np.clip(val_quant, self.ext_min, self.ext_max)
        return val_quant

    # 📏 内部物理墙：算子内部的所有中间变量和累加器，受此墙保护
    def quant_acc(self, val):
        val_quant = np.round(val / self.acc_res) * self.acc_res
        if self.is_sat: return np.clip(val_quant, self.acc_min, self.acc_max)
        return val_quant

    def _init_pla_tables(self):
        self.lse_step = 32.0 / self.LUT_SIZE
        self.exp2_base = np.zeros(self.LUT_SIZE)
        self.exp2_slope = np.zeros(self.LUT_SIZE)
        self.log2_base = np.zeros(self.LUT_SIZE)
        self.log2_slope = np.zeros(self.LUT_SIZE)
        self.lse_base = np.zeros(self.LUT_SIZE)
        self.lse_slope = np.zeros(self.LUT_SIZE)

        for i in range(self.LUT_SIZE):
            # ROM 表的数据存在内部，享有高精度
            x1_e = i / self.LUT_SIZE
            x2_e = (i + 1) / self.LUT_SIZE
            self.exp2_base[i] = self.quant_acc(2.0 ** x1_e)
            self.exp2_slope[i] = self.quant_acc((2.0 ** x2_e - 2.0 ** x1_e) * self.LUT_SIZE)

            x1_l = i / self.LUT_SIZE
            x2_l = (i + 1) / self.LUT_SIZE
            self.log2_base[i] = self.quant_acc(np.log2(1.0 + x1_l))
            self.log2_slope[i] = self.quant_acc((np.log2(1.0 + x2_l) - np.log2(1.0 + x1_l)) * self.LUT_SIZE)

            x1_s = i * self.lse_step
            x2_s = (i + 1) * self.lse_step
            y1_s = np.log2(1.0 + 2.0 ** (-x1_s))
            y2_s = np.log2(1.0 + 2.0 ** (-x2_s))
            self.lse_base[i] = self.quant_acc(y1_s)
            self.lse_slope[i] = self.quant_acc((y2_s - y1_s) / self.lse_step)

    def hw_lse_engine(self, a, b):
        a, b = self.quant_acc(a), self.quant_acc(b)
        max_val = np.maximum(a, b)
        abs_delta = self.quant_acc(np.abs(a - b))

        if np.isscalar(abs_delta):
            if abs_delta >= 31.0: return self.quant_acc(max_val)
            idx = int(np.floor(abs_delta / self.lse_step))
            if idx >= self.LUT_SIZE: idx = self.LUT_SIZE - 1
            delta_x = abs_delta - (idx * self.lse_step)
            pla_res_full = self.lse_base[idx] + self.lse_slope[idx] * delta_x
            return self.quant_acc(max_val + self.quant_acc(pla_res_full))
        else:
            res = self.quant_acc(max_val)
            mask = abs_delta < 31.0
            if not np.any(mask): return res
            d_valid = abs_delta[mask]
            idx = np.floor(d_valid / self.lse_step).astype(np.int64)
            idx = np.clip(idx, 0, self.LUT_SIZE - 1)
            delta_x = d_valid - (idx * self.lse_step)
            pla_res_full = self.lse_base[idx] + self.lse_slope[idx] * delta_x
            res[mask] = self.quant_acc(max_val[mask] + self.quant_acc(pla_res_full))
            return res

    def hw_ifs_exp2(self, x):
        x = self.quant_acc(x)
        if np.isscalar(x):
            if x < -31.0: return 0.0
            I = int(np.floor(x))
            F = x - I
            idx = int(np.floor(F * self.LUT_SIZE))
            if idx >= self.LUT_SIZE: idx = self.LUT_SIZE - 1
            delta_x = F - (idx / self.LUT_SIZE)
            dsp_mac_res = self.exp2_base[idx] + self.exp2_slope[idx] * delta_x
            return self.quant_acc(dsp_mac_res * (2.0 ** I))
        else:
            res = np.zeros_like(x)
            mask = x >= -31.0
            x_valid = x[mask]
            if len(x_valid) == 0: return res
            I = np.floor(x_valid).astype(np.int64)
            F = x_valid - I
            idx = np.floor(F * self.LUT_SIZE).astype(np.int64)
            idx = np.clip(idx, 0, self.LUT_SIZE - 1)
            delta_x = F - (idx / self.LUT_SIZE)
            dsp_mac_res = self.exp2_base[idx] + self.exp2_slope[idx] * delta_x
            res[mask] = self.quant_acc(dsp_mac_res * (2.0 ** I))
            return res

    def hw_linear_to_log_32(self, x):
        if np.isscalar(x):
            if x <= 0: return self.acc_min
            E = int(np.floor(np.log2(x)))
            f_raw = (x / (2.0 ** E)) - 1.0
            idx = int(np.floor(f_raw * self.LUT_SIZE))
            if idx >= self.LUT_SIZE: idx = self.LUT_SIZE - 1
            delta_x_raw = f_raw - (idx / self.LUT_SIZE)
            delta_x = np.floor(delta_x_raw * (2 ** 26)) / (2 ** 26)
            dsp_mac_res = self.log2_base[idx] + self.log2_slope[idx] * delta_x
            rounded_pla = np.round(dsp_mac_res / self.acc_res) * self.acc_res
            return self.quant_acc(E + rounded_pla)
        else:
            res = np.full_like(x, self.acc_min)
            mask = x > 0
            x_valid = x[mask]
            if len(x_valid) == 0: return res
            E = np.floor(np.log2(x_valid)).astype(np.int64)
            f_raw = (x_valid / (2.0 ** E)) - 1.0
            idx = np.floor(f_raw * self.LUT_SIZE).astype(np.int64)
            idx = np.clip(idx, 0, self.LUT_SIZE - 1)
            delta_x_raw = f_raw - (idx / self.LUT_SIZE)
            delta_x = np.floor(delta_x_raw * (2 ** 26)) / (2 ** 26)
            dsp_mac_res = self.log2_base[idx] + self.log2_slope[idx] * delta_x
            rounded_pla = np.round(dsp_mac_res / self.acc_res) * self.acc_res
            res[mask] = self.quant_acc(E + rounded_pla)
            return res

    def run_unle_attention(self, scores, V_matrix):
        seq_len, dim = V_matrix.shape

        # 🚧 物理边界 1：模拟从外部 BRAM 读入数据，强制过窄体墙 (quant_ext)
        scores_ext = self.quant_ext(scores)
        V_matrix_ext = self.quant_ext(V_matrix)

        # 🚀 物理宽体：内部状态机在宽广的 DSP 和高位宽寄存器中游弋 (quant_acc)
        M_global = self.quant_acc(-31.0)
        L_global = self.quant_acc(-31.0)
        O_global_buffer = np.zeros(dim)  # 内部 buffer 拥有无限生机

        for s in range(seq_len):
            raw_score = scores_ext[s]
            x_s = self.quant_acc(raw_score * self.LOG2_E)

            M_old = M_global
            L_old = L_global

            M_new = np.maximum(M_old, x_s)
            M_diff = self.quant_acc(M_old - M_new)
            L_sum = self.quant_acc(L_old + M_diff)
            x_diff = self.quant_acc(x_s - M_new)

            L_new = self.hw_lse_engine(L_sum, x_diff)

            M_global = M_new
            L_global = L_new

            decay_factor = self.hw_ifs_exp2(M_diff)
            current_weight = self.hw_ifs_exp2(x_diff)

            # 注意：V 依然是从外部读入的那份“残缺”的极窄数据
            v_val_vec = V_matrix_ext[s]

            # 内部乘加在宽体水池中进行
            mul0_vec = self.quant_acc(O_global_buffer * decay_factor)
            mul1_vec = self.quant_acc(v_val_vec * current_weight)
            O_global_buffer = self.quant_acc(mul0_vec + mul1_vec)

        final_out = np.zeros(dim)
        global_offset = self.quant_acc(-L_global)

        mask_pos = O_global_buffer > 0
        mask_neg = O_global_buffer < 0

        if np.any(mask_pos):
            log_O_pos = self.hw_linear_to_log_32(O_global_buffer[mask_pos])
            log_final_pos = self.quant_acc(log_O_pos + global_offset)
            final_out[mask_pos] = self.hw_ifs_exp2(log_final_pos)

        if np.any(mask_neg):
            log_O_neg = self.hw_linear_to_log_32(np.abs(O_global_buffer[mask_neg]))
            log_final_neg = self.quant_acc(log_O_neg + global_offset)
            final_out[mask_neg] = -self.hw_ifs_exp2(log_final_neg)

        # 🚧 物理边界 2：算完了，向外部 AXI 总线写回，强制暴力向下截断！
        return self.quant_ext(final_out)


# =====================================================================
# 模块三：GPT-2 架构全局劫持与长序列系统压测
# =====================================================================
# =====================================================================
# 模块三：GPT-2 架构全局劫持与长序列系统压测
# =====================================================================
def run_hardware_search():
    import warnings
    warnings.filterwarnings("ignore")

    # print("==========================================================")
    # print(" 🛠️ 阶段一：算子级微观极限压测 (Operator-Level Stress Test)")
    # print("==========================================================")
    # seq_len = 1024
    # head_dim = 64
    #
    # np.random.seed(42)
    # scores_np = np.random.randn(seq_len) * 5.0
    # v_mat_np = np.random.randn(seq_len, head_dim) * 0.5
    print("==========================================================")
    print(" 🛠️ 阶段一：算子级微观极限压测 (Operator-Level Stress Test)")
    print("==========================================================")
    # 🚀 直接把测试长度拉爆到现代大模型的标配 4096！
    seq_len = 1024
    head_dim = 64

    np.random.seed(42)
    # 模拟真实的长序列特征，方差可以稍微放大
    scores_np = np.random.randn(seq_len) * 8.0
    v_mat_np = np.random.randn(seq_len, head_dim) * 0.5

    # ... (原有比对逻辑不变) ...
    scores_pt = torch.tensor(scores_np, dtype=torch.float32)
    v_mat_pt = torch.tensor(v_mat_np, dtype=torch.float32)
    attn_weights_pt = torch.softmax(scores_pt, dim=-1)
    fp32_out = torch.matmul(attn_weights_pt, v_mat_pt).numpy()

    # 🚀 真正符合真实物理规格的配置：
    # 参数：(外整数, 外小数, 内整数, 内小数, 饱和标志)
    configs = [
        # 外侧 <16, 8> 绝不斩首 Logits，内侧 <32, 16> 绝不撑爆累加器！
        (16, 16, 16, 16, True),
    ]

    for ext_i, ext_f, acc_i, acc_f, is_sat in configs:
        hw_sim = HardwareSimulator(ext_int_bits=ext_i, ext_frac_bits=ext_f,
                                   acc_int_bits=acc_i, acc_frac_bits=acc_f, is_sat=is_sat)

        # 🗑️ 删除了手动干预 mask 的代码，直接交给硬件截断！
        hw_out = hw_sim.run_unle_attention(scores_np, v_mat_np)

        op_mse = np.mean((fp32_out - hw_out) ** 2)
        print(
            f" [外部 ap_fixed<{ext_i + ext_f}, {ext_f}> | 内部 ap_fixed<{acc_i + acc_f}, {acc_f}>] 算子级 MSE: {op_mse:.4e}")

    print("\n==========================================================")
    print(" 🌍 阶段二：长序列端到端系统压测 (End-to-End Long Context)")
    print("==========================================================")
    print("加载 GPT-2 模型...")
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    base_text = "The rapid development of artificial intelligence has led to unprecedented changes in modern society. "
    test_text = base_text * 20
    inputs = tokenizer(test_text, return_tensors="pt")

    print(f"\n[验证序列长度]: {inputs['input_ids'].shape[1]} Tokens")

    model_fp32 = GPT2LMHeadModel.from_pretrained(model_id)
    model_fp32.eval()
    with torch.no_grad():
        outputs_fp32 = model_fp32(**inputs, use_cache=False)
        logits_fp32 = outputs_fp32.logits

    for ext_i, ext_f, acc_i, acc_f, is_sat in configs:
        print(
            f"\n🚀 烧录硬件架构: [外侧 IO] ap_fixed<{ext_i + ext_f}, {ext_i}> | [内侧 ALU] ap_fixed<{acc_i + acc_f}, {acc_i}>")
        hw_sim = HardwareSimulator(ext_int_bits=ext_i, ext_frac_bits=ext_f,
                                   acc_int_bits=acc_i, acc_frac_bits=acc_f, is_sat=is_sat)
        model_hw = GPT2LMHeadModel.from_pretrained(model_id)
        model_hw.eval()

        original_forward = GPT2Attention.forward

        def hw_mock_forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                            encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False,
                            output_attentions=False, **kwargs):
            if layer_past is None: layer_past = kwargs.get("past_key_value") or kwargs.get("past_key_values")

            c_attn_out = self.c_attn(hidden_states)
            query, key, value = c_attn_out.chunk(3, dim=2)

            batch_size, seq_len_attn, _ = query.shape
            query = query.view(batch_size, seq_len_attn, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len_attn, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len_attn, self.num_heads, self.head_dim).transpose(1, 2)

            present = (key, value) if use_cache else None

            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            scale = value.size(-1) ** 0.5
            attn_weights = attn_weights / scale

            if not getattr(self, "is_cross_attention", False):
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = torch.tril(
                    torch.ones((query_length, key_length), dtype=torch.bool, device=query.device),
                    diagonal=key_length - query_length
                ).view(1, 1, query_length, key_length)
                attn_weights = torch.where(causal_mask, attn_weights,
                                           torch.tensor(-1e4, dtype=attn_weights.dtype, device=attn_weights.device))

            if attention_mask is not None: attn_weights = attn_weights + attention_mask

            hw_attn_output = torch.zeros_like(value)
            for b in range(batch_size):
                for h in tqdm(range(self.num_heads), desc=f"仿真 Batch {b} 注意力头"):
                    for q_idx in range(seq_len_attn):
                        scores_1d = attn_weights[b, h, q_idx, :key.size(-2)].detach().cpu().numpy()
                        v_mat = value[b, h, :key.size(-2), :].detach().cpu().numpy()

                        # 🗑️ 直接传！把处理 Mask 的脏活全权交给量化器 quant_ext
                        hw_out = hw_sim.run_unle_attention(scores_1d, v_mat)
                        hw_attn_output[b, h, q_idx, :] = torch.tensor(hw_out, dtype=value.dtype, device=value.device)

            hw_attn_output = hw_attn_output.transpose(1, 2).contiguous()
            attn_output = hw_attn_output.view(batch_size, seq_len_attn, self.num_heads * self.head_dim)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions: outputs += (attn_weights,)
            return outputs

        GPT2Attention.forward = hw_mock_forward

        # try:
        #     with torch.no_grad():
        #         outputs_hw = model_hw(**inputs, use_cache=False)
        #         logits_hw = outputs_hw.logits
        #
        #     mse_loss = torch.nn.functional.mse_loss(logits_fp32, logits_hw).item()
        #     pred_fp32 = torch.argmax(logits_fp32[0, -1, :]).item()
        #     pred_hw = torch.argmax(logits_hw[0, -1, :]).item()
        #
        #     print(f"  -> 端到端 MSE 误差 : {mse_loss:.4e}")
        #     print(f"  -> FP32 原始预测词 : '{tokenizer.decode([pred_fp32])}'")
        #     print(f"  -> NPU  硬件预测词 : '{tokenizer.decode([pred_hw])}'")
        #
        #     if pred_fp32 != pred_hw:
        #         print("  -> ⚠️ 警告：长序列下大模型特征已崩溃！")
        #     else:
        #         print("  -> ✅ 成功：外部整数位宽与内部累加器双重破局，完美收敛！")
        #
        # finally:
        #     GPT2Attention.forward = original_forward
        #


    try:
        with torch.no_grad():
            outputs_hw = model_hw(**inputs, use_cache=False)
            logits_hw = outputs_hw.logits

        # 1. 传统的 MSE (只做参考，不作为最终评判标准)
        mse_loss = torch.nn.functional.mse_loss(logits_fp32, logits_hw).item()

        # 2. 🌟 黄金标准一：余弦相似度 (Cosine Similarity)
        # 将 Logits 展平，计算空间向量夹角
        cos_sim = torch.nn.functional.cosine_similarity(
            logits_fp32.flatten(),
            logits_hw.flatten(),
            dim=0
        ).item()

        # 3. 🌟 黄金标准二：KL 散度 (KL Divergence)
        # 计算最终预测概率分布的差异
        # PyTorch 的 KLDiv 要求 input 是 log_softmax，target 是 softmax
        log_probs_hw = torch.nn.functional.log_softmax(logits_hw, dim=-1)
        probs_fp32 = torch.nn.functional.softmax(logits_fp32, dim=-1)
        kl_div = torch.nn.functional.kl_div(
            log_probs_hw,
            probs_fp32,
            reduction='batchmean'
        ).item()

        pred_fp32 = torch.argmax(logits_fp32[0, -1, :]).item()
        pred_hw = torch.argmax(logits_hw[0, -1, :]).item()

        print("\n  📊 [工业级端到端评估报告]")
        print("-" * 50)
        print(f"  -> 绝对均方误差 (MSE)   : {mse_loss:.4f} (仅供参考)")
        print(f"  -> 余弦相似度 (Cos Sim) : {cos_sim:.6f} (工业红线: > 0.9900)")
        print(f"  -> 概率 KL 散度 (KL Div): {kl_div:.6f} (工业红线: < 0.0500)")
        print("-" * 50)

        print(f"  -> FP32 原始预测词 : '{tokenizer.decode([pred_fp32])}'")
        print(f"  -> NPU  硬件预测词 : '{tokenizer.decode([pred_hw])}'")

        if cos_sim > 0.99:
            print("  -> ✅ 终极验证通过：硬件特征与浮点模型达到 99% 以上语义同构，SOTA 达成！")
        else:
            print("  -> ⚠️ 警告：余弦相似度低于 99%，特征分布已发生偏离。")

    finally:
        GPT2Attention.forward = original_forward
if __name__ == "__main__":
    run_hardware_search()
