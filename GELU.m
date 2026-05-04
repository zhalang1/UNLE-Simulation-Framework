% =========================================================================
% UNLE Engine - GELU (纯 SiLU 基底 + 6-bit ARC 残差补偿)
% =========================================================================
clear; clc; close all;

%% 1. 硬件架构参数配置 (严格对齐 Python 中的 ap_fixed<19, 8>)
frac_bits = 11; % 小数位宽 (对应 resolution = 2^-11)
arc_frac_bits = 4; % ARC 表的切分精度
arc_step = 2^(-arc_frac_bits); % PLA 步长

%% 2. 生成输入数据范围
x_range = linspace(-6, 6, 2000); % 高密度采样以捕获量化锯齿

%% 3. 理论计算 (标准纯正 GELU)
y_true = 0.5 * x_range .* (1 + erf(x_range / sqrt(2)));

%% 4. 硬件算子模拟计算 (物理截断 + LNS + PLA)
y_hw = zeros(size(x_range));
for i = 1:length(x_range)
    y_hw(i) = hw_gelu_engine(x_range(i), frac_bits, arc_step);
end

%% 5. 核心量化指标计算
error_abs = abs(y_hw - y_true);
max_err = max(error_abs);
mse_val = mean((y_hw - y_true).^2);
fig = figure('Name', 'UNLE Engine GELU 硬件近似度分析', 'Position', [100, 100, 950, 650], 'Color', 'w');

% ---------------- 上半图：函数曲线重合度对比 ----------------
subplot(2, 1, 1);
plot(x_range, y_true, 'b--', 'LineWidth', 2.5); hold on;
plot(x_range, y_hw, 'r-', 'LineWidth', 1.5);
grid on;
title('GELU 算子对比: 理论值 (Float) vs 硬件 LNS+PLA 架构 (Fixed-Point ap\_fixed<19, 8>)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('输入 x', 'FontSize', 12);
ylabel('输出 GELU(x)', 'FontSize', 12);
legend({'纯正数学理论曲线 (Dashed)', 'UNLE 硬件输出曲线 (Solid)'}, 'Location', 'northwest', 'FontSize', 11);
set(gca, 'FontSize', 11);

% ---------------- 下半图：硬件 PLA 近似误差分析 ----------------
ax2 = subplot(2, 1, 2);
x_fill = [x_range, fliplr(x_range)];
y_fill = [error_abs, zeros(1, length(error_abs))];
fill(x_fill, y_fill, [1.00 0.85 0.88], 'EdgeColor', 'none'); hold on; % 浅粉色填充

% 画出误差黑色实线
plot(x_range, error_abs, 'k-', 'LineWidth', 1.0);
grid on;
title('硬件架构绝对计算误差 (Absolute Error)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('输入 x', 'FontSize', 12);
ylabel('绝对误差 |y_{hw} - y_{true}|', 'FontSize', 12);
set(gca, 'FontSize', 11);
pos = get(ax2, 'Position'); 
pos(3) = pos(3) * 0.82;     
set(ax2, 'Position', pos);  
str_box = sprintf('【量化评估】\n\nMAX Error:\n%.3e\n\nMSE:\n%.3e', max_err, mse_val);
x_limits = xlim;
y_limits = ylim;

% 动态计算文本框的锚点位置（挂载在图表右侧外部）
text_x = x_limits(2) + (x_limits(2) - x_limits(1)) * 0.12; 
text_y = y_limits(1) + (y_limits(2) - y_limits(1)) * 0.5;

text(text_x, text_y, str_box, ...
    'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial', ...
    'BackgroundColor', [0.97 0.98 0.98], 'EdgeColor', 'k', 'LineWidth', 1.5, ...
    'Margin', 12, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');


function out_val = hw_gelu_engine(x, frac_bits, arc_step)
    % 1. 输入数据物理截断
    x_quant = ap_fixed(x, frac_bits);
    if x_quant == 0
        out_val = 0; return;
    end
    
    % 2. 跑 SiLU LNS 主路
    abs_x = abs(x_quant);
    log2_x = hw_linear_to_log(abs_x, frac_bits);
    
    LOG2_E = ap_fixed(1.4426950408889634, frac_bits);
    exponent_term = ap_fixed(x_quant * LOG2_E, frac_bits);
    
    log2_denom = hw_lse_engine(0, -exponent_term, frac_bits);
    log2_y = ap_fixed(log2_x - log2_denom, frac_bits);
    
    raw_mag = hw_ifs_exp2(log2_y, frac_bits);
    if x_quant >= 0
        silu_base = raw_mag;
    else
        silu_base = -raw_mag;
    end
    
    % 3. 跑 ARC 残差支路 (1阶 PLA 查表)
    arc_val = get_arc_val(x_quant, arc_step, frac_bits);
    
    % 4. 汇聚输出
    out_val = ap_fixed(silu_base + arc_val, frac_bits);
end

% --- ARC 支路查表与 DSP 补偿 ---
function arc_val = get_arc_val(x, step, frac_bits)
    % 提取物理索引并定位区间
    idx = floor(x / step);
    x_start = idx * step;
    x_end = (idx + 1) * step;
    
    % 从 ROM 中读取 Base 和 Slope (模拟物理固化数据)
    y1 = true_arc(x_start);
    y2 = true_arc(x_end);
    
    base = ap_fixed(y1, frac_bits);
    slope = ap_fixed((y2 - y1) / step, frac_bits);
    
    % DSP MAC 计算
    delta_x = ap_fixed(x - x_start, frac_bits);
    arc_val = ap_fixed(base + slope * delta_x, frac_bits);
end

function y = true_arc(x)
    % 真实的理论残差公式: GELU - SiLU
    if x == 0; y = 0; return; end
    silu = x / (1 + exp(-x));
    gelu = 0.5 * x * (1 + erf(x / sqrt(2)));
    y = gelu - silu;
end

% --- LNS 底层引擎模拟 (带物理截断) ---
function res = hw_linear_to_log(val, f_bits)
    E = floor(log2(val)); 
    m = ap_fixed((val / (2^E)) - 1, f_bits); 
    idx = floor(m * 256); if idx > 255; idx = 255; end
    f_b = idx / 256; f_n = (idx + 1) / 256;
    base = ap_fixed(log2(1 + f_b), f_bits);
    slope = ap_fixed((log2(1 + f_n) - log2(1 + f_b)) * 256, f_bits);
    delta_x = ap_fixed(m - f_b, f_bits);
    res = E + ap_fixed(base + slope * delta_x, f_bits);
end

function res = hw_lse_engine(a, b, f_bits)
    max_val = max(a, b);
    abs_delta = ap_fixed(abs(a - b), f_bits);
    if abs_delta >= 31.0; res = max_val; return; end
    step = 0.125;
    idx = floor(abs_delta / step);
    b_x = idx * step; n_x = (idx + 1) * step;
    base = ap_fixed(log2(1 + 2^(-b_x)), f_bits);
    slope = ap_fixed((log2(1 + 2^(-n_x)) - log2(1 + 2^(-b_x))) / step, f_bits);
    delta_x = ap_fixed(abs_delta - b_x, f_bits);
    res = max_val + ap_fixed(base + slope * delta_x, f_bits);
end

function res = hw_ifs_exp2(val, f_bits)
    if val < -31.0; res = 0; return; end
    I = floor(val);
    F = ap_fixed(val - I, f_bits); 
    idx = floor(F * 256); if idx > 255; idx = 255; end
    f_b = idx / 256; f_n = (idx + 1) / 256;
    base = ap_fixed(2^(f_b), f_bits);
    slope = ap_fixed((2^(f_n) - 2^(f_b)) * 256, f_bits);
    delta_x = ap_fixed(F - f_b, f_bits);
    dsp_res = ap_fixed(base + slope * delta_x, f_bits);
    res = ap_fixed(dsp_res * (2^I), f_bits);
end

% --- 定点数量化函数 ---
function q_val = ap_fixed(val, frac_bits)
    res = 2^(-frac_bits);
    q_val = round(val / res) * res;
end