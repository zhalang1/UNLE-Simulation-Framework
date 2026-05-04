% =========================================================================
% RMSNorm 基于 LNS+PLA 架构的定点近似仿真与对比
% =========================================================================
clear; clc; close all;

%% 1. 硬件架构参数配置
int_bits = 8;          % 整数位宽
frac_bits = 8;         % 小数位宽
LUT_SIZE = 256;        % PLA 查表段数
res = 2^(-frac_bits);  % 物理分辨率
max_val = 2^(int_bits-1) - res;
min_val = -2^(int_bits-1);

% 定点截断与饱和保护匿名函数
ap_fixed = @(v) max(min(round(v / res) * res, max_val), min_val);

%% 2. 离线生成 PLA 查表参数 (ROM)
idx = 0:(LUT_SIZE-1);

% Exp2 (2^x) 表
x1_e = idx / LUT_SIZE;
x2_e = (idx + 1) / LUT_SIZE;
exp2_base = 2.^x1_e;
exp2_slope = (2.^x2_e - 2.^x1_e) * LUT_SIZE;

% Log2 (log2(1+x)) 表
x1_l = idx / LUT_SIZE;
x2_l = (idx + 1) / LUT_SIZE;
log2_base = log2(1 + x1_l);
log2_slope = (log2(1 + x2_l) - log2(1 + x1_l)) * LUT_SIZE;

%% 3. 压测环境配置
D = 896;                     % Qwen2 模型维度
epsilon = 1e-6;              % 极小值
eps_hw = epsilon * D;        % 硬件预缩放 epsilon
bg_variance = 1.0;           % 背景特征方差预设
bg_sq_sum = D * bg_variance; % 背景平方和

% 测试序列 (扫描单一维度的输入响应)
x_test = linspace(-10, 10, 5000);
y_pure = zeros(size(x_test));
y_hw   = zeros(size(x_test));

%% 4. 执行算子级仿真
% 预计算常数项 D 的对数
log_N = log2_hw(D, LUT_SIZE, log2_base, log2_slope, ap_fixed);

for i = 1:length(x_test)
    x = x_test(i);
    
    % 模拟总平方和累加
    sq_sum = bg_sq_sum + x^2;
    
    % ----------------------------------------------------
    % [纯正 Float64 浮点通路]
    % ----------------------------------------------------
    y_pure(i) = x / sqrt(sq_sum / D + epsilon);
    
    % ----------------------------------------------------
    % [硬件 LNS+PLA 定点近似通路]
    % ----------------------------------------------------
    if abs(x) == 0
        y_hw(i) = 0;
        continue;
    end
    
    % 1. 计算分母对数
    log_S = log2_hw(sq_sum + eps_hw, LUT_SIZE, log2_base, log2_slope, ap_fixed);
    
    % 2. 减法替代除法，乘 0.5 替代开方
    diff = ap_fixed(log_N - log_S);
    global_offset = ap_fixed(diff * 0.5);
    
    % 3. 输入转对数域并加上偏移量
    log_O = log2_hw(abs(x), LUT_SIZE, log2_base, log2_slope, ap_fixed);
    log_final = ap_fixed(log_O + global_offset);
    
    % 4. 逆对数还原并恢复符号
    final_abs = exp2_hw(log_final, LUT_SIZE, exp2_base, exp2_slope, ap_fixed);
    y_hw(i) = sign(x) * final_abs;
end

%% 5. 误差计算与可视化
abs_err = abs(y_pure - y_hw);
mse_val = mean(abs_err.^2);
max_err = max(abs_err);

figure('Position', [100, 100, 800, 600], 'Color', 'w');

% 上图：响应曲线对比
subplot(2, 1, 1);
plot(x_test, y_pure, '--b', 'LineWidth', 2); hold on;
plot(x_test, y_hw, '-r', 'LineWidth', 1.5);
grid on;
title('RMSNorm 算子响应：纯正浮点理论值 vs 定点 LNS+PLA 硬件架构', 'FontSize', 12);
xlabel('输入 x', 'FontSize', 10);
ylabel('输出 O_i', 'FontSize', 10);
legend('理论值 (Float64 Dashed)', '近似值 (Fixed-Point Solid)', 'Location', 'northwest');
set(gca, 'FontSize', 10);

% 下图：绝对误差
subplot(2, 1, 2);
area(x_test, abs_err, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeAlpha', 0.5);
grid on;
title('定点硬件架构绝对计算误差 (Absolute Error)', 'FontSize', 12);
xlabel('输入 x', 'FontSize', 10);
ylabel('| y_{pure} - y_{hw} |', 'FontSize', 10);
set(gca, 'FontSize', 10);

% 绘制误差指标信息框
% 绘制误差指标信息框 (将文本框移至底部子图的中上部空白处)
bbox = [0.48, 0.3, 0.16, 0.12]; % [x位置, y位置, 宽度, 高度]
info_str = sprintf('【量化评估】\n\nMAX Error:\n%.4e\n\nMSE:\n%.4e', max_err, mse_val);
annotation('textbox', bbox, 'String', info_str, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

%% =========================================================================
% 核心子算子函数定义
% =========================================================================

% 硬件 Linear to Log2 转换子算子
function res = log2_hw(x, LUT_SIZE, base_tbl, slope_tbl, ap_fixed_func)
    E = floor(log2(x));
    f = ap_fixed_func(x / (2^E) - 1.0);
    
    idx = floor(f * LUT_SIZE);
    idx = min(max(idx, 0), LUT_SIZE - 1);
    delta_x = ap_fixed_func(f - idx / LUT_SIZE);
    
    base = ap_fixed_func(base_tbl(idx + 1));
    slope = ap_fixed_func(slope_tbl(idx + 1));
    
    pla_res = ap_fixed_func(base + slope * delta_x);
    res = ap_fixed_func(E + pla_res);
end

% 硬件 Log2 to Linear (2^x) 转换子算子
function res = exp2_hw(x, LUT_SIZE, base_tbl, slope_tbl, ap_fixed_func)
    I = floor(x);
    F = ap_fixed_func(x - I);
    
    idx = floor(F * LUT_SIZE);
    idx = min(max(idx, 0), LUT_SIZE - 1);
    delta_x = ap_fixed_func(F - idx / LUT_SIZE);
    
    base = ap_fixed_func(base_tbl(idx + 1));
    slope = ap_fixed_func(slope_tbl(idx + 1));
    
    pla_res = ap_fixed_func(base + slope * delta_x);
    res = ap_fixed_func(pla_res * (2^I));
end