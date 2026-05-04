% =========================================================================
% UNLE Engine - SiLU (Swish) 硬件架构近似级仿真验证脚本
% =========================================================================
clear; clc; close all;

%% 1. 生成输入数据范围
x_range = linspace(-6, 6, 1000); % 测试范围 -6 到 6

%% 2. 理论计算 (标准 SiLU)
y_true = x_range ./ (1 + exp(-x_range));

%% 3. 硬件算子模拟计算 (调用自建的硬件级函数)
y_hw = zeros(size(x_range));
for i = 1:length(x_range)
    y_hw(i) = hw_silu_engine(x_range(i));
end
figure('Name', 'UNLE Engine SiLU 硬件近似度分析', 'Position', [100, 100, 900, 600], 'Color', 'w');

% 上图：函数曲线重合度对比
subplot(2, 1, 1);
plot(x_range, y_true, 'b--', 'LineWidth', 2.5); hold on;
plot(x_range, y_hw, 'r-', 'LineWidth', 1.5);
grid on;
title('SiLU 算子对比: 理论值 (Float) vs 硬件 LNS+PLA 架构 (Fixed-Point)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('输入 x', 'FontSize', 12);
ylabel('输出 SiLU(x)', 'FontSize', 12);
legend({'纯正数学理论曲线 (Dashed)', 'UNLE 硬件输出曲线 (Solid)'}, 'Location', 'northwest', 'FontSize', 12);
set(gca, 'FontSize', 11);

% 下图：硬件 PLA 近似误差分析 (绝对误差)
error_abs = abs(y_hw - y_true);
max_err = max(error_abs);           % MAX: 最大绝对误差
mse_val = mean((y_hw - y_true).^2); % MSE: 均方误差

% 获取下图的句柄
ax2 = subplot(2, 1, 2);
plot(x_range, error_abs, 'k-', 'LineWidth', 1.5); hold on;
grid on;
title('硬件架构绝对计算误差 (Absolute Error)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('输入 x', 'FontSize', 12);
ylabel('绝对误差 |y_{hw} - y_{true}|', 'FontSize', 12);

% 填充误差区域使其更具视觉冲击力 (保持你设置的颜色)
area(x_range, error_abs, 'FaceColor', [1.00 0.30 0.50], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
set(gca, 'FontSize', 11);

pos = get(ax2, 'Position'); % 获取当前位置 [left bottom width height]
pos(3) = pos(3) * 0.82;     % 宽度缩小为原来的 82%
set(ax2, 'Position', pos);  % 应用新位置

str_box = sprintf('【量化评估】\n\nMAX Error:\n%.3e\n\nMSE:\n%.3e', max_err, mse_val);
% 定位：紧贴图表右侧，高度与图表对齐
box_x = pos(1) + pos(3) + 0.03; 
box_y = pos(2);
box_w = 0.12;
box_h = pos(4);

annotation('textbox', [box_x, box_y, box_w, box_h], ...
    'String', str_box, ...
    'EdgeColor', 'k', 'LineWidth', 1.5, 'BackgroundColor', '#F8F9FA', ...
    'FontName', 'Helvetica', 'FontSize', 12, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

function y_out = hw_silu_engine(x)
    % 如果输入为 0，直接返回 0 防护
    if x == 0
        y_out = 0;
        return;
    end
    
    % 1. 提取幅值，送入 linear_to_log
    abs_x = abs(x);
    log2_x = hw_linear_to_log(abs_x);
    
    % 2. 白嫖 LSE 引擎计算 Sigmoid 分母: LSE(0, -x * log2(e))
    LOG2_E = 1.4426950408889634;
    exponent_term = x * LOG2_E; 
    log2_denominator = hw_lse_engine(0, -exponent_term);
    
    % 3. 对数域减法
    log2_y = log2_x - log2_denominator;
    
    % 4. 桶形移位器重构实数
    raw_mag = hw_ifs_exp2(log2_y);
    
    % 5. 符号恢复
    if x >= 0
        y_out = raw_mag;
    else
        y_out = -raw_mag;
    end
end

% ---------------------------------------------------------
% 子算子 1：hw_linear_to_log (256段 PLA 模拟)
% ---------------------------------------------------------
function res = hw_linear_to_log(val)
    E = floor(log2(val)); % 硬件里的 CLZ 提取整数部分
    m = (val / (2^E)) - 1; % 提取纯尾数 F 属于 [0, 1)
    
    % 模拟 LUTRAM 高 8 位查表 (划分 256 段)
    idx = floor(m * 256);
    if idx > 255; idx = 255; end
    f_base = idx / 256;
    f_next = (idx + 1) / 256;
    
    % ROM 中预存的值
    base_val = log2(1 + f_base);
    slope_val = (log2(1 + f_next) - log2(1 + f_base)) * 256;
    
    % 模拟 DSP 偏移计算
    delta_x = m - f_base;
    pla_res = base_val + slope_val * delta_x;
    
    res = E + pla_res;
end

% ---------------------------------------------------------
% 子算子 2：hw_lse_engine (高位截断，低位 PLA 模拟)
% ---------------------------------------------------------
function res = hw_lse_engine(a, b)
    max_val = max(a, b);
    abs_delta = abs(a - b);
    
    % 硬件越界保护 (大于 31 直接返回最大值)
    if abs_delta >= 31.0
        res = max_val;
        return;
    end
    
    % 模拟 LUTRAM 查表 (步长 0.125，即 1/8)
    step = 0.125;
    idx = floor(abs_delta / step);
    base_x = idx * step;
    next_x = (idx + 1) * step;
    
    % ROM 中预存的值 (log2(1 + 2^-x))
    base_val = log2(1 + 2^(-base_x));
    slope_val = (log2(1 + 2^(-next_x)) - log2(1 + 2^(-base_x))) / step;
    
    % 模拟 DSP 偏移计算
    delta_x = abs_delta - base_x;
    pla_res = base_val + slope_val * delta_x;
    
    res = max_val + pla_res;
end

% ---------------------------------------------------------
% 子算子 3：hw_ifs_exp2 (负数 Floor 补偿与桶形移位模拟)
% ---------------------------------------------------------
function res = hw_ifs_exp2(val)
    if val < -31.0
        res = 0;
        return;
    end
    
    % 硬件级向下取整 (Floor 补偿)
    I = floor(val);
    F = val - I; % F 严格落在 [0, 1)
    
    % 模拟 LUTRAM 高 8 位查表 (划分 256 段)
    idx = floor(F * 256);
    if idx > 255; idx = 255; end
    f_base = idx / 256;
    f_next = (idx + 1) / 256;
    
    % ROM 中预存的 2^F 基准值
    base_val = 2^(f_base);
    slope_val = (2^(f_next) - 2^(f_base)) * 256;
    
    % 模拟 DSP 偏移计算
    delta_x = F - f_base;
    dsp_res = base_val + slope_val * delta_x;
    
    % 模拟动态桶形移位器缩放 2^I
    res = dsp_res * (2^I);
end