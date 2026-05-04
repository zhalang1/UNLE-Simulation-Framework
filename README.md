UNLE: 面向 VLA 大模型的通用非线性计算引擎
算法-硬件协同设计仿真框架 
本项目是 UNLE (Universal Non-Linear Engine) 的黄金参考模型 (Golden Model) 与数值仿真验证框架。该引擎专门针对 $\pi_0$ 等视觉-语言-动作（VLA）模型在边缘端部署时的非线性计算瓶颈（如 Softmax, RMSNorm, SiLU, GELU）进行优化。
在进行 RTL/HLS 硬件实现之前，本项目通过 Python 和 MATLAB 建立了高精度的仿真环境，验证了通过对数数制 (LNS) 与分段线性近似 (PLA) 架构替代高能效比的指数和除法运算的可行性，确保在 FPGA 部署时实现 II=1 的全流水吞吐。
使用 PyTorch 模拟硬件底层的物理行为，支持定点数截断（Fixed-point）与块浮点（BFP）缩放。s
oftmax.py: 实现基于 Integer/Fraction Split 与 雅可比对数 (Jacobian Logarithm) 
RMSNorm.py: 验证 LNS 架构下的 RMSNorm 近似效果。
SiLU.py & GELU.py: 验证“数据通路复用”策略，即利用 SiLU 基底 + ARC 残差补偿实现高效 GELU。
SiLU.m, GELU.m, RMSNorm.m: 自动生成函数拟合曲线，量化分析 256 段 PLA 查表 下的最大绝对误差 (Max Error) 与均方误差 (MSE)。
(仿真结果仅供参考！)
