# 信道编码性能分析与译码可视化

该项目提供卷积码与 LDPC 码的编码/译码仿真，并通过可视化界面展示 BER/BLER 曲线、译码收敛过程和软信息演化。

## 功能特性

- 卷积码编码 + Viterbi 译码
- LDPC 编码 + 置信传播（BP）译码
- BER/BLER 性能曲线
- 译码收敛曲线与迭代统计
- 软信息（LLR）演化可视化

## 环境要求

- Python 3.8+
- 依赖库：见 `requirements.txt`

## 快速开始

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 启动程序：

   ```bash
   python main.py
   ```

## 使用说明

- 在左侧参数面板选择编码类型（卷积码或 LDPC）、调制方式与 SNR 范围。
- 点击“开始仿真”启动后台仿真线程，进度条会显示当前状态。
- 右侧标签页展示 BER/BLER 曲线、收敛曲线、LLR 演化热图与迭代统计。
- 可通过“导出结果”将仿真数据保存为 CSV。

## 目录结构

- `encoders/`：卷积码与 LDPC 编码器
- `decoders/`：Viterbi 与 BP 译码器
- `channel/`：信道与调制模块
- `analysis/`：性能指标与统计
- `gui/`：GUI 界面与可视化
- `visualization/`：可视化辅助模块

## 备注

若在 Linux 下中文字体显示异常，可在系统中安装 `SimHei` 或 `Microsoft YaHei`，或在 `gui/main_window.py` 中修改字体配置。
