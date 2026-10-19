# 本机环境记录

记录时间：2026-10-19T21:08:28+08:00。

|项目|实测值|
|---|---|
|操作系统|Windows-10-10.0.26100-SP0|
|Python|3.11.9|
|虚拟环境|项目目录内 `.venv`|
|GPU|NVIDIA GeForce RTX 5060 Laptop GPU|
|CUDA 可用|True|
|GPU 显存|约8.0 GiB|
|计算能力|12.0|
|NVIDIA 驱动|610.74|
|PyTorch CUDA 运行时|12.8|

|依赖|验证版本|
|---|---|
|torch|2.9.1+cu128|
|torchvision|0.24.1+cu128|
|ultralytics|8.3.203|
|streamlit|1.50.0|
|numpy|1.26.4|
|opencv-python|4.11.0.86|
|pillow|11.3.0|

`requirements.txt` 固定直接依赖，`requirements-lock.txt` 固定全部59项运行依赖。GPU版 PyTorch 与 torchvision 从官方 CUDA 12.8 索引安装，其余包从 PyPI 安装。

已通过 `python -m pip check`、实际 CUDA 矩阵运算和 torchvision CUDA NMS 检查。CPU 与 GPU 都完成了真实权重图片推理，结果见 [复现记录](reproduction.md)。

本环境独立于系统 Python。系统原有 Python 中的 CPU 版 PyTorch 不用于启动本项目。安装与启动方法见 [quick_start.md](quick_start.md)。

版本组合依据：[PyTorch官方安装说明](https://pytorch.org/get-started/previous-versions/)、[Ultralytics安装说明](https://docs.ultralytics.com/quickstart/)。锁定文件针对本次 Windows x64 / Python 3.11 环境生成。
