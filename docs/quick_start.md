# 原项目本地运行

当前支持YOLOv8图片检测、逐帧视频火烟检测、连续确认、分级报警与本机报警日志。数据采集、自行训练、摄像头接入和片段回放按后续实施计划推进。

## 已配置电脑

项目目录：`D:\GPTWORK\fire-alarm-system`。

双击项目根目录的 `启动项目.cmd`，保持启动窗口运行，浏览器打开 <http://127.0.0.1:8501>。关闭服务时在启动窗口按 Ctrl+C。

首页默认使用 `Fire Detection`、`fire_n` 和一幅内置图片。有可用 CUDA 时默认使用 GPU，也可在 `Inference Device` 中切换 CPU。图片输入支持内置样例、网址和本地上传；推理后可下载带框图片。`General` 使用 COCO 通用检测模型，不能代替火烟模型。

侧边栏 `Demo Images` 展示图片。进入 `Demo Videos`，选择内置视频或上传MP4/MOV/AVI/MKV，点击“开始检测”即可看到逐帧的火焰／烟雾框、置信度、帧号、视频时间和当前帧数量。支持暂停、继续和停止；运行期间先停止再调整参数。详细操作与验证结果见 [video_detection.md](video_detection.md)。连续确认后显示关注、预警或报警；在 `Alarm History`可筛选事件、查看等级变化并导出CSV，规则和日志保存说明见 [alarm.md](alarm.md)。

## 在新电脑重新配置

已验证的环境目标为 Windows x64、Python 3.11、NVIDIA CUDA 显卡。先安装 Python 3.11 与合适的显卡驱动，然后在项目目录运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

上述命令在项目内创建 `.venv`，按 `requirements-lock.txt` 安装并检查依赖。首次安装 GPU 运行库会下载数 GB 文件，需保留充足磁盘空间。也提供 `scripts/setup.ps1` 将这些步骤串联执行。

启动：

```powershell
.\启动项目.cmd
```

如果8501端口被占用，可以指定其他端口：

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\1_🔥_Home.py --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

本阶段直接依赖版本写在 `requirements.txt`，全部已安装包版本写在 `requirements-lock.txt`。PyTorch 与 torchvision 使用官方 CUDA 12.8 轮子，不能只复制 `.venv` 文件夹到另一台电脑。

## 模型与演示素材

启动本阶段页面至少需要：

- `fire-models/fire_n.pt`
- `general-models/yolov8n.pt`
- `demo-images/` 中的图片；画廊和视频页分别读取 `demo-images/`、`demo-videos/`。

这些文件来自上游仓库，当前 Fork 中原样保留。普通完整克隆会取得全部文件。本机采用稀疏检出，暂未下载更大的 m/s/l 模型和说明文档动图。若需要恢复完整内容，在网络与磁盘条件允许时运行 `git sparse-checkout disable`；该操作可能继续下载较大的模型和媒体。

模型和原项目来源见 [upstream.md](upstream.md)，本机实际加载的类别与 SHA-256 校验值见 [reproduction.json](../reports/reproduction.json)。上传的视频保存在本机 `artifacts/uploads/` 中，不进入版本库。

## 重新执行复现检查

在项目目录运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_reproduction.py
```

检查内容包括真实权重加载、全部内置图片、火焰／烟雾样例的 CPU 与 CUDA 推理、合成空背景、通用模型、四段视频解码、Streamlit 首页和图片／视频页面。检查需要已有模型和演示文件，使用本地 HTTP 服务验证图片网址输入，不依赖外部图片网站。上游的一些样例图片已经带有检测框，复现结果用于验证流程，正式评测需使用独立的原始素材和人工参考标签。

机器可读结果写入 `reports/reproduction.json`；带框图片写入 `artifacts/reproduction/`。后者属于本机生成的临时结果，已从版本库排除。合成空背景仅用于功能检查，不计入课题的正常场景数据，也不用于证明模型准确率。

验证完整视频检测、暂停／继续、停止和结束处理：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_video.py
```

该检查会逐帧处理四段样例，结果写入 `reports/video_detection.json`。视频页将结果动态显示；当前未导出整段带框视频。

验证连续确认、分级、日志、异常及历史页面：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_alarm.py
```

报警默认参数在 `configs/alarm.json`，实际使用日志保存在 `artifacts/alarms/alarms.sqlite3`，重启后保留且不进入版本库。报警验证使用独立测试数据库，摘要写入 `reports/alarm_verification.json`。

## 常见情况

|情况|处理|
|---|---|
|找不到 Python 3.11|安装 Python 3.11，确认 `py -3.11 --version` 能显示版本。|
|没有 `.venv` 或缺少依赖|先运行 `scripts/setup.ps1`，再启动项目。|
|模型列表为空|检查对应目录的 `.pt` 文件是否存在；稀疏克隆需要取回模型文件。|
|没有 GPU 选项|先运行 `nvidia-smi` 检查驱动，再核对虚拟环境的 PyTorch 是否带 CUDA；CPU 模式仍可用于图片演示。|
|首次推理稍慢|模型加载与 GPU 初始化需要时间；后续模型加载由页面缓存复用。|
|上传文件无法识别|使用实际内容为 PNG/JPEG 的图片，避免仅修改扩展名。|

PyTorch 安装组合参考：[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。环境和复现的实测结论见 [reproduction.md](reproduction.md)。
