# 原项目本地运行

逐页操作、完整训练与质量工作流、备份和常见故障处理见 [完整使用指南](user_guide.md)。

当前已完成检测、可选SAM分割、面积趋势、面积辅助报警、弹窗/可选声音、通道与历史回放，以及数据、标注、训练、统一对比、优化实验和质量工具。状态为“系统功能、实验工具和材料框架完成”；实际数据数量、200幅真人双人标注及正式训练/测试仍待完成，见 [第四题对照表](requirements_audit.md)。

## 已配置电脑

项目目录：`D:\GPTWORK\fire-alarm-system`。

双击项目根目录的 `启动项目.cmd`，保持启动窗口运行，浏览器打开 <http://127.0.0.1:8501>。关闭服务时在启动窗口按 Ctrl+C。

首页默认使用 `Fire Detection`、`fire_n` 和一幅内置图片。有可用 CUDA 时默认使用 GPU，也可在 `Inference Device` 中切换 CPU。图片输入支持内置样例、网址和本地上传；推理后可下载带框图片。`General` 使用 COCO 通用检测模型，不能代替火烟模型。

成功训练的best.pt自动加入Home和Demo Videos的模型列表，显示“自训 · 实验说明 · 任务尾号 · best”。两页可分别选择模型与尺寸；视频/摄像头运行或暂停时模型锁定，停止后切换并重新开始。日志保留每次模型及校验快照。操作与验证范围见 [model_inference.md](model_inference.md)。主数据目录暂无正式自训任务时，完成Training训练后才出现相应选项。

侧边栏 `Demo Images` 展示图片。进入 `Demo Videos`，选择内置视频或上传MP4/MOV/AVI/MKV，点击“开始检测”即可看到逐帧的火焰／烟雾框、置信度、帧号、视频时间和当前帧数量。支持暂停、继续和停止；运行期间先停止再调整参数。详细操作与验证结果见 [video_detection.md](video_detection.md)。连续确认后显示关注、预警或报警；在 `Alarm History`可筛选事件、查看等级变化并导出CSV，规则和日志保存说明见 [alarm.md](alarm.md)。

## 在新电脑重新配置

开启像素分割前，运行 `.venv\Scripts\python.exe scripts\prepare_sam.py` 下载并校验固定版本SAM ViT-B。GroundingDINO使用 `.venv\Scripts\python.exe scripts\prepare_grounding_dino.py` 准备。权重均在忽略目录，不随本轮提交上传。默认不启用分割，纯检测不依赖SAM下载。

新入口：`Channels`编辑命名通道；`Comparison`并排检测/冻结test评测；`Experiments`成对训练及报警策略对照；`Quality`独立标注、人工参考掩码和受控工时。详细规则、未知值处理与统一指标口径见 [completion.md](completion.md)，连贯演示见 [delivery.md](delivery.md)。

检测前可设置“监控点位置与视角（模拟）”，选择A/B预设或修改经纬度和角度；停止后可修改。报警历史会回溯事件发生时的位置与视角，详见 [monitoring.md](monitoring.md)。

使用本机摄像头时，在 `Demo Videos` 的“视频来源”选择“摄像头”，默认编号0，点击开始。暂停会关闭设备并结束本观测段；继续会重新打开并重新确认，历史与已有片段保留。具体行为、权限／占用排查和验证范围见 [camera.md](camera.md)。

报警片段编码还需要本机安装支持libx264的FFmpeg并加入PATH。本机已验证FFmpeg 8.0.1，新电脑安装与检查见 [recording.md](recording.md)；没有编码器时检测与截图仍可用，片段会显示失败提示。

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

准备自己的数据时进入 `Dataset`，先建批次并填写真实来源和采集组，再上传图片/视频、导入本机文件夹或使用USB采集。可抽帧、导出统计和按组划分70/20/10，说明见 [dataset.md](dataset.md)。素材、数据库和划分保存在本机 `data/fire_dataset/`，不随Git提交；验证脚本使用独立目录。

本机GroundingDINO模型已准备并校验。进入 `Annotations`，填写执行者，依次进行“自动预标注→人工复核→YOLO导出”。新电脑安装依赖后先准备模型（约658MiB权重，首次需访问Hugging Face）：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_grounding_dino.py
```

模型文件保存在忽略的 `artifacts/models/`；页面推理只读取本机文件。画框、修改记录、负样本确认、训练目录使用和验证步骤见 [annotations.md](annotations.md)。预标注模型与首页上游YOLO权重用途不同，均未在本项目新数据上完成微调。

完成复核并导出完整YOLO训练数据集后，进入 `Training` 填写实验说明、设置轮数/尺寸/batch/设备并开始训练。后台保存独立数据快照、参数、损失与mAP曲线、best/last权重；训练成功后可单独评估val/test。操作、指标口径、重启/取消与验证范围见 [training.md](training.md)。本机已验证GPU/CPU短训练，正式数据训练和准确率验收仍待完成。

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

媒体默认在首次确认时保存截图，并收集确认前5秒、后10秒的画面。进入 `Alarm History`选择事件，即可查看截图与回放片段；不足部分会标明。媒体文件默认位于 `artifacts/alarms/media/`，参数在 `configs/recording.json`。完整操作与资源上限见 [recording.md](recording.md)。验证：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_recording.py
```

验证本机摄像头、受控报警媒体、文件回归与摄像头页面：

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_camera.py
```

该检查会打开本机摄像头，实拍环节不保存个人画面；公开样例与模拟时间用于报警媒体验证。没有设备时可显式加 `--skip-hardware`，报告会区分未做硬件验收。

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
