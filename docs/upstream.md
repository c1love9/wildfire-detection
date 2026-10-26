# 上游来源与本次复现范围

- 个人仓库：https://github.com/c1love9/wildfire-detection
- 上游仓库：https://github.com/AlimTleuliyev/wildfire-detection
- 起始提交：`a44e1ecdb84b06bda145b1c3ad53854046fe14dd`
- 上游项目许可证：MIT，根目录的原始 LICENSE 保留。
- 上游代码已有能力：YOLOv8 火烟图片检测、通用检测、图片与视频演示页面。
- 上游视频页面播放已有视频，尚未提供实时视频检测与报警系统。

初始复现阶段配置独立 Python 环境、修正兼容问题并验证上述功能。随后增加了 [逐帧视频火烟检测](video_detection.md)，将视频页接入真实推理和开始、暂停、继续、停止控制，并自行实现 [连续确认、分级报警和SQLite日志](alarm.md)、[报警截图与前后片段回放](recording.md)。默认上游火烟权重保留，未生成课题的正式准确率指标。

随后自行增加 [本机摄像头实时输入](camera.md)，以最新帧、主机单调计时、设备超时关闭和独立观测段连接同一报警／媒体流程；上游原项目没有这部分实时采集与生命周期管理。

再增加 [模拟监控点位置与视角](monitoring.md)、地图方向图层、报警关联点与历史快照/CSV，并完成 [题目要求对照](requirements_audit.md)。这是自行增加的展示功能，不包含真实GPS定位或火源地理投影。

随后增加 [数据采集与管理](dataset.md)：自行实现批次/来源/场景清单、图片像素去重与视频字节去重、USB原画采集、视频PTS抽帧、声明数量统计、关联组划分和不可覆盖的版本清单。未将上游演示素材计为实习自采数据。

随后增加 [GroundingDINO预标注与复核](annotations.md)。模型通过Hugging Face Transformers运行官方固定版本；自行实现候选版本、无外部前端依赖的原图坐标画布、追加式复核/差异记录、并发保存检查、负样本确认和YOLO导出。GroundingDINO权重未微调，公开样例和自动化模拟复核仅验证功能。

随后增加 [YOLO训练与独立评估](training.md)：基于Ultralytics固定8.3.203训练/验证接口，自行实现复核版本校验、数据与初始权重快照、独立后台任务/状态、参数与环境记录、曲线/权重下载及指标协议。使用上游COCO预训练YOLOv8n初始化新的两类模型，合成素材训练只用于功能验证。

本轮接入 [自训模型检测与切换](model_inference.md)：自行实现成功训练模型清单、内容/设备区分的受控缓存、图片/视频/摄像头统一加锁推理、停止后切换和逐次模型日志快照。保留默认上游和General功能，兼容上游/自训不同类别编号。未把隔离的合成验证模型导入主数据目录作为正式成果。

本机先取回 `fire_n.pt`、`yolov8n.pt`、演示图片与四段演示视频；较大模型和 README 动图按需取回。稀疏检出仅影响本地下载内容，不删除仓库中的已有模型和媒体。

## 依赖与材料来源

2026-10-20至21日按后续计划增加SAM ViT-B框提示分割、原图并集占比/持久化趋势、连续区域面积辅助分级、去重弹窗/可选声音、绑定视频/USB的命名通道，以及YOLO/DINO统一冻结test评测、固定尺寸/多尺度控制实验、双人独立标注/仲裁/掩码误差与受控计时工具。自行实现统一结果、状态/快照、评测与审计层，底层模型沿用官方固定版本，未微调SAM或GroundingDINO。协议与边界见 [completion.md](completion.md)。教师模板版报告、实际演示与清单见 [delivery.md](delivery.md)。

|组件|用途|来源|
|---|---|---|
|Ultralytics|加载与运行YOLOv8模型|https://github.com/ultralytics/ultralytics|
|PyTorch / torchvision|CPU与CUDA计算、检测算子|https://pytorch.org/get-started/previous-versions/|
|Streamlit|原项目网页界面|https://streamlit.io/|
|OpenCV|图像转换与视频解码|https://opencv.org/|
|D-Fire|上游训练所用数据集；本阶段未重新下载训练数据|https://github.com/gaia-solutions-on-demand/DFireDataset|
|GroundingDINO Tiny|官方开放词汇模型，按英文提示词生成fire/smoke候选框|https://huggingface.co/IDEA-Research/grounding-dino-tiny|
|Transformers|本机GroundingDINO处理器与推理，固定4.57.1|https://huggingface.co/docs/transformers/model_doc/grounding-dino|
|SAM ViT-B|同帧检测框提示掩码；固定revision和SHA-256，未微调|https://huggingface.co/facebook/sam-vit-base|

各依赖和数据沿用各自的许可与使用条件，根目录的 MIT 许可证描述上游项目代码。原模型类别及权重校验值由实测记录 `reports/reproduction.json` 给出。

2026-10-26按原题简化质量工作区：删除逐图仲裁界面和写入接口，A/B原始标注分别导出，质量报告直接使用两份独立参考。旧参考历史保留但不参与门槛或指标。
