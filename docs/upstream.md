# 上游来源与本次复现范围

- 个人仓库：https://github.com/c1love9/wildfire-detection
- 上游仓库：https://github.com/AlimTleuliyev/wildfire-detection
- 起始提交：`a44e1ecdb84b06bda145b1c3ad53854046fe14dd`
- 上游项目许可证：MIT，根目录的原始 LICENSE 保留。
- 上游代码已有能力：YOLOv8 火烟图片检测、通用检测、图片与视频演示页面。
- 上游视频页面播放已有视频，尚未提供实时视频检测与报警系统。

初始复现阶段配置独立 Python 环境、修正兼容问题并验证上述功能。随后增加了 [逐帧视频火烟检测](video_detection.md)，将视频页接入真实推理和开始、暂停、继续、停止控制，并自行实现 [连续确认、分级报警和SQLite日志](alarm.md)、[报警截图与前后片段回放](recording.md)。所用火烟权重为上游预训练权重，当前没有训练自己的模型，也没有生成课题的正式准确率指标。

本机先取回 `fire_n.pt`、`yolov8n.pt`、演示图片与四段演示视频；较大模型和 README 动图按需取回。稀疏检出仅影响本地下载内容，不删除仓库中的已有模型和媒体。

## 依赖与材料来源

|组件|用途|来源|
|---|---|---|
|Ultralytics|加载与运行YOLOv8模型|https://github.com/ultralytics/ultralytics|
|PyTorch / torchvision|CPU与CUDA计算、检测算子|https://pytorch.org/get-started/previous-versions/|
|Streamlit|原项目网页界面|https://streamlit.io/|
|OpenCV|图像转换与视频解码|https://opencv.org/|
|D-Fire|上游训练所用数据集；本阶段未重新下载训练数据|https://github.com/gaia-solutions-on-demand/DFireDataset|

各依赖和数据沿用各自的许可与使用条件，根目录的 MIT 许可证描述上游项目代码。原模型类别及权重校验值由实测记录 `reports/reproduction.json` 给出。
