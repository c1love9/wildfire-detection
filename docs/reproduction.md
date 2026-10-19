# 原项目复现记录

本页保留初始图片与演示页面复现记录。当前视频页的逐帧检测使用与验证见 [video_detection.md](video_detection.md)，`reports/reproduction.json` 会在回归检查后更新为最新执行记录。

执行时间：2026-10-19T21:08:28+08:00。状态：**通过**。

本阶段使用上游已有 `fire_n.pt` 与 `yolov8n.pt`，完成环境和原页面复现。正式训练、独立测试集准确率、实时视频检测和分级报警属于后续阶段。

## 实测结果

|检查|结果|
|---|---|
|独立环境依赖检查|`pip check` 通过|
|GPU计算|CUDA矩阵运算与CUDA NMS通过|
|火烟类别|`0=smoke`，`1=fire`|
|全部内置图片|20幅均完成推理|
|CPU／GPU真实权重推理|两种设备均完成火烟样例处理，输出框坐标有效|
|合成空背景|检出0个目标；仅作为功能负向检查|
|通用模型|80类模型成功加载并推理|
|视频解码|4段上游视频均能读取首帧|
|实际网页服务|本机 `http://127.0.0.1:8501`，健康检查返回 `ok`|
|真实浏览器|Edge 155.0.4283.62，首页、实际上传、PNG下载、损坏图片提示均通过，无页面脚本异常|

设备、依赖版本见 [environment.md](environment.md)。完整记录见 [reproduction.json](../reports/reproduction.json) 和 [browser_reproduction.json](../reports/browser_reproduction.json)。

## 界面流程检查

|流程|结果|
|---|---|
|Home: default nano model and built-in image|通过|
|Home: CPU image prediction|通过|
|Home: general-model selector|通过|
|Home: image URL served locally|通过|
|Home: upload selector with no file|通过|
|Page: 2_📸_Demo Images.py|通过|
|Page: 3_📹_Demo Videos.py|通过|

上述7项通过 Streamlit AppTest 执行。另用实际浏览器上传 `demo-images/image1.jpeg` 并下载 `prediction.png`；下载文件为有效PNG，共1121757字节。随后上传损坏图片，确认界面显示错误信息而没有中断应用。

## CPU与GPU样例

|设备|样例|检出数量|
|---|---|---|
|cpu|demo-images/image0.jpeg|{"fire": 1, "smoke": 2}|
|cpu|demo-images/image8.jpeg|{"smoke": 1}|
|cuda:0|demo-images/image0.jpeg|{"fire": 1, "smoke": 2}|
|cuda:0|demo-images/image8.jpeg|{"smoke": 1}|

这些数量记录模型输出，不代表人工核对后的准确率。上游部分图片已有框或文字；全部图片均为演示素材，合成空背景也不计入课题的自采正常数据。正式评测需使用独立的原始图像与人工参考标签。

## 本次兼容调整

- 将上游 `ultralytics==8.0.136`、`streamlit==1.24.0` 更新为本次实测版本，选用支持当前GPU的CUDA版PyTorch。
- 使用脚本所在目录定位模型和媒体，从其他工作目录运行复现检查也能通过。
- 默认优先选择nano模型，允许只有一个权重文件时正常启动。
- 加入CPU／GPU选择、内置图片入口和缺少权重时的提示。
- 更新图片显示接口；为网址请求增加超时，处理无法识别的上传图片。
- 修正置信度和NMS阈值说明、下载文件名，保留原作者署名和许可证。
- 新增锁定依赖、Windows启动入口、复现脚本与说明；数据、运行输出和密钥文件从提交中排除。

## 重复验证

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_reproduction.py
```

脚本输出复现JSON与本机带框图片。浏览器可按“打开首页 → 选择 Upload from Computer → 上传一张JPEG → 点击 Download Prediction”手工复查。

本次本机截图与图片结果位于 `artifacts/reproduction/`，已排除在版本库外。视频演示页目前播放已有片段，实时逐帧检测将按后续执行方案新增。
