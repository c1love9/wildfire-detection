# YOLO模型训练与评估

侧边栏进入 `Training`。数据准备顺序为 **Dataset采集/导入与分组划分 → Annotations人工复核、导出完整训练目录 → Training训练 → 单独评估**。标签ZIP不含图像，不能直接作为训练输入。

## 操作步骤

|步骤|具体操作|保存结果|
|---|---|---|
|1. 准备数据版本|在Annotations的YOLO导出生成完整训练数据集。训练、验证、测试都须有图片，训练/验证都至少有一个已复核目标；建议每个集合覆盖fire、smoke和正常负样本|固定原图、标签、人工版本和按来源隔离的train/val/test清单|
|2. 新建训练|进入Training，填写执行者与实验说明，选择已复核YOLO数据集|独立任务编号；不会覆盖既有任务或视频报警权重|
|3. 设置参数|默认YOLOv8n COCO预训练初始化、50轮、640尺寸、batch=8、AdamW、lr0=0.001、早停15、种子42；选择GPU或CPU|request.json记录请求值，effective_parameters.json和Ultralytics args.yaml记录实际参数|
|4. 开始与观察|点击开始训练。查看任务状态、已完成轮次、训练/验证损失、验证P/R与mAP曲线；日志可展开查看|每轮results.csv；结束后的results.png、最佳与末轮权重；环境、数据校验与结果JSON|
|5. 切页/重启/取消|后台进程独立于网页服务。切页或关闭网页不影响任务；重启服务后仍可查看。取消会在下一训练/验证批次检查点退出|保留已有轮次和日志，状态标为取消，不当作成功训练；意外退出标为中断，可新建任务重试|
|6. 独立评估|训练成功后选择“模型评估”，选择训练任务best.pt及val或test，设置尺寸、batch、设备、最低置信度与NMS IoU，再开始|独立评估编号；保存Precision/Recall/F1、mAP@0.5、mAP@0.5:0.95、各类别支持数与计时，以及PR/F1/混淆矩阵等实际生成图|
|7. 下载与备份|在任务记录中选择CSV、曲线、参数、环境、指标、数据凭据或best/last.pt下载；大文件也可直接使用本机任务目录|文件保存在数据根目录training下，随整个数据根目录备份；不随Git上传|

同一数据目录同时只运行一个训练或评估任务。设备繁忙或显存不足时保留失败原因与日志，减少batch/尺寸后新建任务。当前支持单GPU编号0和CPU；固定workers=0、AMP关闭、FP32与deterministic=True，减少Windows多进程及额外模型下载问题。所有默认增强、学习率衰减、实际优化器等完整参数仍由args.yaml和effective_parameters.json记录。固定种子不能保证跨硬件/驱动完全逐位相同。

默认使用 `general-models/yolov8n.pt` 的COCO预训练权重，保存其SHA-256；新的两类训练模型固定0=fire、1=smoke。没有把已有上游火烟权重当作自己的训练成果，也没有自动替换报警模型。当前不提供断点续训；取消/中断后的文件用于追溯，新实验重新创建。

成功训练任务的best.pt现可在Home图片页、Demo Videos视频/摄像头页直接选择使用，支持切回上游模型；无需手工复制或覆盖权重。运行/暂停期间锁定模型，停止后切换，日志记录来源与哈希，详见 [model_inference.md](model_inference.md)。

## 数据与结果完整性

训练开始前对照本机数据库核对导出清单、人工批准版本、YOLO标签、原图哈希、来源划分快照、组隔离及data.yaml。不能通过手改TXT、审核JSON或路径把未复核数据混进训练。检测到额外图片、标签缺失/改变、损坏图像或外部路径时失败并记录原因。

每次训练复制一份独立input快照和初始权重，记录逐文件校验值；训练结束后再次核对。以后修改原导出或重新复核不影响已开始任务。可以选择此前已批准的历史导出版本，其人工版本被明确固定；如需采用更新后的标签，应重新导出并新建训练。运行中的input目录不要手工编辑。

评估只接受成功训练的best.pt，核对完成记录、数据凭据和权重哈希，再使用该训练的数据快照；不从其他版本随意换测试数据。结果记录的SHA-256同时保存在数据库中，后续编辑结果或凭据会被检测。Windows后台进程使用隐藏窗口与独立进程作业启动；系统若禁止进程脱离，则新建任务会明确报错，不声称继续运行。

```text
data/fire_dataset/training/
  train_<id>/
    request.json                  # 执行者、说明、请求参数、初始模型来源/哈希
    environment.json              # Python、包版本、系统/GPU、代码版本与代码哈希
    initial.pt                    # 此次初始权重副本
    input/                        # 已复核图像、标签、data.yaml和修改记录快照
    dataset_receipt.json           # 逐文件校验、人工版本、组划分与各集合数量
    effective_parameters.json
    worker.log
    output/
      args.yaml
      results.csv
      results.png
      weights/{best,last}.pt
      ...                         # 实际生成的PR/F1曲线、混淆矩阵及样例图
    metrics.json                  # 最佳权重在val上的最终验证结果
    result.json                   # 完成状态、曲线/权重校验值
  eval_<id>/
    checkpoint.pt                 # 被评估best.pt副本
    request.json
    environment.json
    dataset_receipt.json
    effective_parameters.json
    worker.log
    output/                       # 评估曲线/混淆矩阵等
    metrics.json
    result.json
```

训练与评估索引、状态、进程编号及完成校验值保存在同一catalog.sqlite3新增的training_jobs表。默认数据根目录为 `data/fire_dataset/`，可用 `FIRE_DATASET_ROOT` 更改；整个目录被Git忽略。重启是重启服务进程，关机/系统重启后计算不能继续，已保存记录仍在，活跃任务会标记中断。备份需等任务结束、关闭服务后复制整个数据根目录。

## 指标口径

每轮只使用train更新参数、val选择best.pt，选择指标为验证mAP@0.5:0.95；训练过程不计算test指标。最终方案确定后再显式评估test，反复根据test调参会造成测试泄漏。早停可能使实际轮数少于请求值，两者均记录。

mAP按IoU 0.50至0.95、步长0.05计算；评估NMS IoU（默认0.7）是另一个参数。默认最低置信度0.001用于保留PR/AP曲线候选，不是部署报警阈值。Ultralytics 8.3.203的P/R取该评估集合最大平均F1点、匹配IoU=0.5；不是预先固定置信度下的部署效果。整体F1为有GT类别的逐类F1平均，不能简单用宏平均P/R重新计算。无GT类别显示null，整体仅平均有GT的类别；有GT但没有正确预测时显示0。少量或缺类数据不构成可靠性能评估。

速度保存预处理、推理、后处理每图毫秒数和相应吞吐率，记录设备、FP32、尺寸与batch；验证器已预热，计时不含磁盘、页面或网络。不能直接当作摄像头端到端延迟，也不能把batch>1吞吐率当作单请求延迟。训练验证和单独评估可能使用不同batch，做公平对比时应统一设置。

实现依据：[Ultralytics训练文档](https://docs.ultralytics.com/modes/train/)、[评估文档](https://docs.ultralytics.com/modes/val/)，以本项目固定8.3.203版本实测接口为准。Polars固定1.34.0；本机原2.0.0的CPU特征检查会导致保存检查点失败，已通过固定版本解决。

## 验证与尚未完成项

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe scripts\verify_training.py
```

Python验证使用独立合成图像、模拟复核记录及分组清单，真实运行GPU两轮训练、CPU一轮训练和CPU上的独立test评估；检查参数、损失/mAP CSV、曲线、最佳/末轮权重、骨干参数实际改变与类别映射。报告见 [training_verification.json](../reports/training_verification.json)。

浏览器验证使用该报告中的独立browser_root作为 `FIRE_DATASET_ROOT`，另设独立 `FIRE_ALARM_LOG_PATH`，启动8502服务。用本机Playwright/Edge执行 `scripts/verify_training_browser.mjs`（`PLAYWRIGHT_MODULE`可指定模块、`ALARM_APP_URL`可指定服务）。它实际发起训练/评估、显示曲线并下载文件，然后保留长任务供外部重启；同目录重启服务后加 `--history-only`，核验同一工作进程继续、取消生效和历史指标字节一致。报告见 [training_browser.json](../reports/training_browser.json)。完整重跑前先重新执行Python验证，以得到空的独立浏览器目录。

本机IDM可能接管原生下载。设置 `TRAINING_ALLOW_DOWNLOAD_FALLBACK=1` 时，测试会明确记录取消，再读取同一下载地址的真实HTTP 200文件核对字节；不将其冒充原生下载通过，不修改系统下载设置。用户也可从页面显示的本机路径取文件。

这次短训练验证的是功能，不是正式火灾场景精度实验。题目要求的自采数量、至少200幅真实双人独立标注、预标注质量/人工工时评估、真实数据基线收敛训练、独立测试与优化对照仍须完成，详见 [requirements_audit.md](requirements_audit.md)。合成图形不能计为自采火灾图像，模拟审核不能计为人工GT。
