# GroundingDINO预标注、人工复核与YOLO导出

侧边栏进入 `Annotations`，依次选择“自动预标注”“人工复核”“YOLO导出”。本阶段连接已有Dataset归档和分组划分，不改变视频报警使用的上游YOLO权重。

## 模型准备

本机已下载、校验并实际运行官方 [GroundingDINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)，使用 [Transformers实现](https://huggingface.co/docs/transformers/model_doc/grounding-dino)。新电脑先按 [quick_start.md](quick_start.md) 安装固定依赖，再执行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_grounding_dino.py
```

脚本从官方模型仓库的固定提交下载7个必要文件，核验官方Git/LFS校验值，再记录本地各文件SHA-256。权重约658MiB；模型目录为 `artifacts/models/grounding-dino-tiny/<revision>/`，不提交Git。页面只读取本机模型，不会在打开页面时自动下载或把图片上传推理服务。可用 `GROUNDING_DINO_MODEL_DIR` 指向包含完整文件和 `model_receipt.json` 的同一已校验版本目录；准备脚本仍写默认目录。

|项目|固定设置|
|---|---|
|模型|IDEA-Research/grounding-dino-tiny，未微调|
|提交|a2bb814dd30d776dcf7e30523b00659f4f141c71|
|权重SHA-256|1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3|
|计算|优先CUDA，可选CPU；FP32|
|依赖|Transformers 4.57.1，safetensors 0.6.2，huggingface-hub 0.36.0|
|Windows兼容|关闭自定义CUDA算子编译，使用PyTorch实现|
|默认提示词|fire. flame. burning. blaze. / smoke. smoke plume. heavy smoke.|
|默认参数|框阈值0.30、文本阈值0.25、同类NMS IoU 0.50、每类最多50框、短边640/长边1066|

配置在 `configs/preannotation.json`。按fire、smoke分别运行提示词，再做各类内的重叠框抑制；保留不同类别重叠框供人工判断。小火焰、白烟、反光等仍可能漏检或误检，阈值与提示词需要在训练/验证数据上调整。官方模型可输出“fire”“burning”等文本，最终候选类别由该次类别提示词固定映射为fire或smoke，同时保留匹配文本和置信度。

## 可执行操作

|步骤|具体操作|应看到/保存什么|
|---|---|---|
|1. 归档|在Dataset建真实来源批次，导入图片；视频先抽帧。设置关联素材相同组号，独立场景不同组号|规范PNG、原图尺寸、原始来源及哈希。公开/演示素材如实声明|
|2. 批量预标注|进入Annotations，填写执行者；确认火烟英文提示词、设备和参数，选择尚无结果/全部/手选范围，点击开始。单次最多100幅|逐图待复核/失败状态与候选数；保存模型版本、实际提示词、阈值、尺寸、硬件、依赖版本、耗时与输入哈希。失败不是无目标|
|3. 检查候选|切换人工复核，选择图片。可展开模型参数和置信度；选择历史预标注版本后，点击“采用所选预标注”才替换当前画布|模型原始候选永久保留；再次运行模型不会覆盖人工版本。采用候选将重置该图未保存画布，已有保存记录不变|
|4. 修改框|“画框”拖动新增；“选择/移动”拖动整个框或四角；点编号选择后改类别、精确坐标或删除；可撤销最近25次编辑|显示原图坐标，红色fire、蓝色smoke。预览缩放不改变导出坐标。框须在图内且宽高至少1像素|
|5. 保存草稿|填写真实复核者与修改说明，点保存草稿。切图前先保存重要修改|新增版本，状态仍不能导出。未保存画布只在当前浏览器会话内，刷新/断开不保证保留|
|6. 人工通过|逐框检查、补漏与删误检，填说明后点确认复核通过。确实无目标时删除所有框，再显式勾选整图无fire/smoke|新增已通过版本；空结果不会自动批准。多人窗口同时修改时，旧窗口保存被拒绝，需重新载入|
|7. 查记录|页面下方查看修改记录，或下载此图完整JSON|复核者、UTC时间、草稿/通过、说明、相对上个版本的新增/删除/坐标或类别变化。首次修改以采用的模型候选为基线|
|8. 标签包|YOLO导出选择已复核图片，生成标签ZIP，再选择该导出下载|labels/*.txt、manifest.json、audit.json；只含标签和审计，无图像，不是完整训练包|
|9. 训练目录|Dataset先保存当前分组划分，复核该划分全部图片，再在YOLO导出生成训练数据集|images与labels的train/val/test目录、data.yaml、划分快照、对应人工版本和审计记录。源归档和历史划分不移动|

保存复核必须填写说明；即使核对无误，也写明已检查内容。复核时长记录的是编辑页停留时间，包含空闲，不能直接当作人工有效工时。姓名/编号用于本机记录，没有账号认证，不能仅凭不同编号证明两人独立标注。

“重新载入已保存版本”放弃该图未保存画布，恢复最新人工版本（尚无人工版本时加载候选）。如需切换候选，先保存草稿，避免丢失未保存编辑。

## 导出结构与使用

类别固定 **0=fire，1=smoke**；每行 `class x_center y_center width height`，坐标相对于归档原图归一化。明确批准的负样本生成空TXT。不同来源模型可能采用其他类别序号，不能直接混用。

```text
data/fire_dataset/
  catalog.sqlite3                       # 新增预标注、复核、导出表
  exports/
    labels_<id>.zip                     # 已复核标签与修改记录
    yolo_<id>/
      data.yaml                         # 类别、相对images路径
      images/{train,val,test}/<id>.png
      labels/{train,val,test}/<id>.txt
      source_split.json                 # 固定采集组/父视频划分
      manifest.json                     # 固定已通过人工版本
      audit.json                        # 候选、参数、各次修改与来源
    yolo_<id>.zip                       # 小训练包额外打包
```

训练目录内图像总量不超过128MiB时也生成ZIP供网页下载；更大数据只生成本机目录，避免整包进入页面内存。每次导出新建版本，后续人工修改不会改变以前的包。导出前核验原图哈希、最新人工状态、划分快照哈希和当前素材指纹；新增草稿会阻止沿用旧通过状态。升级前未保存清单校验值的旧划分不能直接导出训练包，需在Dataset重新预览并保存。

`data.yaml` 使用相对图像路径。导出后进入 `Training` 选择此版本并设置参数，即可使用 [训练与独立评估功能](training.md)。如在外部脚本使用，复制/解压整个目录后向YOLO传入该文件的**绝对路径**。

默认所有数据仍保存在 `data/fire_dataset/`（可由 `FIRE_DATASET_ROOT` 修改），被Git忽略。备份/迁移时关闭服务，复制整个数据根目录，包括数据库、原图、划分与exports。Git中的代码与报告不包含实际采集素材、模型权重或标注数据包。

## 验证与题目边界

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe scripts\verify_annotations.py
```

95项测试覆盖原视频/摄像头/报警/数据功能及新版本链、框格式、差异、并发冲突、明确负样本、损坏文件、不可变导出、完整YOLO加载与旧数据库升级。Python验证在独立目录上运行真实GPU和CPU模型，使用两张上游公开图片及合成空白；见 [annotation_verification.json](../reports/annotation_verification.json)。上游部分图片本身带框，只用于流程测试。

浏览器测试先根据上述报告的 `browser_root` 设置独立 `FIRE_DATASET_ROOT`，另设独立 `FIRE_ALARM_LOG_PATH`，启动8502服务，再使用本机Playwright/Edge运行 `scripts/verify_annotations_browser.mjs`（可用 `PLAYWRIGHT_MODULE` 指定模块、`ALARM_APP_URL` 指定服务）。测试实际预标注、画框改类/缩放/删除/坐标调整/撤销、草稿与通过、空标签确认及ZIP内容。同一数据目录重启服务后加 `--history-only` 核验版本和导出字节不变。重新跑完整浏览器测试前先重跑Python验证，生成新目录。结果见 [annotation_browser.json](../reports/annotation_browser.json)。

本机IDM会接管ZIP链接，原生自动化下载表现为204/取消；实际服务端返回200及完整ZIP。测试设置 `ANNOTATION_BLOB_DOWNLOAD_FALLBACK=1` 后，使用浏览器fetch同一服务响应、保存Blob并核对内容；报告逐次记录此方式，不声称原生下载通过，也不修改IDM设置。日常操作可在下载管理器接收文件，或直接使用页面显示的本机exports路径。

自动化测试中的两个执行者是模拟角色，**不构成题目要求的至少200幅、两名真实标注者独立人工标注**。当前界面是模型辅助复核，不提供盲法独立参考标注研究流程；还须另外组织真实独立标注、冲突仲裁，计算预标注Precision/Recall、标注IoU和有对照的工时估计。也不能把功能测试结果当作准确率、降低误报或节约工时的实证结论。后续缺口见 [requirements_audit.md](requirements_audit.md)。
