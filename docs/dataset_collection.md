# 最低数量数据集整理记录

日期：2026年10月22日。状态：**最低数量候选素材已收集、入库并划分；YOLO标签为待人工复核的草稿，场景严格验收尚未完成。**

本次按用户明确告知的教师认可口径，将公开无框标注视频的自主选片、剪辑及抽帧计入自行采集。原始来源仍完整记为public，来源说明注明“非本人实拍”，没有把原作者的相机采集归属改成学生本人。该口径只适用于本次教师认可的作业要求。

## 下载与选择

D-Fire完整包已下载到 `data/public_sources/dfire/dfire_kaggle.zip`，共21,527幅图片及21,527份标签，原始类别顺序为0=smoke、1=fire。官方仓库指向的Kaggle镜像版本1大小3,049,605,157字节；完整包保留备查，仅将筛选后的200幅正常图像纳入本次交付数据分母。

|来源|入选图片|入选3秒视频|原始标注与许可|
|---|---:|---:|---|
|KMU Fire & Smoke|250|150|原始AVI，无YOLO框；限学术研究使用，保留KMU来源与指定引用|
|FIRESENSE|200|100|原始视频，无YOLO框；有pos/neg视频级分类；CC BY 4.0|
|FLAME 2 第6组RGB原视频|50|50|原视频无YOLO框；该数据集另有帧级分类标签，未将其当框真值；CC BY 4.0|
|D-Fire正常场景扩充|200|0|原始空YOLO标签，经去重及画面筛查；保留原始来源和CC0集合说明|
|合计|700|300|火烟场景500幅，正常场景200幅|

选择依据与来源：[D-Fire作者仓库](https://github.com/gaia-solutions-on-demand/DFireDataset)、[FIRESENSE原始发布页](https://zenodo.org/records/836749)、[KMU作者数据库与学术许可](https://cvpr.kmu.ac.kr/Dataset/Dataset.htm)、[FLAME 2作者机构记录与CC BY 4.0](https://experts.nau.edu/en/datasets/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-d-2/)。FLAME 2原站下载受访问条件限制，本轮使用[原始文件公开镜像](https://huggingface.co/datasets/hanvithSai/gavin-flame2-raw-partial)，已核对镜像LFS SHA-256，不将镜像作者写成原始采集作者。

视频合计900秒，即15分钟；300段是从46个原视频选取的非重叠片段，不是300起独立火灾。筛除带旧检测轮廓、明显处理覆盖层、烟火不清晰及无火烟时间段；FIRESENSE烟雾第3、5、6、7段因包含间歇或持续的旧检测覆盖层而排除。FLAME 2只用RGB，排除IR；第6段后部转向建筑，未作为火烟样本保留。D-Fire中两幅虽有原始空标签但存在火灾新闻画面或疑似烟柱的候选被排除，排除清单保存在configs/collection_negative_exclusions.json。

按本次认可口径，图像自采比例为500/700=71.43%，视频为300/300=100%，图像与视频合计为800/1000=80%。三种口径均超过30%。未经筛选的完整原始压缩包不加入已交付数据集数量，不把原视频和剪辑片段重复计数。

## 统计与划分

|内容|train|val|test|合计|
|---|---:|---:|---:|---:|
|图片|484|140|76|700|
|图片占比|69.14%|20.00%|10.86%|100%|
|视频片段|210|24|66|300|

使用固定种子42，以原视频与已知同场景为组划分；相同原视频的抽帧和片段不跨集合。相同KMU室外实验场地合为一组，部分同场景FIRESENSE视频合组，FLAME 2同一次燃烧只进入一个集合，D-Fire无法细分原始事件的AoF监控图像保守合组。完整组隔离导致视频比例不等于7:2:1，不能通过拆同源片段凑齐比例。

场景统计以下列500幅火烟图片为准，200幅正常图像的场景字段暂未逐张细分，记为未说明：

|维度|已选图片数量|
|---|---|
|室内外|室内123，室外377|
|光照|白天281，夜间115，逆光42，未说明62|
|距离|近景269，中景114，远景117|
|视角|平视450，俯视50|
|明火与烟雾外观|明火场景297，烟雾外观203；不是经测量证实的燃烧阶段|

**“早期小火星”尚未严格确认。** 本次有小范围明火与点火附近片段，但不能把远处的小火焰、打火器亮点、烟火喷焰或单纯曝光变化直接认定为真实火灾的早期火星；相关审看片保存在review目录。浓烟外观也不能单独证明处于充分燃烧期。因此数量达标不等于所有场景与燃烧阶段均已验收。

|类别|来源场景声明下的图片数|DINO自动候选框涉及的图片数|
|---|---:|---:|
|fire|334|454|
|smoke|307|356|
|normal|200|保留D-Fire原始空标签|

两类可同时出现，不能相加作为唯一图片数。来源声明基于原视频及选段，不等于逐帧真值；候选框数量也不是人工确认的类别数量。DINO存在误检和漏检，当前29幅已选正样本未得到候选框，单独列入`yolo_draft/unresolved_positive.json`，不得作为无目标负样本训练。

## 目录与标注规范

```text
data/public_sources/                   原始压缩包 下载凭据 来源与许可快照
data/minimum_collection/
  images/positive/                     入选火烟原始帧
  images/negative/                     D-Fire入选正常图像
  videos/                              300段H.264静音片段
  preannotations/                      DINO候选 参数 提示词 原图哈希
  yolo_draft/
    images/train val test/             已划分的700幅图片
    labels/train val test/             YOLO候选TXT 尚未人工通过
    data.yaml                          0=fire 1=smoke
    DO_NOT_TRAIN_BEFORE_REVIEW.txt
    unresolved_positive.json
  registration.json                    素材 原视频 时间段 哈希 场景与划分
  dataset_manifest.csv                 可用Excel打开的来源统计清单
  dataset_statistics.json              数量 类别 场景 划分及待补项
  review/                              筛选用联系表与初始点火审看片
data/fire_dataset/                     已导入现有系统的1000项及候选标注
```

每行YOLO标签为`class_id x_center y_center width height`，坐标相对于对应原图归一化至0～1，矩形框至少1像素。同一文件名对应PNG和TXT。D-Fire原始0=smoke/1=fire与本项目相反；本次只用其空标签，扩充正样本时必须转换类别，不能直接复制编号。

所有标签目前是草稿，人工通过数为0。GroundingDINO固定版本、提示词和阈值已记录，模型空结果没有自动批准为负样本。本轮未训练模型、未填写精度结论，也未生成伪造的双人标注记录。

## 在系统中使用

启动项目，打开Dataset的统计与划分页，可看到700幅图像、300段视频、500幅外部原视频抽帧及15分钟火烟片段。来源说明明确标注教师认可的公开素材整理口径。Annotations可直接选择图片、查看DINO候选、修改并保存复核；空候选的29幅正样本应优先检查和补框。

完成人工复核后，在Annotations的YOLO导出选择当前划分，生成正式已复核训练数据集，再进入Training。不要把本轮的`yolo_draft`直接当成已验收真值。本次采样以最低数量和流程可用为目标，正式精度实验前仍需核对小火星覆盖、每帧场景、负样本真值、近似重复及各集合类别分布。

## 复现和验证

按顺序运行scripts/download_collection_sources.py、inspect_collection_sources.py、build_minimum_collection.py的collect/prelabel阶段、finalize_minimum_collection.py的negatives/sheets/register/export阶段，以及verify_minimum_collection.py。仅在空目标工作区执行register，脚本遇到已有正式数据会停止，避免静默合并；替换选片使用显式--replace-selection。原始数据下载及本机生成物均被Git忽略。

校验记录见reports/minimum_collection_verification.json：700幅图像哈希与尺寸、300段视频完整解码共13,500帧、900秒时长、YOLO候选几何、非重叠片段与组隔离均通过。原始7个ZIP通过CRC；FIRESENSE匹配官方MD5，FLAME 2匹配镜像LFS SHA-256。验证只证明文件与整理规则，不证明标注准确率或全部题目场景达标。

## 独立数据包

运行`scripts/package_minimum_collection.py`生成`data/minimum_collection/最低数量数据集_700图_300视频_待复核.zip`。包内仅包含最终清单中的700幅图像、300段片段、700份YOLO候选TXT、500份DINO候选明细、来源与统计、20张筛查联系表及校验记录，不含数GB原始压缩包或历史淘汰候选。源文件目录可能保留筛选过程中的历史候选，以`registration.json`与本数据包清单为最终计数依据。

解压后使用`package_assets.csv`中的`package_path`查找素材，图像位于`yolo_draft/images/对应集合`，视频位于`videos/对应集合`；`package_manifest.json`逐项记录文件哈希及原视频时间。原始`path`和`source_video`保留项目内来源路径，不是解压后的查找路径。先读`yolo_draft/DO_NOT_TRAIN_BEFORE_REVIEW.txt`，人工复核后再从系统生成正式训练导出。

各源署名及用途限制见[来源与许可说明](collection_sources.md)；本数据包含KMU的限学术用途素材，不统一改成商业可用许可。回归测试165项通过，浏览器中的数量、候选复核展示、CSV导出及未复核数据的训练限制已验证，记录在reports/minimum_collection_browser.json。
