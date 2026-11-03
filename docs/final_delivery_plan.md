# 最终交付包内容规划

日期：2026-11-03。依据原题第3～4页验收与提交要求及[本次全部子任务复查](requirements_recheck_20261103.md)。这是正式交付规划，当前数据标签与正式实验未全部完成，不代表最终包已经生成。

## 交付形态

推荐先整理一个统一版本的总目录，再压缩为 `智能火灾报警系统_姓名_学号_最终交付.zip`。姓名/学号由本人填写。GitHub保存代码、配置与可公开说明；教师交付包另含最终数据、权重、实验和演示。Git忽略这些大文件不代表题目允许不交。

若上传大小受限，可以拆为“报告源码”“数据标注”“模型实验”三份，使用相同release_id和总校验清单；README写清三份均需接收。可选离线模型另成一份附包，不与本人训练权重混淆。

```text
智能火灾报警系统_姓名_学号/
  README_先看这里.md
  manifest.json
  SHA256SUMS.txt
  01_设计报告/
    设计报告.docx
    设计报告.pdf                         # 可选阅读版，不能替代DOCX
    figures/                             # 可编辑图源及实际结果图
  02_源代码/
    source_code.zip
    repository.txt                       # 仓库URL、分支、提交、上游来源
    requirements-lock.txt
    environment.json
  03_最终数据集/
    README.md
    data.yaml
    images/{train,val,test}/
    labels/{train,val,test}/              # 0=fire，1=smoke；真实负样本空TXT
    videos/                              # 至少300段，合计不少于10分钟
    statistics.json
    statistics.csv
    sources.csv                          # 来源、原视频、时间、帧号、哈希
    split_manifest.json                  # 冻结划分、采集组、原视频关联
    class_mapping.json
    licenses/                            # 数据源许可、署名及下载/校验记录
  04_标注过程与质量/
    annotation_protocol.md
    prelabel_configs/                    # 提示词、固定版本、阈值、尺寸
    prelabels/                           # 原模型候选，保留版本和输入哈希
    final_label_versions.json
    human_review_history.json
    independent_200_manifest.json
    annotator_A/labels/
    annotator_B/labels/
    quality_report.json
    quality_report.csv
    tuning_log.md
    tuning_results.json
    worktime.csv
    worktime_report.md
  05_模型与训练/
    baseline/
      best.pt
      last.pt                            # 建议保留
      initial_weights_receipt.json
      args.yaml
      effective_parameters.json
      environment.json
      train.log
      results.csv
      results.png
      training_time.json
    optimized/                           # 与基线相同的完整结构
    inference_models.json                # 本人模型、DINO、SAM用途和校验值
  06_评测与实验/
    protocol.json
    frozen_test_manifest.json
    yolo/                                # 逐图预测、指标、计时与失败案例
    grounding_dino/
    model_comparison.csv
    optimization_comparison.csv
    negative_false_alarms.csv
    scene_generalization.csv
    device_generalization.csv
    area_reference_masks/                # 原图同尺寸，人工独立参考
    area_predictions/
    area_error.csv                       # IoU、Dice、占比误差
    README_结果口径.md
  07_演示与操作/
    系统演示.mp4
    完整使用指南.docx
    quick_start.md
    演示步骤.md
    demo_sources.json
    demo_media/                          # 必要、获准使用的演示素材
  08_验收与校验/
    requirements_checklist.md
    acceptance_evidence.json
    unit_tests.json
    browser_checks.json
    reproduction_check.md
    verification.json
  09_过程附录/                            # 可选，不充当最终结果
    trial_3epochs/
    盲评对照说明.md
```

## 各部分收集什么、现在还缺什么

|目录|交付重点|现有基础|打包前要补齐|
|---|---|---|---|
|根目录|一页说明如何启动、如何演示、文件用途、统一版本及全包哈希|已有安装和使用说明|最终数据/模型版本、相对路径和清晰入口；若仍有待完成项必须列明，不能标成最终验收通过|
|01 报告|严格教师模板、可编辑正文/公式/表格；数据、方法、训练、对比、系统、误差与局限|模板DOCX草稿和图源已有|真实封面信息、最终统计/训练/测试/工时/面积结果；教师签章和评分由教师填写|
|02 源码|版本库可访问、ZIP与指定提交一致、完整依赖和环境、上游署名|持续push，固定依赖与启动脚本已有|从最终提交重新导出；使用完整commit ID；上游LICENSE/致谢保留|
|03 数据|至少500有效火烟图+200正常图，300段不少于10分钟，最终标签、目录与来源规范|700图/300段/15分钟候选，来源与分组已整理|确认标签和严格场景；正式排除5幅时补足；更新统计。data.yaml及来源路径应可迁移，不能依赖本机D盘绝对路径|
|04 标注|开放域候选、人工修改痕迹、200幅双人原始标签及P/R/IoU、提示词调参、有效工时|A/B各200份、旧/增强候选、质量与调参记录已有|保存最终标签采用关系；补辅助复核有效用时。A/B与模型候选分别留存，不增加强制逐图仲裁|
|05 模型|本人训练best及可复现的完整训练目录、初始权重来源、硬件时长和调度|3轮试训best/last可加载，训练工具齐全|最终标签上的充分训练、曲线及收敛分析；基线和优化独立目录，不覆盖旧结果|
|06 评测|同一人工test、同一协议的YOLO/DINO结果及优化前后差值、误报、场景/设备限制和面积误差|统一评测器、成对实验、掩码工具已实现|真实独立test与人工参考掩码；正式实验结果。设备身份未知则标未知，不伪造跨设备测试|
|07 演示|建议3～5分钟连贯展示；指南、复现步骤、素材出处|约3.5分钟旧演示及Word指南已有|用最终自训模型更新相关部分；涵盖检测/分割→面积/报警→截图片段→历史地图→通道/视角→对比结果|
|08 验收|原题逐条证据、测试范围、版本、文件校验和重新启动/迁移复现|各阶段测试与浏览器记录已有|在最终版本重做必要检查，区分历史测试、页面打开检查、完整操作验证和正式实验，不混写时间与范围|
|09 附录|过程实验、失败案例、短试训或盲评的解释材料|3轮试训和21幅盲评已有|明确临时标签与真实来源；不能把附录试训的best放成主要正式权重|

## 两份容易遗漏的说明

### 模型来源与离线复现

至少清楚区分：①本人最终YOLO；②COCO初始权重；③上游火烟YOLO辅助候选；④GroundingDINO Tiny/Base；⑤SAM ViT-B。每项记录模型名、固定revision、文件SHA-256、用途、是否本项目微调及样本量。上游辅助模型和本人模型不得互相冒用。

本人最终best必须交付。DINO/SAM等大权重可以在联网复现方案中提供官方固定下载地址、校验值和准备命令；若答辩环境离线，应额外准备许可允许的离线权重附包并演练导入。本轮不把下载地址当作本人best权重的替代品。

### 数据来源、隐私与盲评

原始公开视频包可以作为可选来源归档附包，至少交来源清单、原视频标识、帧号/时间、哈希与处理方法。500幅自主整理图仍保留public原始来源和教师认可的课程口径；KMU学术用途限制及其他数据集署名不因剪辑而消失，不把完整数据/权重直接推入公开Git仓库。

教师需要的A/B标注、最终标签和审计记录要交；本机角色访问码、认证token、账号凭据、私人摄像头画面、缓存及`.venv`不打包。盲评允许先隐藏两套框来源，但来源对照留存、评后可揭盲，不把机器操作记为学生独立人工操作。

## manifest与打包验收

总manifest至少记录release_id、生成时间、代码commit、原题哈希、数据/划分/标注/模型版本、每个文件的路径/字节数/SHA-256、用途、必交/可选属性、未完成项以及正式验收状态。SHA256SUMS覆盖所有提交文件；拆包时共用同一总清单，避免混版本。

生成正式包前逐项检查：

1. 从冻结的最终提交导出源码，确认清单commit与ZIP一致；不将旧10月24日源码包直接沿用。
2. 重数有效火烟、正常图、视频段和总时长，核对原视频/场景组隔离、图像与标签一一对应、归一化边界及类别映射。
3. 检查200幅A/B两份原始标签、冻结候选、P/R/匹配IoU及未匹配数和有效工时说明完整；无需逐图仲裁说明。
4. 在新位置按README安装/准备模型，使用相对数据路径加载本人best并跑图片、视频及可用USB，检查报警、暂停/停止、通道释放、位置/视角、历史回放与导出。
5. 正式test只使用既定模型/阈值，保存统一指标、资源、预/后处理计时、误报和失败案例；面积指标对应独立人工参考。
6. 报告、表格、曲线、训练目录和权重相互对应；不将短试训、合成夹具或训练集诊断填进正式性能表。
7. 检查ZIP CRC和总清单哈希，确认数据与本人权重确实在交付内容中，剔除访问码/凭据和私人画面，再完成一次现场连贯演练。

现有 `artifacts/release/` 包内文件一致性检查通过，但内容为2026-10-24功能快照且缺数据/模型。最终产物应新建版本，旧包保留用于历史追溯。
