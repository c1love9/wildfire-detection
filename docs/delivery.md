# 本轮交付与现场演示

首次交付日期：2026-10-21；原题逐条复查与报告更新：2026-10-24。状态为 **系统功能、实验工具和材料框架完成，原题尚未全部验收完成**。正式目录已有700幅候选图像与300段视频（900秒），人工复核、双人独立标注和正式训练均尚未完成。报告采用用户提供的《设计报告模板.docx》，保留原封面、A4版心和成绩表结构；身份信息待填、教师签名和评分留空，正式实验结论仍待补。

## 交付内容

|内容|位置|说明|
|---|---|---|
|源代码与小型验证摘要|GitHub的main分支|按阶段独立提交，保留上游来源；模型和个人画面不提交|
|可编辑设计报告|artifacts/release/智能火灾报警系统设计报告.docx|教师模板版；正文、表格、公式可编辑；源稿docs/template_report.md|
|完整使用指南|artifacts/release/智能火灾报警系统完整使用指南.docx|17页可编辑操作手册；源稿docs/user_guide.md|
|系统演示视频|artifacts/release/智能火灾报警系统演示.mp4|约3～5分钟，实际浏览器操作，字幕说明，静音|
|交付包和校验清单|artifacts/release/智能火灾报警系统交付包.zip、manifest.json|包含报告、视频及安装/操作文档，不包含权重和数据|
|源码附录|交付包内source_code.zip|与manifest中提交一致的代码、配置、测试和说明；过滤模型/数据/演示媒体|
|安装与复现|docs/quick_start.md、docs/completion.md|固定依赖、模型准备、全部功能入口|
|架构与题目对照|docs/architecture.md、docs/requirements_audit.md|明确已完成工具及正式实验缺口|
|逐条实查与验证摘要|reports/requirements_acceptance.json、reports/requirements_verification.json|48条原文要求、正式库只读计数、功能验证与尚未关闭的验收门槛|

本机生成物被Git忽略。其他电脑从GitHub获取源代码后，需要按quick_start安装依赖、准备模型，并单独复制有权使用的数据/本人模型及交付包。交付包不是独立可执行程序。

## 现场演示顺序

1. 启动8501，Home选择Fire Detection / fire_n（或正式训练best），先展示图片检测，再开启火烟像素分割，下载原图尺寸掩码。
2. Demo Videos选择命名通道A，查看模拟位置/角度，开启分割，开始检测。展示掩码、占比、等级和趋势；弹窗出现后确认。提示音默认关闭，需主动开启。
3. 暂停，说明不新增采样；停止，切换通道B后开始。USB通道同样操作，但正式演示录像不录个人摄像头画面。
4. Alarm History按通道筛选，查看事件当时位置、模型、截图、片段、等级变化、趋势并导出CSV。
5. Comparison运行YOLO/DINO并排检测；说明统一test协议。没有正式训练/人工GT时显示待评测，不能以并排画面说明精度优劣。
6. Experiments查看同初始权重的固定尺寸/多尺度配置及单帧/连续确认工具。Quality展示独立访问、人工参考掩码与工时记录入口。
7. 结束时说明候选数量已经具备，但场景与标签确认、200幅真人双标、收敛训练、独立精度、实际面积误差、工时节省与优化收益仍待补。

演示录像使用隔离的数据根目录与日志，主数据目录不注入合成样例。录像中出现的短训练和质量记录明确为自动化流程验证。

## 生成和检查

2026年10月22日完成代码复核、两项操作问题修复与完整使用指南，当时164项回归通过。2026年10月24日按原题4页拆分核对48条要求，补充占比变化率指标与曲线、修复正式数据必须为空的过期断言，并更新模板报告中的采集状态；当前171项回归、11页浏览器操作、真实服务重启后的历史内容一致性检查通过。演示视频保留原功能录像，正式实验继续显示待补。历史验证记录保留各自时间与范围，不改写为本轮结果。

- scripts/record_delivery_demo.mjs：录制实际浏览器操作，需已启动隔离服务、Edge、Playwright及其FFmpeg；ALARM_APP_URL可改地址，PLAYWRIGHT_MODULE可改模块路径。当前电脑复用已安装FFmpeg 8.0.1作为录制编码器。
- scripts/build_template_report.py --template <教师模板路径>：按docs/report_template_contract.md套版，需python-docx/Pillow/lxml；先用scripts/render_report_figures.mjs渲染可编辑SVG流程图。原始模板不提交Git；同SHA模板方可直接重建。通用稿另由scripts/build_design_report.py生成，避免覆盖模板版。
- scripts/build_user_guide.py：从docs/user_guide.md生成完整使用指南，使用已配置的文档运行环境；与报告分别输出，生成后同样逐页渲染检查。
- scripts/render_design_report.py：调用文档技能的render_docx.py逐页输出PNG。无LibreOffice时可显式指定--word，使用本机Word只读导出PDF再复用渲染管线。PDF/PNG仅用于排版检查。
- scripts/check_delivery.py：核对模型/历史完整性、生产数据隔离、视频时长/编码、报告可编辑结构及交付文件SHA-256。
- scripts/verify_delivery_browser.mjs：重启后检查生产11个页面、正式实验待补状态及隔离历史回放/CSV；scripts/check_delivery.py --snapshot在重启前保存内容摘要，重启后不加参数执行比对。
- scripts/package_delivery.py：生成校验清单和ZIP，权重、数据、访问码、个人画面均不纳入。
- scripts/audit_requirements.py --source <原题PDF路径> --require-complete：生成只读正式库实查与48条核对清单；本轮严格验收返回2，表示真人工作与正式实验尚未完成，不能凭功能测试通过改成最终验收通过。

重新生成报告后必须再次逐页查看渲染结果。重新录制视频后核对时长、画面与操作是否连贯，不把等待页面或录制失败片段交付。

## 交付后仍需补充

候选素材逐帧场景/阶段核对、标签复核与最终数量统计、200幅真人双人独立标注、独立人工参考掩码、充分收敛训练和一次冻结的正式test评测。公开原视频自主整理计自采沿用教师已认可口径，保留真实来源；不改写为本人实拍。正式误报率需要正常场景参考，实际工时节省需要控制实验，优化可有效也可无效。当前所有待测结果必须保留，完成后再据实填写。最终提交还须另行交付已复核数据与本人训练权重，不能因为Git与当前功能交付包排除了大文件而遗漏。
