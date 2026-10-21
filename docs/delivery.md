# 本轮交付与现场演示

交付日期：2026-10-21。状态为 **系统功能、实验工具和材料框架完成**。报告已采用用户提供的《设计报告模板.docx》，保留原封面、A4版心和成绩表结构；身份信息待填、教师签名和评分留空，正式数据与实验结论仍待补。

## 交付内容

|内容|位置|说明|
|---|---|---|
|源代码与小型验证摘要|GitHub的main分支|按阶段独立提交，保留上游来源；模型和个人画面不提交|
|可编辑设计报告|artifacts/release/智能火灾报警系统设计报告.docx|教师模板版；正文、表格、公式可编辑；源稿docs/template_report.md|
|系统演示视频|artifacts/release/智能火灾报警系统演示.mp4|约3～5分钟，实际浏览器操作，字幕说明，静音|
|交付包和校验清单|artifacts/release/智能火灾报警系统交付包.zip、manifest.json|包含报告、视频及安装/操作文档，不包含权重和数据|
|源码附录|交付包内source_code.zip|与manifest中提交一致的代码、配置、测试和说明；过滤模型/数据/演示媒体|
|安装与复现|docs/quick_start.md、docs/completion.md|固定依赖、模型准备、全部功能入口|
|架构与题目对照|docs/architecture.md、docs/requirements_audit.md|明确已完成工具及正式实验缺口|

本机生成物被Git忽略。其他电脑从GitHub获取源代码后，需要按quick_start安装依赖、准备模型，并单独复制有权使用的数据/本人模型及交付包。交付包不是独立可执行程序。

## 现场演示顺序

1. 启动8501，Home选择Fire Detection / fire_n（或正式训练best），先展示图片检测，再开启火烟像素分割，下载原图尺寸掩码。
2. Demo Videos选择命名通道A，查看模拟位置/角度，开启分割，开始检测。展示掩码、占比、等级和趋势；弹窗出现后确认。提示音默认关闭，需主动开启。
3. 暂停，说明不新增采样；停止，切换通道B后开始。USB通道同样操作，但正式演示录像不录个人摄像头画面。
4. Alarm History按通道筛选，查看事件当时位置、模型、截图、片段、等级变化、趋势并导出CSV。
5. Comparison运行YOLO/DINO并排检测；说明统一test协议。没有正式训练/人工GT时显示待评测，不能以并排画面说明精度优劣。
6. Experiments查看同初始权重的固定尺寸/多尺度配置及单帧/连续确认工具。Quality展示独立访问、人工参考掩码与工时记录入口。
7. 结束时说明正式数据数量、200幅真人双标、收敛训练、独立精度、实际面积误差、工时节省与优化收益仍待补。

演示录像使用隔离的数据根目录与日志，主数据目录不注入合成样例。录像中出现的短训练和质量记录明确为自动化流程验证。

## 生成和检查

- scripts/record_delivery_demo.mjs：录制实际浏览器操作，需已启动隔离服务、Edge、Playwright及其FFmpeg；ALARM_APP_URL可改地址，PLAYWRIGHT_MODULE可改模块路径。当前电脑复用已安装FFmpeg 8.0.1作为录制编码器。
- scripts/build_template_report.py --template <教师模板路径>：按docs/report_template_contract.md套版，需python-docx/Pillow/lxml；先用scripts/render_report_figures.mjs渲染可编辑SVG流程图。原始模板不提交Git；同SHA模板方可直接重建。通用稿另由scripts/build_design_report.py生成，避免覆盖模板版。
- scripts/render_design_report.py：调用文档技能的render_docx.py逐页输出PNG。无LibreOffice时可显式指定--word，使用本机Word只读导出PDF再复用渲染管线。PDF/PNG仅用于排版检查。
- scripts/check_delivery.py：核对模型/历史完整性、生产数据隔离、视频时长/编码、报告可编辑结构及交付文件SHA-256。
- scripts/verify_delivery_browser.mjs：重启后检查生产11个页面、正式实验待补状态及隔离历史回放/CSV；scripts/check_delivery.py --snapshot在重启前保存内容摘要，重启后不加参数执行比对。
- scripts/package_delivery.py：生成校验清单和ZIP，权重、数据、访问码、个人画面均不纳入。

重新生成报告后必须再次逐页查看渲染结果。重新录制视频后核对时长、画面与操作是否连贯，不把等待页面或录制失败片段交付。

## 交付后仍需补充

采集与自采比例、真实场景覆盖、200幅真人双人独立标注、独立人工参考掩码、充分收敛训练和一次冻结的正式test评测。正式误报率需要正常场景参考，实际工时节省需要控制实验，优化可有效也可无效。当前所有“待正式数据”结果必须保留，完成后再据实填写。
