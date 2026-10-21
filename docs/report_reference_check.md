# 设计报告参考文献核验

核验日期：2026-10-21。使用nature-ref-verifier流程，对9条候选的作者顺序、标题、年份、出处与页码核对。DOI先请求Crossref，再与原始会议页面、出版社或作者项目交叉检查。完整原始元数据留在本机template_work/crossref_metadata.json；不把来源间差异隐藏为一致。

|编号|采用记录|来源与核验说明|
|---|---|---|
|1|Redmon等 YOLO CVPR2016 779–788|[Crossref DOI](https://doi.org/10.1109/CVPR.2016.91)与[CVF原始会议页](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)标题/作者/年/页一致|
|2|Lin等 FPN CVPR2017 2117–2125|[CVF会议页](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)及其论文PDF采用该会议版页码；候选DOI查询失败，因此不附未经证实的DOI|
|3|Lin等 Focal Loss ICCV2017 2999–3007|[DOI/Crossref](https://doi.org/10.1109/ICCV.2017.324)使用IEEE出版页码。[CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)为2980–2988，已明确选择IEEE版，未混搭|
|4|Liu等 Grounding DINO ECCV2024 38–55|[ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php)核对标题与作者；[Springer出版记录](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_3)及Crossref核对页码。在线2024-11-23，Springer建议引用年为2025，报告采用出版年2025，会议名仍为ECCV2024|
|5|Kirillov等 SAM ICCV2023 3992–4003|[DOI/Crossref](https://doi.org/10.1109/ICCV51070.2023.00371)使用IEEE出版页码；[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)为4015–4026，报告统一采用IEEE版|
|6|Lin等 COCO ECCV2014 740–755|[Springer](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)与Crossref标题/作者/年/页一致|
|7|Loshchilov与Hutter AdamW ICLR2019|[作者项目](https://github.com/loshchil/AdamW-and-SGDW)明确两名作者、正式标题及ICLR2019；[原始会议链接](https://openreview.net/forum?id=Bkg6RiCqY7)当前触发浏览器验证。采用作者一手证据，不编造页码或DOI，不标成双源完全一致|
|8|Everingham等 VOC IJCV2010 88(2) 303–338|[Springer](https://link.springer.com/article/10.1007/s11263-009-0275-4)与Crossref标题/作者/卷期页一致。Crossref在线日期2009，采用期刊卷归属2010|
|9|Dice Ecology1945 26(3) 297–302|[Wiley原始条目](https://esajournals.onlinelibrary.wiley.com/doi/10.2307/1932409)、同期期刊目录及Crossref一致|

报告按顺序编码制列出文献，正文引用用上标方括号。作者超过3人以“et al.”截断属于统一著录格式，不表示漏掉原始作者。FPN和AdamW分别按公开会议版本或作者原始证据采用，核验范围如表所示。文献仅支撑方法背景，不用于替代本项目真实实验结果。
