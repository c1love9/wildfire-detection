# 本轮数据来源与署名

本轮素材仅用于课程与学术研究。KMU的学术用途限制仍然有效，整理后的数据包不能统一重新声明为CC0或商业可用。截帧、去音轨、缩放和剪辑是本轮所做的处理；原作者署名、出处和原有水印保留。

|来源|原始发布与许可|本轮使用与署名|
|---|---|---|
|D-Fire|[作者仓库](https://github.com/gaia-solutions-on-demand/DFireDataset)；CC0集合，第三方PublicDataset部分仍受原提供方条款约束|Gaia Solutions on Demand / D-Fire Dataset；完整包来自作者README所链接的[Kaggle镜像](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)。本轮200幅正常图像未选PublicDataset前缀。保留原始文件名与原标签路径。|
|FIRESENSE|[Zenodo原始发布](https://zenodo.org/records/836749)，DOI:10.5281/zenodo.836749；[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)|FIRESENSE project / FIRESENSE Database of Videos for Flame and Smoke Detection。使用原始AVI选片和抽帧，保留源文件名；发布包内的pos/neg分类并非YOLO框标注。|
|KMU Fire & Smoke|[KMU CVPR Lab数据库](https://cvpr.kmu.ac.kr/Dataset/Dataset.htm)；版权归KMU CVPR Lab，仅允许免费用于学术研究|署名KMU CVPR Lab，并按原页要求引用下列论文之一；本轮选择该页列出的2011年论文。保留来源链接、原始文件名与视频时间。|
|FLAME 2|[作者机构数据记录](https://experts.nau.edu/en/datasets/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-d-2/)，DOI:[10.21227/swyw-6j78](https://doi.org/10.21227/swyw-6j78)；[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)|署名原FLAME 2数据集作者团队，引用原数据DOI。取第6组RGB原视频的可见烟雾时间段，未选IR。下载采用[第三方原始文件镜像](https://huggingface.co/datasets/hanvithSai/gavin-flame2-raw-partial)，镜像不是原始采集者。|

KMU原页面所列引用：ByoungChul Ko, SunJae Ham, Jae-Yeal Nam. “Modeling and Formalization of Fuzzy Finite Automata for Detection of Irregular Fire Flames.” IEEE Transactions on Circuits and Systems for Video Technology, 21(12), 1903–1912, 2011。

下载日期为2026年10月22日。每个原始包的URL、字节数与SHA-256保存在`data/public_sources/download_manifest.json`；FIRESENSE另校验官方MD5，FLAME 2校验镜像LFS SHA-256。该校验保证下载内容一致，不将镜像校验冒称原作者签名。

在教师已同意的本次课程统计口径下，公开原视频的自主筛选、剪辑和抽帧记入自行采集；所有记录仍明确`original_provenance=public`及“非本人实拍”。向其他教师、平台或期刊提交时，应如实说明此口径。
