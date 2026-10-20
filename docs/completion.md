# 暂缓真实数据期间的后续功能

此阶段沿用现有演示素材与隔离合成样例；不采集或整理正式数据，不将流程验证写成正式准确率结论。

## 像素分割

运行 `.venv\Scripts\python.exe scripts\prepare_sam.py` 准备固定版本的facebook/sam-vit-base，校验官方文件与权重SHA-256；权重约375MB，保存在忽略目录。推理使用现有Transformers 4.57.1，不微调。

Home或Demo Videos在检测设置中开启“火烟像素分割”；先运行所选YOLO，再用检测框提示SAM。同帧输出火焰/烟雾半透明掩码及轮廓。Home提供原图尺寸的火焰、烟雾、并集二值PNG下载。General通用检测不启用火烟分割。

面积为原图像素占比，同类与跨类分别取并集，不用框面积替代。无目标时有效分割为零；关闭、缺失或失败时为未知。分割失败不终止检测，面积不使用上一帧结果。SAM会在打开摄像头之前加载与预热；实际处理速度随设备记录，不承诺与纯YOLO相同帧率。

参考：[SAM模型](https://huggingface.co/facebook/sam-vit-base)、[Transformers 4.57.1 SAM文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam)。正式面积误差需要独立人工参考掩码。
