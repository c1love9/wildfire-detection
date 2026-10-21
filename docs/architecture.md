# 系统架构与处理流程

## 运行结构

```mermaid
flowchart TD
  I[图片 上传视频 命名视频 USB摄像头] --> C[输入生命周期与通道快照]
  C --> Y[YOLO 上游或本人已训练best]
  Y --> R[统一FrameResult 检测框 原图尺寸 模型 耗时]
  R --> S{开启分割}
  S -->|是| M[SAM ViT B 检测框提示 同帧掩码]
  S -->|否或失败| U[面积未知 保留检测]
  M --> A[同类和跨类并集 原图像素占比]
  A --> T[有界实时曲线 完整采样落盘]
  A --> E[同区域连续确认 时间与面积分级]
  U --> E
  E --> N[去重弹窗 可选声音]
  E --> D[SQLite事件 等级转移 观测快照]
  E --> V[确认截图 前后片段 H264]
  T --> D
  D --> H[历史通道 地图 趋势 CSV 回放]
  V --> H
```

|层次|主要模块|责任|
|---|---|---|
|网页与配置|Home、pages、inference_ui、monitoring_ui、configs|模型/分割/通道选择、展示与导出|
|输入与推理|video_runtime、camera_runtime、model_runtime、inference_result、segmentation|解码/设备释放、统一结果、同帧掩码与兼容旧接口|
|报警与历史|alarm、alerts、alarm_log、recording、trends、channels|区域连续确认、分级、媒体、趋势与历史快照|
|数据与实验|dataset_store、annotations、grounding_prelabel、training、comparison、experiments、quality|版本、复核、训练、统一评测、控制实验与质量工具|

## 数据和实验流程

```mermaid
flowchart LR
  A[采集或有授权导入] --> B[来源 场景 去重 关联组]
  B --> C[冻结train val test]
  C --> D[GroundingDINO候选]
  D --> E[人工复核 版本记录]
  E --> F[冻结YOLO导出]
  F --> G[训练及val选best]
  G --> H[统一独立test]
  G --> I[图片 视频 摄像头选模型]
  J[两名真人各自空白画布] --> K[双提交锁定后仲裁]
  K --> L[预标注P R 框IoU]
  M[独立人工参考掩码] --> N[IoU Dice 面积误差]
```

正式test须保持来源隔离和人工参考独立性。辅助复核不等于独立标注。相同模型既用于预标注又用于对比时必须披露，不以模型掩码自动充当真值。

## 状态与持久化

视频暂停保留当前位置且不采样；摄像头暂停关闭设备，继续时重新打开并创建新观测段。运行或暂停时禁止切换输入与模型。停止、视频结束或设备异常关闭本段观测，不生成“火情恢复”结论。

报警数据库默认位于artifacts/alarms/alarms.sqlite3，媒体位于其同级media；趋势存入observation_samples。数据、训练、比较、实验及质量索引位于data/fire_dataset/catalog.sqlite3及对应子目录。模型下载在artifacts/models。整个数据根目录与报警数据库/媒体必须成套备份，先结束任务并关闭服务。旧记录缺失新字段时显示未记录，不能补造数值。
