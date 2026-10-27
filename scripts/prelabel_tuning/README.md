# 本轮预标注试验的复现脚本

这组脚本对应`docs/preannotation_optimization.md`，保留实际执行的算法、选择与失败记录。它们面向本项目已冻结的200幅A/B任务及现有划分；不是可任意套用的新数据集优化器。输入必须恢复完整`data/fire_dataset/`、原始A/B提交、Tiny/Base和上游火烟权重。仓库版本另给批量写入增加了`--run`开关，未指定时只显示说明。

脚本默认把预测、选样与报告写入被Git忽略的`artifacts/prelabel_optimization_20261027/`，不把图片、框标签或访问码提交仓库。原始执行脚本也保留在该目录；仓库版本补充了模块搜索路径与固定输出路径，算法相同。

先在项目根目录准备固定权重：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_grounding_dino.py
.\.venv\Scripts\python.exe scripts\prepare_grounding_dino.py --base
```

按本轮顺序复现；真实模型运行需要时间：

```powershell
.\.venv\Scripts\python.exe scripts\prelabel_tuning\trial.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\trial.py --confirm
.\.venv\Scripts\python.exe scripts\prelabel_tuning\trial_yolo.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\fusion_trial.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\highres_trial.py
```

原执行还冻结了新的27幅检查清单与剩余93幅清单。它们不是可以事后任意更换的输入；本机`protocol.json`、`round2_confirmation_protocol.json`及`remaining93_protocol.json`已保存选样ID、种子和来源组。要准确复现本轮，请成套保留该目录中的这些清单。

```powershell
.\.venv\Scripts\python.exe scripts\prelabel_tuning\round2_recheck.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\fallback_trial.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\multilabel_trial.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\final_validation.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\base_trial.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\base_validate.py
.\.venv\Scripts\python.exe scripts\prelabel_tuning\verify_adapter.py
```

除最后的批量入库脚本外，上述试验只读正式数据库。它们不批准标签、不改A/B、不修改独立test，也不更新已保存的人工作业。阈值搜索多为离线重筛已有模型输出，不是每个参数组合都重新做一次模型前向。

批量入库脚本`run_batch.py`会写入新的候选版本，先验证实际整体改善和适配器一致性，创建数据库备份，再分批追加；同一批号成功的图片可跳过以恢复任务。它明确记录逐小批严格门槛未全通过，不宣称所有场景无退化。当前700幅已跑完，不必再次执行：

```powershell
.\.venv\Scripts\python.exe scripts\prelabel_tuning\run_batch.py --run
```

原200幅质量任务冻结的是Tiny历史候选，其报告保持原口径。增强候选的统计另存；不能用新结果回填原任务，再将调参后统计称为原始零样本成绩。
