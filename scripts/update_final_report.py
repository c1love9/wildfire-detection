"""Fill report source slots with completed formal evidence, preserving template."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training import read_json


def replace_section(text,title,next_title,body):
    start=text.index(title);end=text.index(next_title,start+len(title))
    return text[:start]+title+'\n\n'+body.strip()+'\n\n'+text[end:]


def update(workspace):
    formal=read_json(workspace/'formal_experiments.json')
    stats=read_json(workspace/'final_dataset_statistics.json');curves=read_json(workspace/'convergence_analysis.json')
    source=ROOT/'docs/template_report.md';text=source.read_text(encoding='utf-8')
    count=stats['totals'];selected=formal['selection']
    pct=lambda x:f'{100*x:.2f}%'
    # Replace the previous snapshot date regardless of whether this builder was run before.
    text=re.sub(r'截至2026年\d+月\d+日，当前状态为.*?\n\n',
                f"截至2026年11月8日，700幅图片已由用户完成复核，保留同一200幅的A/B各200份原始独立标注。最终含火烟{count['positive']}幅、正常{count['negative']}幅，其中自主整理来源且含火烟{count['approved_own_positive_images']}幅。300段视频合计900秒，来自46个公开原视频。固定尺寸和多尺度两组各50轮正式训练及同一冻结test评测已完成。严格自主火烟数量尚差5幅、正常图尚差7幅，候选另放待补，不混入本次冻结数据。用户同意人工参考掩码、实际面积误差和有效辅助复核工时暂留待补；报告封面身份暂不填写。公开原视频自主整理计自采采用教师认可的课程口径，保留原始public来源，不写成本人实拍。\n\n",
                text,count=1)
    text=re.sub(r'本项目由负责人统筹.*?\n\n',
                '本项目由负责人统筹选型、整合、实验设计、验证及材料整理，代码实现、测试和文档编排使用AI编程助手辅助。负责人和队友已完成同一200幅的A/B独立标注，用户确认700幅辅助标注复核全部完成。两套独立提交及最终复核版本分别留存，不把自动化操作计为学生独立人工操作。教师评分与签章由教师实际填写，上游模型不冒充本人训练成果。\n\n',text,count=1)
    text=re.sub(r'Dataset支持采集、导入、来源和场景登记.*?\n\n',
                'Dataset支持采集、导入、来源和场景登记、去重、数量统计及分组划分。父视频、采集组及重复内容保持集合隔离，防止相邻帧随机进入train/val/test。当前冻结版本700幅，含507幅火烟和193幅正常，fire框489个、smoke框348个。自主整理来源500幅中495幅仍含火烟，另5幅被确认为正常；D-Fire来源200幅中12幅最终标有目标、188幅正常。来源声明与实际标签因此分别统计，不沿用“500火烟+200正常候选”作为最终结论。场景和真实燃烧阶段仍需人工确认；演示及合成夹具不计入正式数据。\n\n',text,count=1)
    text=text.replace('本项目默认请求50轮、640尺寸、种子42，实际参数与早停结果以保存文件为准。上述默认值为可执行起点，并不表示已找到正式火灾数据的最优设置。',
                      '本轮两组各请求50轮、640尺寸、batch8、种子42，关闭早停，固定同一初始权重、数据、增强和其他参数。实际学习率调度、权重衰减与增强参数以effective_parameters.json和args.yaml为准；这组控制实验不代表已找到最优配置。')
    quality_sentence='两人独立标注同批200幅，同类IoU至少0.5的一对一匹配共160对，匹配平均IoU为0.7207，A有110个、B有129个未匹配框，40幅双方均空。平均IoU只反映匹配部分，不能隐藏未匹配框。原Tiny分别对A/B为TP122/110、FP273/285、FN148/179，F1为36.69%/32.16%；增强候选为TP128/123、FP114/119、FN142/166，F1为50.00%/46.33%。自动框相对A/B的匹配IoU原为0.7353/0.7518，增强为0.7317/0.7416，故P/R改善并不代表定位IoU也提高。'
    anchor='质量报告保留同类一对一匹配框IoU'
    if quality_sentence not in text:
        position=text.index(anchor);text=text[:position]+quality_sentence+'\n\n'+text[position:]
    split_table=['表3.4 最终已复核数据及支持数','','|集合|图像及正常数量|fire与smoke框数|','|---|---|---|']
    for name in ('train','val','test'):
        row=stats['by_split'][name];split_table.append(f"|{name}|{row['images']}幅，正常{row['negative']}幅|{row['boxes_fire']} / {row['boxes_smoke']}|")
    experiment=['Training从已复核冻结版本启动独立后台任务。本次两组均由COCO预训练YOLOv8n初始化，50轮、640、batch8、AdamW、初始学习率0.001、seed42、FP32、workers0、确定性开关开启，关闭早停。两组仅multi_scale不同，内置线性学习率调度与其他增强保持一致。保存请求、实际参数、环境、逐轮CSV、曲线、best/last、数据和权重哈希。',
                '\n'.join(split_table),
                f"主模型在查看test前按验证集mAP50-95选择为{selected['variant']}，训练任务{selected['training_id']}。验证集用于best和方案选择，test没有参与训练或调参。多尺度首次工作进程在6轮后中断，保留原始日志，使用相同配置新建任务完整重跑50轮；不中途拼接两次曲线或覆盖旧文件。",
                '表3.5 统一比较协议\n\n|项目|固定设置|\n|---|---|\n|输入与计算|等比例填充640×640 batch1 FP32 同一RTX5060 Laptop GPU|\n|固定点P R F1|conf0.25 同类一对一IoU0.5 有GT类别宏平均|\n|AP|候选0.001 IoU0.50～0.95 步长0.05|\n|计时|预热 同步 含预处理推理后处理 不含读盘加载及画框|',
                '独立开放域对比采用固定GroundingDINO Tiny，fire.与smoke.分别查询，文本阈值0.25，本项目微调样本数0；不使用增强预标注的Base+辅助YOLO融合。Tiny和Base均曾用于候选，最终测试参考为用户通过版本，仍应披露预标注依赖及实际人工协议。YOLO与DINO置信度不是已校准的同一概率。',
                '表3.6 同一冻结test的实际指标\n\n|模型或实验|P R F1|mAP50与mAP50至95|\n|---|---|---|']
    aliases={'baseline':'YOLO固定尺寸','multiscale':'YOLO多尺度','comparison_GroundingDINO':'DINO Tiny'}
    displayed=[row for row in formal['rows'] if row['experiment'] in aliases]
    for row in displayed:
        experiment.append(f"|{aliases[row['experiment']]}|{pct(row['precision'])} / {pct(row['recall'])} / {pct(row['f1'])}|{pct(row['map50'])} / {pct(row['map50_95'])}|")
    # Join table rows without blank paragraphs so the Word table parser retains them.
    experiment_text='\n\n'.join(experiment[:-len(displayed)])+'\n'+'\n'.join(experiment[-len(displayed):])
    speed=['表3.7 同协议速度及资源成本','','|模型|平均耗时与FPS|参数与权重|','|---|---|---|']
    comparison=read_json(ROOT/'data/fire_dataset/comparisons'/formal['comparison_id']/'result.json')
    for name,row in comparison['models'].items():
        speed.append(f"|{name}|{row['mean_ms_per_image']:.2f}ms / {row['fps']:.2f}FPS|{row['parameters']/1e6:.2f}M / {row['file_bytes']/1024**2:.2f}MiB|")
    experiment_text+='\n\n'+'\n'.join(speed)
    delta=formal['difference_multiscale_minus_baseline']
    experiment_text+=f"\n\n多尺度相对固定尺寸的mAP50-95变化{100*delta['map50_95']:+.2f}个百分点，F1变化{100*delta['f1']:+.2f}个百分点。差值反映本次固定配置的真实结果，不预设多尺度必然有效。两组依次运行、两个模型交替计时，系统负载和模型缓存会影响速度，因此不把该值称硬件极限。训练页最大平均F1点指标与这里固定阈值指标分开保存。\n\n"
    for variant,curve in curves.items():
        identifier=formal['jobs'][variant]
        figure=(workspace/'figures'/(variant+'_curves.png')).relative_to(ROOT).as_posix()
        experiment_text+=f"![图3.{3 if variant=='baseline' else 4} {variant}实际50轮损失与验证mAP](../{figure})\n\n"
        a,b=curve['last_10_validation_range']
        experiment_text+=f"{variant}实际训练{curve['training_wall_seconds']/60:.2f}分钟，CSV最佳验证轮为第{curve['best_validation_epoch_csv']}轮。最后10轮验证mAP50-95范围{pct(a)}～{pct(b)}；训练框损失前5轮中位数{curve['train_box_loss_first_5_median']:.3f}、末5轮{curve['train_box_loss_last_5_median']:.3f}，验证框损失相应为{curve['val_box_loss_first_5_median']:.3f}与{curve['val_box_loss_last_5_median']:.3f}。完成50轮并不自动证明充分收敛，若训练下降而验证波动或偏高，应解释泛化受限或可能过拟合。\n\n"
    experiment_text+='test只有76幅且仅1幅正常图，正常误报率不能据此作稳定结论。按来源声明的场景分层已导出，场景未逐图确认，设备身份未知，不能把不同数据源冒称跨摄像头独立测试。保留逐图预测与失败案例，不在看过test后回调阈值或改写标签来美化本次成绩。'
    text=replace_section(text,'## 3.6 训练 统一比较与优化实验','## 3.7 验证结果与正式实验位置',experiment_text)
    log=(workspace/'regression_tests.txt').read_text(encoding='utf-8-sig')
    match=re.findall(r'Ran (\d+) tests? in ',log);assert match and log.rstrip().endswith('OK')
    tests=int(match[-1])
    validation=f'''本轮{tests}项回归测试通过。功能检查覆盖同帧掩码、重叠面积、报警边界、暂停与断流、设备释放、通道切换、历史媒体及导出、训练快照完整性、统一匹配和交付恢复。新增修复是统一比较排除YOLO画框耗时，并为中断训练提供保留旧记录的同配置重试，绝不自动重启用户取消任务。

表3.8 本轮证据与未完成项

|检查|实际证据|结论边界|
|---|---|---|
|人工框复核|700幅最新通过，A/B各200份原始提交保留|不额外强制逐图仲裁|
|训练与优化|两组完整50轮，best/last、曲线、环境和时长|优化差值如实报告；轮数不替代收敛分析|
|统一独立test|同76幅、同协议YOLO/DINO与两组模型比较|正常图仅1幅，误报率不稳定|
|真实自训集成|图片、视频、SAM、报警媒体与浏览器检查|公开演示不能替代正式精度|
|数据数量|507火烟、193正常；自主火烟495|待补5幅自主火烟及7幅正常，候选单列|
|面积和工时|工具已实现，无人工参考和有效辅助计时|用户同意待补，指标不填写|

原700幅复核及A/B原始提交在训练与功能验证后通过指纹检查保持不变，训练和评测只读取冻结副本。已知原视频/场景组隔离，D-Fire部分原事件未知及公开预训练重叠未知的限制保留。来源、标签、训练、测试和报告通过版本及哈希关联。

交付包实际包含已复核images/labels、300段视频、A/B原始标签、原始预标注和修改历史、本人两组best/last、曲线、逐图测试、可编辑教师模板报告、约3～5分钟演示及指南。另含清除旧访问凭据的运行状态和新目录恢复映射，拒绝覆盖已有不同文件。SAM/DINO等第三方大模型按固定官方版本下载并校验；数据及个人画面不推入代码仓库。

本轮仍不能标记为“所有原题要求完成”。严格场景及早期火星/真实燃烧阶段、数量差额、独立人工参考掩码和有效工时尚未关闭。最后两项由用户明确同意暂留待补，身份信息暂不填写，教师成绩及签章留空，现场验收由实际操作者和教师完成。'''
    text=replace_section(text,'## 3.7 验证结果与正式实验位置','# 设计总结',validation)
    old_marker='当前阶段完成的是系统功能、实验工具和材料框架'
    start=text.index(old_marker if old_marker in text else '当前已完成系统功能、700幅人工复核')
    end=text.index('本方案也有明确局限',start)
    text=text[:start]+f'当前已完成系统功能、700幅人工复核、200幅双人独立提交、两组真实训练、独立test统一比较和可复现交付材料。本轮{tests}项回归及浏览器检查验证实现，真实训练和逐图指标另行留存；它们不能替代缺失的人工掩码或工时实测。\n\n后续只需按缺口补充并确认5幅自主火烟帧、7幅正常图与严格场景证据，取得独立人工参考掩码和受控工时，再冻结新数据版本。当前700幅结果和本次实验不覆盖；正式模型表现有限时保留失败案例与支持数，不能用上游论文分数或合成测试充数。\n\n'+text[end:]
    text=text.replace('正式结果补齐后，应同步更新报告、数据凭据、模型和交付清单，确保文字与实际系统保持一致。','后续待补项完成后，应新建报告及数据版本，同步凭据和交付清单，保留本次已完成实验。')
    text=text.replace('fire框489个、smoke框348个。','fire框489个、smoke框348个；含fire图像335幅、含smoke图像300幅，其中128幅同时含两类。')
    text=text.replace('两组仅multi_scale不同，内置线性学习率调度与其他增强保持一致。',
                      '两组仅multi_scale不同，内置线性学习率调度与其他增强保持一致。实际lrf0.01、cos_lr关闭、预热3轮、warmup_bias_lr0.1、momentum0.937、weight_decay0.0005；mosaic1.0、最后10轮关闭mosaic、scale0.5、fliplr0.5。预热期间各参数组学习率不固定为0.001。')
    text=text.replace('新增修复是统一比较排除YOLO画框耗时，并为中断训练提供保留旧记录的同配置重试，绝不自动重启用户取消任务。',
                      '新增修复包括排除YOLO画框耗时、同配置保留旧记录重试，以及CPU先运行再切GPU的可见性问题。推理传递torch.device避免依赖改写CUDA_VISIBLE_DEVICES，异常也恢复原环境；实测CPU后首次GPU加载及实际USB生命周期通过。用户取消任务不自动重启。')
    text=text.replace('用户取消任务不自动重启。','用户取消任务不自动重启。训练曲线改为自含SVG，避免切换任务时引用过期Vega数据；逐轮数据与下载保留。')
    text=text.replace('该融合流程不称为纯DINO零样本，独立test仍待完成。','该融合流程不称为纯DINO零样本。该阶段的诊断不是独立test，本次正式测试结果见3.6节。')
    text=text.replace('test只有76幅且仅1幅正常图，正常误报率不能据此作稳定结论。','test只有76幅且仅1幅正常图。基线和DINO均在该图误检（1/1），多尺度未误检（0/1）；样本过少，不据此判定误报优劣。')
    source.write_text(text,encoding='utf-8')

    guide=ROOT/'docs/user_guide.md';value=guide.read_text(encoding='utf-8')
    value=value.replace('版本日期：2026年10月22日。','版本日期：2026年11月8日。')
    value=re.sub(r'当前交付状态为.*?\n\n','当前已保存700幅真实复核、A/B各200份独立标签、两组50轮正式训练和统一独立test。剩余为5幅自主火烟帧及7幅正常图、严格场景确认、人工参考掩码和有效辅助复核工时；后两项按用户意见待补，封面身份暂不填写。原700幅无需重新复核。功能演示、验证集与独立test指标分开保存。\n\n',value,count=1)
    value=value.replace('Home 保持 Fire Detection 和 fire_n，确认内置图片能显示检测框；可开启“火烟像素分割”查看面积。',f"Home 保持 Fire Detection，选择自训列表中任务尾号{selected['training_id'][-8:]}的best，确认图片检测；可开启火烟像素分割。fire_n是保留的上游模型。")
    value=value.replace('中断任务目前新建重试。','成对实验的失败/中断任务可由执行脚本按同配置新建重试，保留旧日志；用户取消任务不自动重启。')
    marker='## 十六 本次正式模型与交付恢复'
    if marker in value:value=value[:value.index(marker)]
    value+=f'''\n\n{marker}

本次推荐模型任务为{selected['training_id']}，实验组{selected['variant']}，已训练50轮。Home的Select Model Size、Demo Videos的检测模型可选择尾号{selected['training_id'][-8:]}；先停止视频再切换，图片和视频的选择彼此独立。

本人权重位于data/fire_dataset/training/{selected['training_id']}/output/weights/best.pt。训练页查看val；Comparison历史记录{formal['comparison_id']}查看固定conf0.25的test。Experiments记录{formal['experiment_id']}保留两组差值，均不需要再次点击评测按钮。详细数值见docs/formal_results.md。

旧的3轮试训和artifacts/release的10月功能包只作过程历史；正式交付以新版本总目录manifest指定的提交、数据和模型为准。交付的03目录是最终标签，不是yolo_draft。09中的12幅仅为待确认候选，没有计入本次700幅或test。

换电脑时先在全新目录解压02_源代码/source_code.zip，安装依赖，然后在该源码目录执行下列命令，将引号里的路径换成已解压的总交付目录。

```powershell
.\\.venv\\Scripts\\python.exe scripts\\restore_final_delivery.py --package "D:\\交付材料\\智能火灾报警系统_交付材料"
```

恢复工具逐文件校验并拒绝覆盖已有不同内容，因此应使用新目录。它会恢复700幅数据、300段视频、两组训练和自训模型列表、比较历史；旧A/B访问码不随包提供，200幅原始标签和质量报告直接查看04目录。SAM和DINO仍按第二章固定脚本准备，FFmpeg需单独安装。双击启动项目.cmd后按第一章验证。

本机已有数据不需要执行恢复命令，也不要为看结果重跑训练。需要备份时保留整个data/fire_dataset和报警数据库及media，并单独保管原来的访问码；不要复制待补候选当作已经人工通过的标签。
'''
    if 'prepare_grounding_dino.py --base' not in value:
        value+='\n\n增强预标注默认使用Base模型，还需准备该固定版本；只准备Tiny时，可在Annotations选择原方案Tiny。Comparison的本次正式对比使用Tiny，不需要Base。先完成本节恢复，再按需执行下面的命令：\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\prepare_grounding_dino.py --base\n```\n'
    guide.write_text(value,encoding='utf-8')
    for name in ('docs/report_template_contract.md','artifacts/release/template_work/artifact.md'):
        path=ROOT/name;contract=path.read_text(encoding='utf-8')
        contract=re.sub(r'Cover date2026年\d+月\d+日', 'Cover date2026年11月8日', contract)
        contract=contract.replace('Collected candidate counts reflect the October22 intake; human labels/formal results remain pending.',
                                  'Final labels and two 50-epoch real experiments fill the existing result slots; manual reference masks/worktime and strict data shortfalls remain explicitly pending.')
        path.write_text(contract,encoding='utf-8')
    print(json.dumps({'report_source_updated':True,'guide_updated':True,'unit_tests':tests,'selected_training_id':selected['training_id']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    update(parser.parse_args().workspace.resolve())
