"""Apply the owner's measured review-rate estimate without inventing mask truth."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def worktime_section(value):
    manual=value['manual_baselines'];n=value['target_images']
    return f'''辅助复核单幅有效用时由项目负责人实测提供，约{value['assisted']['seconds_per_image']:g}秒/幅。未保存该次实测的样本数量和逐图配对日志，因此以下总工时为推算，不写成700幅全量实测。纯人工基线来自同一200幅的A/B原始独立标注计时：A共{manual['A']['total_seconds']:.3f}秒，B共{manual['B']['total_seconds']:.3f}秒。两人的总时间除以400次独立提交，得到单人单幅均值{value['manual_mean_seconds_per_image']:.4f}秒，不将两人的工时相加后除以200来夸大节省。

|工时方案|单幅秒|{n}幅预计分钟|
|---|---:|---:|
|纯人工A基线|{manual['A']['mean_seconds_per_image']:.4f}|{value['per_annotator_reference']['A']['estimated_manual_seconds']/60:.2f}|
|纯人工B基线|{manual['B']['mean_seconds_per_image']:.4f}|{value['per_annotator_reference']['B']['estimated_manual_seconds']/60:.2f}|
|纯人工合并均值|{value['manual_mean_seconds_per_image']:.4f}|{value['estimated_manual_seconds']/60:.2f}|
|预标注后人工复核|约{value['assisted']['seconds_per_image']:g}|约{value['estimated_assisted_seconds']/60:.2f}|

总工时按“图像数×单幅秒数÷60”估算；节省率为“1－辅助复核单幅时间÷纯人工单幅时间”。据此{n}幅预计节省{value['estimated_saved_seconds']/60:.2f}分钟，约{value['estimated_saved_fraction']*100:.2f}%。分别使用A/B基线时，预计节省43.86～59.07分钟、42.92%～50.31%；这是两名标注者的差异范围，不是置信区间。估算仅覆盖查看和修正标注的人工作业，不含模型推理、环境准备、导出和无关等待；假定抽样与全部图片的难度、质量标准相近。未进行同人同图的随机配对实验，不据此声称严格因果提升。原始计时和人工提交不改写，也不生成700条假计时。'''


def apply_report(text, value):
    text=text.replace('用户同意人工参考掩码、实际面积误差和有效辅助复核工时暂留待补；',
                      '辅助复核工时采用负责人实测约5秒/幅进行估算；现有框IoU用于定位一致性，分割IoU和面积误差仍待独立参考；')
    text=text.replace('参考掩码必须来自独立人工绘制，并与原图尺寸一致。',
                      '若采用掩码IoU评价SAM分割，本系统要求参考来自独立人工绘制或可靠分割真值，并与原图尺寸一致。原题要求面积估计误差分析，未指定必须手绘掩码、使用Dice或标多少幅。')
    iou_note='现有框IoU仅表示检测框定位一致性，不是SAM像素掩码IoU；不能用1减框IoU充当面积占比误差。分割IoU和实际面积误差继续标为待补。'
    marker='质量报告保留同类一对一匹配框IoU'
    if iou_note not in text:
        position=text.index(marker);text=text[:position]+iou_note+'\n\n'+text[position:]
    begin='<!-- worktime-estimate:start -->';end='<!-- worktime-estimate:end -->'
    block=begin+'\n'+worktime_section(value)+'\n'+end
    if begin in text:
        left=text.index(begin);right=text.index(end,left)+len(end);text=text[:left]+block+text[right:]
    else:
        anchor='工作计时需主动开始与暂停'
        left=text.index(anchor);right=text.index('\n\n',left)
        text=text[:left]+block+text[right:]
    text=text.replace('|面积和工时|工具已实现，无人工参考和有效辅助计时|用户同意待补，指标不填写|',
                      '|工时估算|负责人实测约5秒/幅，结合A/B原始计时|700幅估计节省51.46分钟，46.87%|\n|面积分割|现有框IoU已列，暂无人工分割参考|分割IoU和面积误差仍待补|')
    text=text.replace('本轮仍不能标记为“所有原题要求完成”。严格场景及早期火星/真实燃烧阶段、数量差额、独立人工参考掩码和有效工时尚未关闭。最后两项由用户明确同意暂留待补，身份信息暂不填写，教师成绩及签章留空，现场验收由实际操作者和教师完成。',
                      '本轮仍不能标记为“所有原题要求完成”。严格场景及早期火星/真实燃烧阶段、数量差额、分割IoU与实际面积误差仍待补。工时估算已依据负责人提供的实测单幅用时补充，说明样本量未提供和总量推算的限制。身份信息暂不填写，教师成绩及签章留空，现场验收由实际操作者和教师完成。')
    text=text.replace('它们不能替代缺失的人工掩码或工时实测。','工时另按负责人实测约5秒/幅估算并披露限制；已有框IoU不能替代缺失的分割参考。')
    text=text.replace('取得独立人工参考掩码和受控工时，再冻结新数据版本。','补充独立分割参考后分析分割IoU及面积误差，再冻结新数据版本；若需提高工时估算可信度，可补充配对计时样本。')
    return text


def apply_formal(text, value):
    old='用户同意暂留待补：独立人工参考掩码与实际面积误差、有效辅助复核工时及节省率。'
    text=text.replace(old,'工时已按负责人实测约5秒/幅估算：700幅辅助复核约58.33分钟，对比A/B单人基线均值约109.80分钟，估计节省51.46分钟、46.87%；实测样本量未提供，未进行全量配对计时。现有框IoU仅作定位一致性，分割IoU与面积误差仍缺独立参考。')
    return text


def main():
    value=json.loads((ROOT/'reports/worktime_estimate_20261108.json').read_text(encoding='utf-8'))
    path=ROOT/'docs/template_report.md';path.write_text(apply_report(path.read_text(encoding='utf-8'),value),encoding='utf-8')
    path=ROOT/'docs/formal_results.md';path.write_text(apply_formal(path.read_text(encoding='utf-8'),value),encoding='utf-8')
    content='# 辅助复核工时与检测框IoU补充说明\n\n'+worktime_section(value)+'\n\n## 检测框IoU的解释\n\n'
    boxes=json.loads((ROOT/'reports/annotation_iou_20261108.json').read_text(encoding='utf-8'))
    content+='同一200幅原始A/B参考保持不变。按类别一致、置信度从高到低、一对一IoU≥0.5匹配；平均IoU只计算匹配框，须同时报告未匹配数量。\n\n|候选版本及参考|匹配框平均IoU|匹配数|未匹配预测框|未匹配人工框|\n|---|---:|---:|---:|---:|\n'
    for variant,label in (('original_frozen','原Tiny'),('enhanced_latest','增强预标注')):
        for role,row in boxes['results'][variant].items():
            content+=f"|{label}对{role}|{row['matched_mean_box_iou']:.4f}|{row['matched_pairs']}|{row['unmatched_prediction_boxes']}|{row['unmatched_reference_boxes']}|\n"
    content+='\n这些数值属于检测框定位一致性，不是SAM掩码IoU。增强候选的P/R改善而匹配IoU略降，应同时披露；没有人工分割参考时，mask IoU及面积占比误差保持未知。原题没有要求额外强制标700幅掩码。本轮不重训模型、不改标签、不伪造参考掩码。\n'
    (ROOT/'docs/worktime_and_iou.md').write_text(content,encoding='utf-8')
    contract=ROOT/'docs/report_template_contract.md'
    text=contract.read_text(encoding='utf-8').replace('manual reference masks/worktime and strict data shortfalls remain explicitly pending.',
            'review worktime is estimated from the owner-reported approximately 5 seconds per image and immutable A/B timers; mask IoU/area analysis and strict data shortfalls remain pending.')
    contract.write_text(text,encoding='utf-8')


if __name__=='__main__':main()
