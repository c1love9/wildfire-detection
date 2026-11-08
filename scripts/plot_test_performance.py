"""Plot frozen independent-test evidence without changing labels or running inference."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

COLORS={'YOLO':'#3376a5','GroundingDINO':'#bb7146'}
DISPLAY={'YOLO':'自训YOLO','GroundingDINO':'DINO Tiny'}
MODELS=('YOLO','GroundingDINO')
SPLITS=('train','val','test')
METRICS=('precision','recall','f1','map50','map50_95')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_csv(path,rows):
    if not rows:raise ValueError('Source data cannot be empty')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def audit_split(assets,expected_counts):
    images=[a for a in assets if a['kind']=='image']
    result={'images':len(images),'excluded_videos':sum(a['kind']=='video' for a in assets),'counts':{},'known_groups_per_split':{},'cross_split':{}}
    all_ids=[a['id'] for a in images]
    if len(set(all_ids))!=len(all_ids):raise ValueError('Repeated image ID')
    groups={}
    for split in SPLITS:
        rows=[a for a in images if a['split']==split]
        if len(rows)!=expected_counts[split]:raise ValueError('Frozen split count differs')
        groups[split]={'ids':{a['id'] for a in rows},'sha256':{a['sha256'] for a in rows},
                       'components':{a['component'] for a in rows},'groups':{g for a in rows for g in a['groups']}}
        result['counts'][split]=len(rows)
        result['known_groups_per_split'][split]=len(groups[split]['groups'])
    if sum(result['counts'].values())!=len(images):raise ValueError('Unassigned image')
    for index,first in enumerate(SPLITS):
        for second in SPLITS[index+1:]:
            overlap={key:len(groups[first][key]&groups[second][key]) for key in groups[first]}
            if any(overlap.values()):raise ValueError('Known image/source group crosses splits')
            result['cross_split'][first+'__'+second]=overlap
    result['limitation']='No known-group or exact-file overlap; unknown events, near-duplicates and public pretraining overlap are not ruled out.'
    return result


def check_test_ids(comparison,assets):
    wanted={a['id'] for a in assets if a['kind']=='image' and a['split']=='test'}
    if set(comparison['references'])!=wanted:raise ValueError('Metrics must use exactly the frozen test IDs')
    for model in MODELS:
        if set(comparison['predictions'][model])!=wanted:raise ValueError('Missing or extra model predictions')
        if len(comparison['timings_ms'][model])!=len(wanted):raise ValueError('Timing count differs from test count')
    return wanted


def presence_confusion(references,predictions,threshold):
    counts={'tn':0,'fp':0,'fn':0,'tp':0}
    for key,boxes in references.items():
        actual=bool(boxes);predicted=any(b['confidence']>=threshold for b in predictions[key])
        counts['tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn']+=1
    n=sum(counts.values());counts['images']=n
    counts['accuracy']=(counts['tp']+counts['tn'])/n if n else None
    return counts


def pr_points(references,predictions,label,threshold=.25):
    from comparison import iou
    targets={k:[b['xyxy'] for b in boxes if b['class']==label] for k,boxes in references.items()}
    total=sum(map(len,targets.values()))
    candidates=sorted([(float(b['confidence']),k,b['xyxy']) for k,boxes in predictions.items() for b in boxes if b['class']==label],key=lambda x:-x[0])
    used={k:set() for k in targets};tp=[]
    for _,key,box in candidates:
        available=[(iou(box,gt),j) for j,gt in enumerate(targets[key]) if j not in used[key]]
        matched=False
        if available:
            overlap,j=max(available)
            if overlap+1e-9>=.5:used[key].add(j);matched=True
        tp.append(int(matched))
    scores=np.array([v[0] for v in candidates]);tp=np.array(tp,dtype=float)
    cumulative=np.cumsum(tp);recall=cumulative/max(total,1);precision=cumulative/np.arange(1,len(tp)+1)
    selected=scores>=threshold;true=int(tp[selected].sum());false=int(selected.sum())-true
    point={'tp':true,'fp':false,'fn':total-true,'precision':true/(true+false) if true+false else 0,
           'recall':true/total if total else 0,'instances':total}
    return scores,precision,recall,point


def configure():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if not font.is_file():raise OSError('Chinese font missing; install Microsoft YaHei or configure a CJK font before rendering.')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans'],
        'font.size':10.5,'axes.titlesize':11.5,'axes.labelsize':10.5,'xtick.labelsize':10,'ytick.labelsize':10,
        'legend.fontsize':10,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.8})


def percent_axis(ax):
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('百分比（%）')
    ax.grid(axis='y',color='#e5e9ed',linewidth=.6);ax.set_axisbelow(True)


def finish(fig,title,subtitle,footer):
    fig.suptitle(title,x=.055,y=.98,ha='left',fontsize=15,fontweight='bold')
    fig.text(.055,.90,subtitle,fontsize=10)
    fig.text(.055,.035,footer,va='bottom',fontsize=9,linespacing=1.55,color='#46515b')


def build(package,output):
    if output.exists():raise FileExistsError('Use a new output directory; old figures are retained')
    manifest=read(package/'manifest.json')
    source_paths={
        'comparison_result.json':'03_性能评测/comparison/result.json',
        'comparison_request.json':'03_性能评测/comparison/request.json',
        'formal_metrics.csv':'03_性能评测/formal_metrics.csv',
        'final_dataset_statistics.json':'01_数据与标注/最终数据集/final_dataset_statistics.json',
        'final_image_statistics.csv':'01_数据与标注/最终数据集/final_image_statistics.csv',
        'source_split.json':'01_数据与标注/最终数据集/source_split.json',
        'optimization_evaluation.json':'04_优化对照/optimization/evaluation.json',
        'model_selection.json':'02_模型与训练/model_selection.json'}
    for name in source_paths.values():
        info=manifest['files'][name];path=package/name
        if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:raise OSError('Input differs from package inventory: '+name)
    result=read(package/source_paths['comparison_result.json']);request=read(package/source_paths['comparison_request.json'])
    split=read(package/source_paths['source_split.json']);stats=read(package/source_paths['final_dataset_statistics.json'])
    selection=read(package/source_paths['model_selection.json'])
    split_audit=audit_split(split['assets'],{k:stats['by_split'][k]['images'] for k in SPLITS})
    ids=check_test_ids(result,split['assets'])
    protocol=result['protocol']
    expected={'version':'common_detection_v2','imgsz':640,'batch':1,'precision':'FP32','candidate_conf':.001,'operating_conf':.25,'match_iou':.5}
    if any(protocol[k]!=v for k,v in expected.items()):raise ValueError('Independent-test protocol differs')
    if request['protocol']!=protocol:raise ValueError('Request/result protocols differ')
    if result['dataset_fingerprint']!=request['dataset_fingerprint']:raise ValueError('Dataset fingerprints differ')
    if selection['training_id']!=request['training_id'] or selection['test_used_for_selection']:raise ValueError('Selection must precede test and use validation')
    if datetime.fromisoformat(selection['selected_at'])>=datetime.fromisoformat(request['created_at']):raise ValueError('Selection chronology differs')
    if sha(package/'02_模型与训练/baseline/output/weights/best.pt')!=request['yolo']['sha256']:raise OSError('YOLO weight differs')
    inventory=[]
    for key in sorted(ids):
        for suffix,category in (('.png','images'),('.txt','labels')):
            filename=key+suffix;path=package/'03_性能评测/comparison/input'/filename
            final=package/'01_数据与标注/最终数据集'/category/'test'/filename
            value=sha(path)
            if value!=result['inventory']['input/'+filename] or value!=sha(final):raise OSError('Test input snapshot differs')
            inventory.append({'asset_id':key,'kind':category,'filename':filename,'sha256':value})
    from comparison import evaluate_predictions
    optimization=read(package/source_paths['optimization_evaluation.json'])
    if optimization['reference']!=result['references'] or optimization['protocol']!=protocol:
        raise ValueError('Optimization must use the same frozen test references/protocol')
    for variant in ('baseline','multiscale'):
        saved=optimization['results'][variant]
        computed=evaluate_predictions(optimization['reference'],saved['predictions'])
        if set(saved['predictions'])!=ids:raise ValueError('Optimization test IDs differ')
        for key in METRICS:
            if not math.isclose(computed['overall'][key],saved['overall'][key],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Optimization metric differs')
    if len(ids)!=76 or {k:stats['by_split'][k]['images'] for k in SPLITS}!={'train':484,'val':140,'test':76}:
        raise ValueError('This figure contract is pinned to the 700-image release; establish a new contract for a new split')
    checked={};curves={};presence={};overall=[];per_class=[]
    for model in MODELS:
        computed=evaluate_predictions(result['references'],result['predictions'][model]);saved=result['models'][model]
        for key in METRICS:
            if not math.isclose(computed['overall'][key],saved['overall'][key],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Recomputed metric differs')
        times=np.asarray(result['timings_ms'][model],dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times<=0):raise ValueError('Invalid timing')
        if not math.isclose(float(times.mean()),saved['mean_ms_per_image'],rel_tol=1e-10):raise ValueError('Mean timing differs')
        checked[model]=computed
        overall.append({'model':model,'test_images':len(ids),**saved['overall'],'mean_ms':float(times.mean()),
                        'fps':1000/float(times.mean()),'median_ms':float(np.median(times)),
                        'q1_ms':float(np.quantile(times,.25)),'q3_ms':float(np.quantile(times,.75)),
                        'parameters':saved['parameters'],'weight_MiB':saved['file_bytes']/1024**2})
        for row in computed['per_class']:
            scores,p,r,point=pr_points(result['references'],result['predictions'][model],row['class'])
            if any(point[k]!=row[k] for k in ('tp','fp','fn')):raise ValueError('PR operating point differs')
            curves[model,row['class']]=(scores,p,r,point)
            per_class.append({'model':model,**row})
        presence[model]=presence_confusion(result['references'],result['predictions'][model],.25)
    output.mkdir(parents=True);data=output/'source_data';data.mkdir();figures=output/'figures';figures.mkdir()
    for name,source in source_paths.items():shutil.copy2(package/source,data/name)
    save_csv(data/'test_input_inventory.csv',inventory)
    save_csv(data/'detection_metrics.csv',overall);save_csv(data/'per_class_metrics.csv',per_class)
    save_csv(data/'image_presence_accuracy.csv',[{'model':model,**value} for model,value in presence.items()])
    save_csv(data/'latency_per_image.csv',[{'model':model,'asset_id':key,'latency_ms':ms}
        for model in MODELS for key,ms in zip(result['references'],result['timings_ms'][model])])
    save_csv(data/'split_distribution.csv',[{'split':s,**stats['by_split'][s]} for s in SPLITS])
    write(output/'split_audit.json',split_audit)
    source_hashes={name:{'package_member':relative,'sha256':sha(data/name)} for name,relative in source_paths.items()}
    pr_rows=[]
    for (model,label),(scores,p,r,point) in curves.items():
        pr_rows.extend({'model':model,'class':label,'candidate_rank':i+1,'confidence':float(score),
                        'precision':float(precision),'recall':float(recall)} for i,(score,precision,recall) in enumerate(zip(scores,p,r)))
    save_csv(data/'test_pr_points.csv',pr_rows)
    configure();records=[]
    with (data/'final_image_statistics.csv').open(encoding='utf-8-sig',newline='') as stream:source_data_rows=list(csv.DictReader(stream))
    if len(source_data_rows)!=700 or {r['asset_id'] for r in source_data_rows}!={a['id'] for a in split['assets'] if a['kind']=='image'}:
        raise ValueError('Per-image CSV does not cover the frozen images')
    with (data/'formal_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:formal_rows=list(csv.DictReader(stream))
    formal_by_name={r['experiment']:r for r in formal_rows}
    for name,values in [('baseline',optimization['results']['baseline']['overall']),('multiscale',optimization['results']['multiscale']['overall']),
                        ('comparison_YOLO',result['models']['YOLO']['overall']),('comparison_GroundingDINO',result['models']['GroundingDINO']['overall'])]:
        for key in METRICS:
            if not math.isclose(float(formal_by_name[name][key]),values[key],rel_tol=1e-10):raise ValueError('Formal CSV metric differs')
    methods={r['model']:r for r in overall}
    pdf=PdfPages(output/'独立测试性能图表汇总.pdf')

    def export(fig,stem,title,claim,caption):
        fig.canvas.draw()
        from matplotlib.text import Text
        renderer=fig.canvas.get_renderer();page=fig.bbox
        outside=[];sizes=[]
        for item in fig.findobj(match=Text):
            if not item.get_visible() or not item.get_text().strip():continue
            bounds=item.get_window_extent(renderer);sizes.append(item.get_fontsize())
            if bounds.x0<page.x0-1 or bounds.y0<page.y0-1 or bounds.x1>page.x1+1 or bounds.y1>page.y1+1:
                outside.append(item.get_text())
        if outside:raise ValueError('Text outside figure page: '+repr(outside))
        for ext in ('svg','pdf','png'):
            fig.savefig(figures/(stem+'.'+ext),dpi=360 if ext=='png' else 150)
        pdf.savefig(fig)
        records.append({'stem':stem,'title':title,'claim':claim,'caption':caption,
                        'text_inside_page':True,'minimum_text_pt':min(sizes),
                        'minimum_text_pt_at_180mm':min(sizes)*(180/25.4)/fig.get_figwidth()})
        plt.close(fig)

    # One evaluation, two metric families: fixed-threshold operating point and AP sweep.
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.8));fig.subplots_adjust(left=.08,right=.98,top=.77,bottom=.29,wspace=.30)
    for ax,fields,labels,title in [(axes[0],('precision','recall','f1'),('Precision\n查准率','Recall\n召回率','F1'),'a  固定置信度0.25'),
                                   (axes[1],('map50','map50_95'),('mAP50','mAP50–95'),'b  AP候选阈值0.001')]:
        x=np.arange(len(fields))
        for j,model in enumerate(MODELS):
            values=[methods[model][key]*100 for key in fields]
            bars=ax.bar(x+(j-.5)*.34,values,width=.32,color=COLORS[model],label=DISPLAY[model])
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=9)
        ax.set_xticks(x,labels);percent_axis(ax);ax.set_title(title,loc='left');ax.legend(loc='upper right')
    finish(fig,'独立测试集检测性能','同一冻结test：76幅图像，fire 58框、smoke 84框；自训YOLO与未微调DINO Tiny',
           'P/R/F1按类别平均；同类一对一IoU≥0.50。AP在IoU 0.50～0.95评估。\n单次训练seed=42、单一划分；无跨种子误差条或显著性结论。\n测试标签经人工复核；DINO曾参与预标注，仍须披露标注依赖及未知公开预训练重叠。')
    export(fig,'01_独立测试检测性能','独立测试检测性能','YOLO固定工作点更精确，DINO召回和AP较高；不存在所有指标同时占优。',
           '两种模型在相同76幅test上的P/R/F1及mAP；不是训练或验证集成绩。')

    fig,axes=plt.subplots(1,3,figsize=(9.4,4.5));fig.subplots_adjust(left=.09,right=.98,top=.76,bottom=.24,wspace=.55)
    ax=axes[0];values=[methods[m]['fps'] for m in MODELS]
    bars=ax.barh([1,0],values,color=[COLORS[m] for m in MODELS],height=.45)
    ax.set_yticks([1,0],[DISPLAY[m] for m in MODELS]);ax.set_xlim(0,max(values)*1.28);ax.set_xlabel('FPS（幅/秒）');ax.set_title('a  平均耗时折算FPS',loc='left')
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=10)
    ax=axes[1];bp=ax.boxplot([result['timings_ms'][m] for m in MODELS],positions=[2,1],orientation='horizontal',tick_labels=[DISPLAY[m] for m in MODELS],patch_artist=True,widths=.45,
                           medianprops={'color':'#243747','linewidth':1},flierprops={'markeredgecolor':'#667580','markersize':4})
    for patch,m in zip(bp['boxes'],MODELS):patch.set_facecolor(COLORS[m]);patch.set_alpha(.65)
    ax.set_xlim(left=0);ax.set_xlabel('逐图耗时（ms）');ax.set_title('b  全部76幅耗时分布',loc='left')
    ax=axes[2];ax.axis('off');ax.set_title('c  模型规模',loc='left')
    rows=[[DISPLAY[m],f"{methods[m]['parameters']/1e6:.2f}",f"{methods[m]['weight_MiB']:.2f}"] for m in MODELS]
    table=ax.table(cellText=rows,colLabels=['模型','参数量\n百万','权重\nMiB'],cellLoc='center',bbox=[-.05,.25,1.1,.55],colWidths=[.40,.30,.30])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():cell.set_edgecolor('#d9dfe4');cell.set_facecolor('#e9eff4' if row==0 else 'white')
    finish(fig,'独立测试速度与模型资源','同一GPU、640×640、batch=1、FP32；只使用同一次YOLO/DINO对比记录',
           '耗时含预处理、模型、后处理；排除加载、读盘、画框、界面及SAM。权重大小不是GPU显存。\n箱体Q1～Q3，中线中位数，须至1.5×IQR规则内的极值，散点为离群值；不是跨种子置信区间。')
    export(fig,'02_速度与资源','速度与模型资源','本次设备上YOLO更快且规模更小，但速度不是完整报警系统帧率。',
           'FPS=1000/平均毫秒；76个逐图耗时为描述性分布，不是76次独立硬件重复实验。')

    fig,axes=plt.subplots(1,3,figsize=(9.4,5.0));fig.subplots_adjust(left=.07,right=.97,top=.75,bottom=.29,wspace=.48)
    ax=axes[0];positive=[stats['by_split'][s]['positive'] for s in SPLITS];negative=[stats['by_split'][s]['negative'] for s in SPLITS];x=np.arange(3)
    first=ax.bar(x,positive,color='#7fa3bf',label='含火烟');second=ax.bar(x,negative,bottom=positive,color='#cfd7dd',label='正常')
    for i,(pos,neg) in enumerate(zip(positive,negative)):
        ax.text(i,pos/2,str(pos),ha='center',va='center',fontsize=10)
        if neg>5:ax.text(i,pos+neg/2,str(neg),ha='center',va='center',fontsize=10)
        else:ax.annotate(f'正常{neg}',(i,pos+neg),xytext=(12,6),textcoords='offset points',fontsize=9)
        ax.text(i,pos+neg+(48 if neg<=5 else 27),str(pos+neg),ha='center',fontsize=10,fontweight='bold')
    ax.set_ylim(0,600);ax.set_xticks(x,['train','val','test']);ax.set_ylabel('图片数量');ax.set_title('a  正负样本分布',loc='left');ax.legend(loc='upper right',fontsize=9)
    ax=axes[1]
    for j,(label,color) in enumerate([('fire','#bd7956'),('smoke','#7b8b9c')]):
        v=[stats['by_split'][s]['boxes_'+label] for s in SPLITS];bars=ax.bar(x+(j-.5)*.34,v,width=.32,color=color,label=label);ax.bar_label(bars,padding=3,fontsize=9)
    ax.set_ylim(0,350);ax.set_xticks(x,['train','val','test']);ax.set_ylabel('人工标注目标框');ax.set_title('b  类别覆盖与不平衡',loc='left');ax.legend(loc='upper right',fontsize=9)
    ax=axes[2];sources=('dfire','firesense','kmu','flame2');matrix=np.array([[sum(r['source']==s and r['split']==split for r in source_data_rows) for split in SPLITS] for s in sources])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=int(matrix.max()),aspect='auto')
    for i in range(4):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>matrix.max()*.6 else '#263746')
    ax.set_xticks(x,['train','val','test']);ax.set_yticks(range(4),['D-Fire','FIRESENSE','KMU','FLAME 2']);ax.set_title('c  来源分布（图片数）',loc='left')
    finish(fig,'数据划分与独立测试范围','700幅：train 484 / val 140 / test 76；本组性能图只使用test，划分图不提供train或val成绩',
           'test中75幅含火烟、正常仅1幅；val仅11个smoke框，类别覆盖明显不均。\n图片ID、文件哈希、已知来源组跨集合交集均为0；不能排除未知原事件、近重复或公开预训练重叠。\n76幅test仅来自2个已知采集组，不代表76次独立火灾事件；300段视频不参与本次图像AP。')
    export(fig,'03_划分与类别来源分布','划分与类别来源分布','已知组隔离但类别和来源分布不均，正常测试支持不足。','使用全部700幅统计划分，性能指标仅来自其中76幅test；来源不等同于独立摄像头。')

    fig,axes=plt.subplots(1,2,figsize=(9.4,4.5));fig.subplots_adjust(left=.08,right=.96,top=.76,bottom=.25,wspace=.40)
    ax=axes[0];x=np.arange(2)
    for j,m in enumerate(MODELS):
        vals=[row['recall']*100 for row in checked[m]['per_class']];bars=ax.bar(x+(j-.5)*.34,vals,width=.32,color=COLORS[m],label=DISPLAY[m]);ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=4,fontsize=10)
    ax.set_xticks(x,['fire（58框）','smoke（84框）']);percent_axis(ax);ax.set_title('a  分类别召回率',loc='left');ax.legend(loc='upper right')
    ax=axes[1];matrix=np.array([[row[k] for k in ('tp','fp','fn')] for m in MODELS for row in checked[m]['per_class']])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=int(matrix.max()),aspect='auto')
    for i in range(4):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=11,color='white' if matrix[i,j]>matrix.max()*.65 else '#263746')
    ax.set_xticks(range(3),['TP 检出','FP 误检','FN 漏检']);ax.set_yticks(range(4),['YOLO · fire','YOLO · smoke','DINO · fire','DINO · smoke']);ax.set_title('b  检测框错误计数',loc='left')
    finish(fig,'分类别检出与错误分析','相同test，固定conf=0.25、同类一对一IoU≥0.50；TP / FP / FN为目标框计数',
           'DINO在两类上的召回更高，同时误检框更多。FP来自全部测试图，不只是正常图片。\n漏检数按人工参考框计算；图片级判断与目标框级判断不能混用。')
    export(fig,'04_分类别错误分析','分类别检出与错误分析','DINO多检出目标的同时产生更多误检，两个类别均需保留错误数量。','fire参考58框、smoke参考84框；展示全部类别，不选择性隐藏困难类别。')

    fig,axes=plt.subplots(1,2,figsize=(9.4,4.5));fig.subplots_adjust(left=.08,right=.97,top=.76,bottom=.26,wspace=.30)
    for ax,label,n in zip(axes,('fire','smoke'),(58,84)):
        for m in MODELS:
            scores,p,r,point=curves[m,label]
            ap=next(row['map50'] for row in checked[m]['per_class'] if row['class']==label)
            ax.step(r*100,p*100,where='post',color=COLORS[m],lw=1.4,label=f'{DISPLAY[m]}  AP50={ap*100:.2f}%')
            ax.scatter([point['recall']*100],[point['precision']*100],marker='o',s=33,color=COLORS[m],edgecolor='white',linewidth=.6,zorder=4)
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='Recall（%）',ylabel='Precision（%）',title=f'{label}（人工参考{n}框）');ax.legend(loc='upper right',fontsize=9);ax.grid(color='#e5e9ed',lw=.6)
    finish(fig,'独立测试集Precision–Recall曲线','冻结候选阈值0.001；同类一对一IoU≥0.50，圆点标出conf=0.25工作点',
           '曲线展示全部实际候选排序产生的PR阶梯，不额外平滑、不补造到100%召回的外推段。\nAP50使用冻结评测器的插值积分定义；不是本图像素面积，也不是mAP50–95。')
    export(fig,'05_独立测试PR曲线','独立测试PR曲线','固定阈值的P/R与跨阈值AP回答不同问题，低阈值下的取舍必须可核查。','两类分别显示YOLO/DINO原始PR阶梯，工作点和AP来自同一套冻结预测。')

    fig,axes=plt.subplots(1,2,figsize=(9.4,4.8));fig.subplots_adjust(left=.12,right=.94,top=.73,bottom=.30,wspace=.48)
    for ax,m in zip(axes,MODELS):
        c=presence[m];matrix=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=75)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=16,color='white' if matrix[i,j]>40 else '#263746')
        ax.set_xticks([0,1],['判正常','判有火烟']);ax.set_yticks([0,1],['实际正常（1幅）','实际有火烟（75幅）']);ax.set_xlabel('预测图像类别');ax.set_title(f'{DISPLAY[m]}  Accuracy={c["accuracy"]*100:.2f}%',fontsize=11.5)
    finish(fig,'补充分析 图像级Accuracy与混淆矩阵','至少有一个置信度≥0.25的fire/smoke框即判“有火烟”；此处不检验框的定位是否正确',
           'Accuracy=(TP+TN)/76；这是图像有无目标的判别，不能替代目标检测Precision、Recall或mAP。\n正常图仅1幅：全部判有火烟也能得到98.68% Accuracy，故高Accuracy不能证明误报控制可靠。\n没有负样本数量足够的独立测试，不能据此声称实际误报率达标。')
    export(fig,'06_补充图像级Accuracy','补充图像级Accuracy','DINO图像级98.68%与全判正基线相同，单看Accuracy会掩盖正常图支持不足。','额外提供明确口径的Accuracy；不将其混称成检测框准确率或完整报警系统准确率。')

    fig,axes=plt.subplots(1,2,figsize=(9.4,4.8));fig.subplots_adjust(left=.08,right=.96,top=.76,bottom=.30,wspace=.40)
    ax=axes[0];labels=['Precision','Recall','F1','mAP50','mAP50–95'];x=np.arange(5)
    for j,(name,color) in enumerate([('baseline','#3376a5'),('multiscale','#8099ad')]):
        values=[float(formal_by_name[name][k])*100 for k in METRICS]
        bars=ax.bar(x+(j-.5)*.34,values,width=.32,color=color,label='固定尺寸' if name=='baseline' else '多尺度')
    percent_axis(ax);ax.set_xticks(x,['P','R','F1','mAP50','mAP\n50–95']);ax.set_title('a  同test上的策略对照',loc='left');ax.legend(loc='upper right',fontsize=9)
    ax=axes[1];delta=[(float(formal_by_name['multiscale'][k])-float(formal_by_name['baseline'][k]))*100 for k in METRICS]
    bars=ax.barh(np.arange(5),delta,color=['#568a7a' if value>=0 else '#bb7146' for value in delta],height=.55)
    ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.axvline(0,color='#46515b',lw=.8);ax.set_xlim(-5,17);ax.set_xticks([-5,0,5,10,15]);ax.set_xlabel('多尺度 − 固定尺寸（百分点）');ax.set_title('b  改善与退化均保留',loc='left')
    ax.bar_label(bars,labels=[f'{v:+.2f}' for v in delta],padding=4,fontsize=10)
    finish(fig,'训练策略的独立测试对照','同初始化、数据、seed=42和50轮；仅multi_scale开关不同，统一640×640评测',
           '多尺度F1增加3.77个百分点、mAP50增加1.49个百分点，但mAP50–95下降0.89个百分点。\n主模型已在查看test前由val选定baseline；不在看到test后重新选模型或调阈值。\n单seed、单划分对照，无显著性或普遍有效结论；仅方法策略结果，不混用另次速度记录。')
    export(fig,'07_训练策略测试对照','训练策略的独立测试对照','多尺度并非全面改善，较严格的mAP50–95反而下降。','附加展示formal_metrics中的训练策略两行，基线不同运行记录不视为两个不同模型。')
    pdf.close()

    for record in records:
        record['exports']={ext:{'file':'figures/'+record['stem']+'.'+ext,'sha256':sha(figures/(record['stem']+'.'+ext))} for ext in ('png','svg','pdf')}
    summary={'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),'source_release':manifest['release_id'],
             'source_commit':manifest['source_commit'],'test_images':len(ids),'test_fire_instances':58,'test_smoke_instances':84,
             'test_positive_images':75,'test_normal_images':1,'formal_metrics_recomputed_from_predictions':True,
             'input_files_verified':len(inventory),'source_files':source_hashes,'protocol':protocol,'figures':records,
             'optimization_metrics_recomputed':True,'row_exclusions':{'train_val_performance_rows':'Not used for independent-test metrics; all700 images used only for split figures.',
                               'videos':300,'reason':'Video clips have no full per-frame truth and are not the frozen image AP test.'},
             'presence_accuracy':presence,'all_positive_baseline_accuracy':75/76,'split_audit':split_audit,
             'source_data_modified':False,'new_inference_or_training_run':False,
             'uncertainty':'One seed/split. No confidence intervals or significance claims. Latency box plots describe 76 measured images, not independent benchmark repeats.',
             'formal_acceptance_complete':False}
    write(output/'figure_verification.json',summary)
    shutil.copy2(Path(__file__),output/'plot_test_performance.py')
    shutil.copy2(ROOT/'docs/performance_figure_contract.md',output/'图表统计约定.md')
    captions=['# 独立测试性能图表说明','',
        '本附件用于原题“具体量化结果、合理划分、禁止训练集指标充数”的性能部分。图表均来自已保存的冻结test预测，不重训、不重跑、不调整阈值。',
        '测试76幅：75幅含火烟、1幅正常；fire58框、smoke84框。主图使用目标检测P/R/F1和mAP，图06另外给出定义清楚的图像级Accuracy。',
        '图像级Accuracy为YOLO80.26%、DINO98.68%；全判有火烟也为98.68%，因此不能据此声称实际误报率可靠。',
        '速度采用同一次比较记录，YOLO48.09FPS、DINO2.27FPS；不是SAM开启后的完整系统FPS。',
        '划分统计使用全部700幅，训练和验证的成绩未用于测试性能图。已知组隔离检查通过，但test仅2个已知采集组且正常图仅1幅，分布与覆盖不充分。',
        'DINO曾参与自动预标注；最终参考来自人工复核。必须披露该依赖，不能声称排除了所有标签偏倚或公开预训练重叠。','']
    for i,rec in enumerate(records,1):captions += [f'## 图{i} {rec["title"]}',rec['caption'],rec['claim'],'',f'![{rec["title"]}](figures/{rec["stem"]}.png)','']
    captions += ['## 文件使用','PNG用于Word和演示；SVG保留可编辑文字，PDF用于打印，合并PDF共7页。建议按约180mm宽插图；不可过度缩小使字体难读。',
        'source_data保留原始JSON/CSV与导出的图表数据；figure_verification.json记录源文件哈希、数据范围与指标重算结果。',
        '此附件加入03性能评测作为补充即可。原提交包及其manifest保持不变，不直接覆盖其中已有文件；需要重新合包时统一生成新版本清单。',
        '本次增加图表不会自动补齐真实正常样本、严格场景确认、跨设备证据或分割面积误差。',
        '复现：在原项目依赖环境执行脚本，--package指向完整提交材料目录，--output使用新的空目录。脚本依赖项目统一评测器。','']
    (output/'README_图表使用说明.md').write_text('\n\n'.join(captions),encoding='utf-8')
    cards=''.join('<section><h2>'+html.escape(rec['title'])+'</h2><p>'+html.escape(rec['caption'])+'</p><img src="figures/'+rec['stem']+'.png" alt="'+html.escape(rec['title'])+'"><p><a href="figures/'+rec['stem']+'.pdf">PDF</a> · <a href="figures/'+rec['stem']+'.svg">可编辑SVG</a></p></section>' for rec in records)
    (output/'打开图表.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>独立测试性能图表</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6f8;color:#203446;max-width:1150px;margin:35px auto;padding:0 22px}section{background:white;padding:22px 28px;margin:24px 0;border:1px solid #dde5eb}h1,h2{line-height:1.5}img{display:block;width:100%;height:auto}a{color:#28658d}</style><h1>独立测试性能图表</h1><p>同76幅冻结test；75幅火烟、1幅正常。检测框Precision与图像级Accuracy分别定义。正常样本过少，不能声称实际误报率可靠。</p><p><a href="独立测试性能图表汇总.pdf">打开7页汇总PDF</a> · <a href="README_图表使用说明.md">图注与统计说明</a></p>'+cards+'</html>',encoding='utf-8')
    print(json.dumps({'status':'passed','output':str(output),'figures':len(records),'test_images':len(ids),'presence_accuracy':presence},ensure_ascii=False))
    return summary


def bundle(output):
    files={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'manifest.json',{'schema_version':1,'kind':'independent_test_figure_supplement','files':files})
    (output/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted({**files,'manifest.json':{'sha256':sha(output/'manifest.json')}}.items())),encoding='utf-8')
    archive=output.parent/(output.name+'.zip')
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise OSError('Archive CRC failed')
        for name,info in files.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=info['sha256']:raise OSError('Archive content differs')
    write(output.parent/'supplement_verification.json',{'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'payload_files':len(files)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.package.resolve(),args.output.resolve())
