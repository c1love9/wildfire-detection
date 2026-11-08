"""Readable report curves from all 50 real epochs; no smoothing or resampling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(workspace):
    root=Path(__file__).resolve().parents[1]
    formal=json.loads((workspace/'formal_experiments.json').read_text(encoding='utf-8'))
    target=workspace/'figures';target.mkdir(exist_ok=True)
    histories={}
    for variant,identifier in formal['jobs'].items():
        source=root/'data/fire_dataset/training'/identifier/'output/results.csv'
        with source.open(encoding='utf-8') as stream:
            histories[variant]=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        assert len(histories[variant])==50
    contract={'conclusion':'Training losses decline while validation remains variable; 50 epochs alone do not prove convergence.',
              'panels':{'a':'All train box/class/DFL losses','b':'All validation box/class/DFL losses','c':'Validation mAP50 and mAP50-95'},
              'archetype':'quantitative grid','backend':'Python matplotlib','dimensions_inches':[7.5,2.85],
              'report_width_inches':5.65,'minimum_rendered_font_pt':10*5.65/7.5,
              'seeds':[42],'uncertainty':'One training trace per variant; no replicate CI or significance test.',
              'row_exclusions':0,'smoothing':False,'interpolation':'Straight segments between recorded epochs',
              'other_variables':'Precision/recall/learning-rate columns are retained in source CSV; this figure focuses on requested losses and mAP.',
              'source_data_sha256':{}}
    (target/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':12,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,
                         'legend.fontsize':10,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.8})
    limits={}
    for prefix in ('train','val'):
        maximum=max(row[prefix+'/'+loss+'_loss'] for rows in histories.values() for row in rows for loss in ('box','cls','dfl'))
        limits[prefix]=math.ceil(maximum*1.07*2)/2
    maximum=max(row[metric] for rows in histories.values() for row in rows for metric in ('metrics/mAP50(B)','metrics/mAP50-95(B)'))
    limits['map']=math.ceil(maximum*1.08*20)/20
    for variant,rows in histories.items():
        fig,axes=plt.subplots(1,3,figsize=(7.5,2.85),layout='constrained')
        epochs=[row['epoch'] for row in rows]
        for axis,prefix,title,panel in zip(axes[:2],('train','val'),('Train loss','Validation loss'),('a','b')):
            for label,color in (('box','#356d9d'),('cls','#c27832'),('dfl','#6e7983')):
                axis.plot(epochs,[row[prefix+'/'+label+'_loss'] for row in rows],color=color,lw=1.5,label=label.upper() if label=='dfl' else label.capitalize())
            axis.set_title(title);axis.set(xlabel='Epoch',ylabel='Loss',xlim=(1,50),ylim=(0,limits[prefix]));axis.legend(loc='upper right')
            axis.text(-.18,1.08,panel,transform=axis.transAxes,fontweight='bold',fontsize=12)
        axis=axes[2]
        axis.plot(epochs,[row['metrics/mAP50(B)'] for row in rows],color='#356d9d',lw=1.5,label='mAP50')
        axis.plot(epochs,[row['metrics/mAP50-95(B)'] for row in rows],color='#c27832',lw=1.5,ls='--',label='mAP50-95')
        axis.set_title('Validation mAP');axis.set(xlabel='Epoch',ylabel='AP',xlim=(1,50),ylim=(0,limits['map']));axis.legend(loc='upper left')
        axis.text(-.18,1.08,'c',transform=axis.transAxes,fontweight='bold',fontsize=12)
        for axis in axes:axis.set_xticks([1,10,20,30,40,50]);axis.grid(axis='y',color='#e4e6e8',lw=.5)
        fig.savefig(target/(variant+'_curves.svg'))
        fig.savefig(target/(variant+'_curves.pdf'))
        fig.savefig(target/(variant+'_curves.png'),dpi=600)
        plt.close(fig)
        source=root/'data/fire_dataset/training'/formal['jobs'][variant]/'output/results.csv'
        (target/(variant+'_source.csv')).write_bytes(source.read_bytes())
        contract['source_data_sha256'][variant]=hashlib.sha256(source.read_bytes()).hexdigest()
    (target/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    plot(parser.parse_args().workspace.resolve())
