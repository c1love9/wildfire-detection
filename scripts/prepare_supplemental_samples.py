"""Prepare 7 normal and 5 curated-video candidates, without approving labels."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training import read_json,write_json


def prepare(workspace):
    out=workspace/'pending_samples'
    out.mkdir(exist_ok=False)
    normal=workspace/'normal_candidates'
    candidates=read_json(normal/'candidates.json')
    selected=[candidates[index-1] for index in (3,7,9,10,11,17,18)]
    records=[]
    for index,row in enumerate(selected,1):
        target=out/'normal'/row['filename'];target.parent.mkdir(exist_ok=True);shutil.copy2(normal/row['filename'],target)
        records.append({**row,'number':f'N{index:02}','file':target.relative_to(out).as_posix(),
                        'accounting_origin':'public','intended_content':'normal',
                        'source_url':'https://github.com/gaia-solutions-on-demand/DFireDataset',
                        'approved_by_user':False})
    workflow=read_json(workspace/'workflow.json')
    audit=read_json(Path(workflow['export_path'])/'audit.json')
    approved={row['asset']['id']:row for row in audit}
    rows=read_json(ROOT/'data/minimum_collection/registration.json')['records']
    eligible=[row for row in rows if row['kind']=='image' and row.get('accounting_origin')=='own'
              and approved[row['asset_id']]['review']['boxes']]
    selected=[];seen=set()
    # Keep source diversity; every added image remains a candidate, not a GT.
    for source,preferred in [('kmu','室内'),('firesense','室内'),('kmu','室外'),('firesense','室外'),('flame2','室外')]:
        pool=[r for r in eligible if r['source']==source and r['scene']['setting']==preferred and r['collection_group'] not in seen]
        if not pool:pool=[r for r in eligible if r['source']==source and r['collection_group'] not in seen]
        row=pool[len(pool)//2];seen.add(row['collection_group']);selected.append(row)
    for index,row in enumerate(selected,1):
        source=ROOT/row['source_video'];capture=cv2.VideoCapture(str(source))
        fps=capture.get(cv2.CAP_PROP_FPS);frame=int(round((row['source_seconds']+.37)*fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=capture.read();capture.release()
        if not ok:raise OSError('Cannot decode supplemental candidate')
        image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        image.thumbnail((1280,960))
        target=out/'curated_fire'/f'F{index:02}_{row["source_id"]}.png';target.parent.mkdir(exist_ok=True);image.save(target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if digest in {a['asset']['sha256'] for a in audit}:raise ValueError('Exact existing image')
        records.append({'number':f'F{index:02}','file':target.relative_to(out).as_posix(),'sha256':digest,
                        'source':row['source'],'source_video':row['source_video'],'source_url':row['source_url'],
                        'source_sha256':row['source_sha256'],'collection_group':row['collection_group'],
                        'source_seconds':frame/fps,'frame_number':frame,'accounting_origin':'own',
                        'original_provenance':'public','intended_content':'fire_or_smoke',
                        'scene_declaration_not_new_confirmation':row['scene'],'approved_by_user':False})
    write_json(out/'candidates.json',{'count':len(records),'images':records,'used_for_current_training':False,
                                    'status':'Not counted until user checks and approves; independent final test is unchanged.'})
    sheet=Image.new('RGB',(1000,4*215),'white');draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(records):
        image=Image.open(out/row['file']).convert('RGB');image.thumbnail((240,185))
        x=(index%4)*250;y=(index//4)*215;sheet.paste(image,(x,y));draw.text((x+4,y+188),row['number']+' '+Path(row['file']).stem,fill='black')
    sheet.save(out/'contact_sheet.jpg')
    cards=''.join('<article><h2>'+r['number']+'</h2><a href="'+html.escape(r['file'])+'"><img src="'+html.escape(r['file'])+'"></a><p>'+html.escape(r['intended_content'])+'</p></article>' for r in records)
    (out/'查看候选.html').write_text('<meta charset="utf-8"><title>待补候选</title><style>body{font:18px sans-serif;margin:30px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}img{width:100%;max-height:300px;object-fit:contain}article{border:1px solid #ccc;padding:12px}</style><h1>待补的12幅候选</h1><p>点击图片看原图。这些图片尚未记为人工通过，也未加入本次700幅训练。</p><main>'+cards+'</main>',encoding='utf-8')
    (out/'操作说明.md').write_text('# 补足数量的候选\n\n原700幅无需重做。最终标签重数为507含目标、193正常；其中自主整理500幅来源帧有5幅被确认正常，严格“自行采集500幅火灾图”口径还差5幅。\n\n'
       'N01～N07是D-Fire原空标签正常候选；逐张确认无火烟即可。F01～F05由3套已使用公开视频新增抽帧，先检查含火/烟，再补框或运行预标注后修正。不是本人实拍。\n\n'
       '打开查看候选.html可看原图与来源清单。确认后再由数据导入工具按candidates.json中的origin和collection_group登记，不按单张新文件名拆分原视频组。\n'
       '新增候选尚未导入正式库，不计入完成数；不能用它们悄悄改写当前700幅冻结test或冒充模型正式测试。若未来加入训练，重新冻结新数据版本并保留本次对照。\n',encoding='utf-8')
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    prepare(parser.parse_args().workspace.resolve())
