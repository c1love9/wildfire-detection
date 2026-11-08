"""Training curves whose data lives in the SVG, including fragment reruns."""
import html
import math

LABELS={'box_loss':'框损失','cls_loss':'分类损失','dfl_loss':'DFL损失',
        'precision(B)':'Precision','recall(B)':'Recall','mAP50(B)':'mAP50','mAP50-95(B)':'mAP50–95'}


def history_svg(rows,columns,title):
    if not rows or not columns:return ''
    valid=lambda value:isinstance(value,(int,float)) and math.isfinite(value)
    epochs=[r['epoch'] for r in rows if valid(r.get('epoch'))]
    if not epochs:return ''
    values=[r[c] for r in rows for c in columns if valid(r.get(c))]
    minimum=min(0,min(values,default=0));maximum=max(values,default=1)
    maximum=maximum*1.1 if maximum>0 else 1
    first,last=min(epochs),max(epochs)
    x=lambda value:64+(value-first)/max(1,last-first)*856
    y=lambda value:225-(value-minimum)/max(1e-12,maximum-minimum)*170
    esc=lambda text:html.escape(str(text),quote=True)
    body=[f'<svg viewBox="0 0 960 285" role="img" aria-label="{esc(title)}" style="width:100%;background:#18222d;border-radius:6px">']
    for index in range(5):
        value=minimum+(maximum-minimum)*index/4;position=y(value)
        body.append(f'<path d="M64 {position:.2f} H920" fill="none" stroke="#47515b" stroke-width=".6"/>')
        body.append(f'<text x="8" y="{position+4:.2f}" fill="#ddd" font-size="13">{value:.3g}</text>')
    ticks=sorted({first,last,*[v for v in range(10,int(last)+1,10) if first<=v<=last]})
    if len(ticks)>8:ticks=[ticks[0],*ticks[1:-1:max(1,len(ticks)//6)],ticks[-1]]
    for value in ticks:
        body.append(f'<text x="{x(value):.2f}" y="250" fill="#ddd" text-anchor="middle" font-size="13">{value:g}</text>')
    body.append('<text x="480" y="274" fill="#ddd" text-anchor="middle" font-size="13">Epoch</text>')
    palette=['#76b7e5','#ffb56a','#9bc89b','#cf9ee8']
    for index,column in enumerate(columns):
        color=palette[index%len(palette)];commands=[];connected=False
        label=LABELS.get(column.split('/')[-1],column)
        body.append(f'<text x="{75+index*215}" y="27" fill="{color}" font-size="14">{esc(label)}</text>')
        for row in rows:
            value=row.get(column);epoch=row.get('epoch')
            if not valid(value) or not valid(epoch):connected=False;continue
            commands.append(f'{"L" if connected else "M"}{x(epoch):.2f} {y(value):.2f}');connected=True
            body.append(f'<circle cx="{x(epoch):.2f}" cy="{y(value):.2f}" r="2" fill="{color}"><title>Epoch {epoch:g} {esc(label)} {value:.5g}</title></circle>')
        body.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="1.7"/>')
    return ''.join(body)+'</svg>'
