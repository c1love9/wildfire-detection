// Synthetic teaching visuals only. No formal collection images or access codes are published.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const output=path.join(root,'docs/teammate_assets');
const work=path.join(root,'artifacts/teammate_manual');
await fs.mkdir(output,{recursive:true});await fs.mkdir(work,{recursive:true});
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1150},deviceScaleFactor:1.5});
const text=(x,y,t,size=26,fill='#17212c',anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" text-anchor="${anchor}">${t}</text>`;
const svg=(w,h,body)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei,Arial,sans-serif">${body}</g></svg>`;
async function render(name,w,h,body,publish=true){
 const source=svg(w,h,body);await fs.writeFile(path.join(work,name+'.svg'),source);
 await page.setViewportSize({width:w,height:h});await page.setContent(`<style>body{margin:0}</style>${source}`);await page.evaluate(()=>document.fonts.ready);
 await page.screenshot({path:path.join(publish?output:work,name+'.png')});
}
try{
 let body='';
 const lanes=[['实时检测',['输入画面','YOLO 检测框','SAM 掩码 可选','连续证据与分级','报警 日志 媒体']],['数据与训练',['来源与分组','DINO 候选框','人工修改确认','训练与独立测试','接入自训模型']],['独立质量检查',['相同一批原图','A 与 B 各自画框','提交后锁定','人工仲裁','标注质量报告']]];
 for(let r=0;r<lanes.length;r++){
  const y=30+r*175;body+=text(20,y+26,lanes[r][0],27);
  for(let c=0;c<5;c++){
   const x=22+c*260;body+=`<rect x="${x}" y="${y+48}" width="235" height="80" rx="8" fill="${r===1?'#eef5fb':'#f4f6f8'}" stroke="#b3becb"/>`+text(x+117.5,y+96,lanes[r][1][c],24,'#17212c','middle');
   if(c<4)body+=`<path d="M${x+238} ${y+88}h17m-6 -6 6 6-6 6" fill="none" stroke="#566676" stroke-width="2.5"/>`;
  }
 }
 await render('architecture',1320,535,body);
 body='';
 const flame='M210 84 C218 150 285 166 278 245 C276 310 240 347 197 347 C133 347 104 304 112 252 C116 211 156 185 161 139 C171 165 171 193 179 210 C210 175 211 126 210 84 Z';
 for(let i=0;i<3;i++){
  const x=i*425;body+=`<g transform="translate(${x},0)"><rect x="10" y="40" width="405" height="340" fill="#f4f6f8"/><path d="${flame}" fill="#f39c45"/><path d="M193 202 C210 246 247 261 230 308 C213 340 166 331 158 299 C146 265 179 244 193 202" fill="#ffdf6c"/>`;
  const rect=i===0?[105,80,180,273]:i===1?[154,194,96,140]:[35,52,350,310];
  body+=`<rect x="${rect[0]}" y="${rect[1]}" width="${rect[2]}" height="${rect[3]}" fill="none" stroke="${i===0?'#26734d':'#bf4040'}" stroke-width="5"/>`;
  body+=text(212,420,['包住完整目标','裁掉目标边缘','包含过多背景'][i],27,'#17212c','middle');
  body+=text(212,462,['按可见外缘收紧','会漏掉可见目标','会把背景当作目标'][i],23,'#536171','middle')+'</g>';
 }
 await render('box_rules',1275,490,body);
 body=`<rect x="40" y="62" width="550" height="440" fill="#f4f6f8" stroke="#b8c2cd" stroke-width="2"/>`;
 body+=text(40,38,'原图 1000 × 800',26)+text(593,91,'x',22)+text(19,70,'y',22);
 body+='<rect x="95" y="172" width="165" height="220" fill="#ffdfce" stroke="#db554c" stroke-width="4"/>';
 body+=text(98,155,'(100, 200)',23)+text(265,409,'(400, 600)',23);
 body+='<circle cx="177.5" cy="282" r="5" fill="#17212c"/>';
 body+=text(190,278,'中心 (250, 400)',23);
 const lines=['框宽 300   框高 400','中心 x  250 / 1000 = 0.25','中心 y  400 / 800 = 0.50','框宽    300 / 1000 = 0.30','框高    400 / 800 = 0.50','标签  0 0.25 0.50 0.30 0.50'];
 lines.forEach((s,i)=>body+=text(640,102+i*68,s,i===5?24:26));
 body+=text(40,548,'缩放预览只影响显示  标签始终相对原图',24,'#536171');
 await render('yolo_coordinates',1110,575,body);
 body='<rect x="0" y="0" width="1000" height="650" fill="#eef3f7"/><rect x="0" y="505" width="1000" height="145" fill="#d9dee4"/>';
 body+=text(26,42,'教学示意图  非正式数据',26,'#566676');
 for(const [x,y,rx,ry]of[[470,170,100,70],[545,120,120,68],[630,80,116,54],[485,260,70,90]])body+=`<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="#9099a1" opacity="0.6"/>`;
 body+='<path d="M430 324 C445 386 501 385 490 445 C485 493 442 514 407 507 C370 499 358 465 371 431 C386 391 424 377 430 324Z" fill="#ef9940"/><path d="M411 412 C441 450 455 472 422 494 C390 504 384 469 411 412Z" fill="#ffe276"/>';
 await render('lesson_scene',1000,650,body,false);
 if(process.argv.includes('--capture-canvas')){
  const base=process.env.TEAMMATE_APP_URL||'http://127.0.0.1:8503';
  await page.setViewportSize({width:1440,height:1100});await page.goto(base+'/Annotations');
  await page.getByRole('heading',{name:'自动预标注与人工复核',exact:true}).waitFor();
  await page.getByText('人工复核',{exact:true}).click();
  await page.frameLocator('iframe').first().locator('#canvas').waitFor({timeout:60000});
  await page.frameLocator('iframe').first().getByRole('button',{name:/^1 · fire/}).click();
  await page.addStyleTag({content:'header[data-testid="stHeader"]{visibility:hidden!important} [data-testid="stToolbar"]{visibility:hidden!important}'});
  await page.waitForTimeout(600);
  await page.locator('iframe').first().screenshot({path:path.join(output,'annotation_canvas.png')});
 }
 console.log(JSON.stringify({synthetic_only:true,output,canvas:process.argv.includes('--capture-canvas')}));
}finally{await browser.close();}
