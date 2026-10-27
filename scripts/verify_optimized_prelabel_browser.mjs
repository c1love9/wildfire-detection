// Public/synthetic fixture only; no automated labels enter the formal catalog.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const root=fileURLToPath(new URL('..',import.meta.url));
const out=path.join(root,'artifacts/prelabel_optimization_20261027');
const fixture=JSON.parse(await fs.readFile(path.join(out,'browser_fixture.json'),'utf8'));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1200}});const errors=[];
page.on('pageerror',e=>errors.push(e.message));
async function idle(){await page.waitForTimeout(450);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:180000});assert.equal(await page.getByTestId('stException').count(),0);}
const button=name=>page.getByRole('button',{name,exact:true});
try{
 await page.goto((process.env.ALARM_APP_URL||'http://127.0.0.1:8507')+'/Annotations');
 await page.getByRole('heading',{name:'自动预标注与人工复核',exact:true}).waitFor({timeout:90000});await idle();
 const actor=page.getByRole('textbox',{name:'执行者/复核者姓名或编号',exact:true});await actor.fill('浏览器自动化功能验证，非人工标注');await actor.press('Tab');await idle();
 assert.equal(await page.getByRole('textbox',{name:'火焰提示词',exact:true}).inputValue(),'fire.');
 assert.equal(await page.getByRole('textbox',{name:'烟雾提示词',exact:true}).inputValue(),'smoke.');
 await page.getByText('模型与推理参数',{exact:true}).click();await idle();
 assert.equal(Number(await page.getByRole('spinbutton',{name:'火焰候选阈值',exact:true}).inputValue()),.3);
 assert.equal(Number(await page.getByRole('spinbutton',{name:'烟雾候选阈值',exact:true}).inputValue()),.35);
 await button('开始自动预标注').click();
 await page.getByText('预标注任务结束，请进入人工复核；候选框未自动批准。',{exact:true}).waitFor({timeout:180000});await idle();
 await page.getByText('人工复核',{exact:true}).click();await idle();
 const selector=page.getByTestId('stSelectbox').filter({has:page.getByText('复核图片',{exact:true})}).getByRole('combobox');
 await selector.click();await selector.fill(fixture.asset_ids[0].slice(0,12));await page.getByRole('option').filter({hasText:fixture.asset_ids[0].slice(0,12)}).click();await idle();
 await button('采用所选预标注').click();await idle();
 const frame=page.frameLocator('iframe[title*="box_editor"]');await frame.locator('#count').waitFor();
 const count=Number((await frame.locator('#count').innerText()).split(' ')[0]);assert(count>0);
 await page.getByText('查看该次模型与提示词记录',{exact:true}).click();await idle();
 await page.getByText('人工复核',{exact:true}).waitFor();
 await page.screenshot({path:path.join(out,'enhanced_browser.png'),fullPage:true});
 await page.getByText('自动预标注',{exact:true}).click();await idle();
 await page.getByText('原方案：Tiny独立查询',{exact:true}).click();await idle();
 assert.equal(await page.getByRole('textbox',{name:'火焰提示词',exact:true}).inputValue(),'fire. flame. burning. blaze.');
 const python=`import json,sqlite3,sys
from pathlib import Path
p=Path(sys.argv[1])/'catalog.sqlite3';db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
rows=list(db.execute('SELECT p.status,p.boxes_json,r.metadata_json FROM annotation_proposals p JOIN annotation_runs r ON r.id=p.run_id'));assert len(rows)==3
for row in rows:
 m=json.loads(row['metadata_json']);assert row['status']=='ready';assert m['runtime']['model_id']=='IDEA-Research/grounding-dino-base';assert m['auxiliary_model']['classes']=={'0':'smoke','1':'fire'}
 for b in json.loads(row['boxes_json']):assert b['prelabel_source'] in ('grounding_dino_base_joint','upstream_yolo_fire_n')
assert db.execute('SELECT COUNT(*) FROM annotation_reviews').fetchone()[0]==0
print(json.dumps({'ready_proposals':3,'no_human_approvals':True,'actual_model_and_class_mapping_verified':True}))`;
 const check=spawnSync(path.join(root,'.venv/Scripts/python.exe'),['-c',python,fixture.root],{encoding:'utf8'});assert.equal(check.status,0,check.stderr);assert.deepEqual(errors,[]);
 const report={status:'passed',verified_at:new Date().toISOString(),scope:'Isolated public/synthetic fixtures, no formal annotation writes',enhanced_profile_controls:true,real_base_and_auxiliary_inference:true,review_canvas_boxes:count,legacy_profile_switch:true,checks:JSON.parse(check.stdout),page_errors:errors};
 await fs.writeFile(path.join(root,'reports/optimized_prelabel_browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await browser.close();}
