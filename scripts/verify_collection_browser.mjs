import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.COLLECTION_APP_URL||'http://127.0.0.1:8501';
async function idle(){await page.waitForTimeout(350);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});await page.waitForTimeout(350);assert.equal(await page.getByTestId('stException').count(),0);}
try{
 await page.goto(base+'/Dataset');await page.getByRole('heading',{name:'数据采集与管理',exact:true}).waitFor();await idle();
 await page.getByRole('tab',{name:'统计与划分',exact:true}).click();
 for(const [label,value] of [['去重图片','700'],['去重视频','300'],['视频抽帧图片','500']]){
  const metric=page.getByTestId('stMetric').filter({has:page.getByText(label,{exact:true})});assert.equal((await metric.getByTestId('stMetricValue').innerText()).trim(),value);
 }
 const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'导出来源清单 CSV',exact:true}).click()]);
 const csv=await fs.readFile(await download.path(),'utf8');assert(csv.split('\n').filter(Boolean).length===1001);
 await page.goto(base+'/Annotations');await page.getByRole('heading',{name:'自动预标注与人工复核',exact:true}).waitFor();await idle();
 assert.equal((await page.getByTestId('stMetric').filter({has:page.getByText('人工通过',{exact:true})}).getByTestId('stMetricValue').innerText()).trim(),'0');
 await page.getByText('人工复核',{exact:true}).click();await idle();
 await page.getByText(/当前画布/).waitFor();
 await page.frameLocator('iframe').first().locator('#canvas').waitFor({timeout:60000});
 assert(await page.getByRole('button',{name:'确认复核通过',exact:true}).isDisabled());
 await page.goto(base+'/Training');await page.getByRole('heading',{name:'YOLO模型训练与评估',exact:true}).waitFor();await idle();
 assert(await page.getByRole('button',{name:'开始训练',exact:true}).isDisabled());assert.deepEqual(errors,[]);
 const result={status:'passed',verified_at:new Date().toISOString(),dataset_images:700,dataset_videos:300,external_video_frames:500,downloaded_manifest_rows:1000,
  candidates_visible_in_review_workspace:true,human_approval_not_fabricated:true,training_gated_until_reviewed_export:true,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/minimum_collection_browser.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
}finally{await browser.close();}
