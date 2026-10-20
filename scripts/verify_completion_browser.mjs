import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('..', import.meta.url));
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser = await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const page = await browser.newPage({viewport:{width:1440,height:1200}});
const errors=[]; page.on('pageerror',e=>errors.push(e.message));
const stage=process.argv[2] || 'segmentation';
let notices=0;
const output=path.join(root,'artifacts/completion_checks/browser'); await fs.mkdir(output,{recursive:true});
const base=process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
async function idle(){await page.waitForTimeout(400);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});await page.waitForTimeout(400);assert.equal(await page.getByTestId('stException').count(),0);}
async function toggle(label){await page.getByTestId('stCheckbox').filter({has:page.getByText(label,{exact:true})}).locator('label').click();await idle();}
async function expand(label){const exp=page.getByTestId('stExpander').filter({has:page.getByText(label,{exact:true})});if(!await exp.locator('details').evaluate(e=>e.open))await exp.locator('summary').click();}
try{
 await page.goto(base);await page.getByText('Select Model Type',{exact:true}).waitFor({timeout:120000});await idle();
 await toggle('开启火烟像素分割');
 await page.getByText('火焰画面占比',{exact:true}).waitFor({timeout:120000});await idle();
 assert.equal(await page.getByRole('button',{name:/下载.*占比掩码/}).count(),3);
 await page.getByTestId('stImage').locator('img').last().evaluate(e=>e.decode());
 await page.screenshot({path:path.join(output,'segmentation_image.png'),fullPage:true});
 await page.goto(base+'/Demo_Videos');await page.getByRole('heading',{name:'视频火焰与烟雾检测',exact:true}).waitFor();await idle();
 if(stage==='channels'){await page.getByText('命名通道',{exact:true}).click();await idle();await page.getByText(/当前关联：演示监控点 A/).waitFor();}
 await expand('检测设置');await toggle('开启火烟像素分割');
 await page.getByRole('button',{name:'开始检测',exact:true}).click();
 await page.getByText('火焰画面占比',{exact:true}).waitFor({timeout:120000});
 if(stage!=='segmentation') {
   await page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}).waitFor();
   const deadline=Date.now()+120000;
   while(Date.now()<deadline){
     const ack=page.getByRole('button',{name:'确认并关闭',exact:true});
     if(await ack.isVisible()){await ack.click();notices++;await page.waitForTimeout(200);}
     const frame=await page.evaluate(()=>{const metric=[...document.querySelectorAll('[data-testid="stMetric"]')].find(e=>e.textContent.includes('当前帧'));return Number(metric?.querySelector('[data-testid="stMetricValue"]')?.textContent.split('/')[0]);});
     if(frame>=(stage==='alerts'?80:25))break;
     await page.waitForTimeout(200);
   }
   const ack=page.getByRole('button',{name:'确认并关闭',exact:true});if(await ack.isVisible()){await ack.click();notices++;await page.waitForTimeout(200);}
   if(stage==='alerts')assert(notices>0,'No alarm dialog was observed');
 }
 await page.getByRole('button',{name:'暂停检测',exact:true}).click();await idle();
 assert(await page.getByRole('checkbox',{name:'开启火烟像素分割',exact:true}).isDisabled());
 await page.screenshot({path:path.join(output,stage+'_video.png'),fullPage:true});
 await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();
 if(stage==='channels'){
   const control=page.getByTestId('stSelectbox').filter({has:page.getByText('监控通道',{exact:true})}).getByRole('combobox');
   await control.click();await page.getByRole('option').filter({hasText:'监控通道 B'}).click();await idle();
   await page.getByText(/当前关联：演示监控点 B/).waitFor();
   await page.getByRole('button',{name:'开始检测',exact:true}).click();await page.getByText(/监控通道 B · 正在检测/).waitFor();
   await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();
 }
 if(stage!=='segmentation') {
   await page.goto(base+'/Alarm_History');await page.getByRole('heading',{name:'报警日志',exact:true}).waitFor();await idle();
   await page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}).waitFor();
 }
 assert.deepEqual(errors,[]);
 const report={status:'passed',stage,verified_at:new Date().toISOString(),image_masks_and_download_controls:true,video_segmented_pause_stop:true,alarm_dialogs_acknowledged:notices,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/'+stage+'_browser.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
}catch(error){await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});throw error;}finally{await browser.close();}
