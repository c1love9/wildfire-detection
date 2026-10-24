import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
async function idle(){await page.waitForTimeout(400);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});await page.waitForTimeout(300);assert.equal(await page.getByTestId('stException').count(),0);}
const base=process.env.PRODUCTION_APP_URL||'http://127.0.0.1:8501';
const fixture=process.env.ALARM_APP_URL||'http://127.0.0.1:8502';
try{
 const routes=['','Demo_Images','Demo_Videos','Alarm_History','Dataset','Annotations','Training','Channels','Comparison','Experiments','Quality'];
 for(const route of routes){await page.goto(base+'/'+route);await page.getByTestId('stSidebarNav').waitFor({timeout:120000});await idle();console.log('Verified normal workspace: '+(route||'Home'));}
 await page.goto(base+'/Comparison');await idle();await page.getByText('统一test评测',{exact:true}).click();await idle();
 assert(await page.getByRole('button',{name:'运行统一测试评测',exact:true}).isDisabled());
 await page.getByText(/待评测：请先选择本人成功训练的模型/).waitFor();
 await page.goto(base+'/Experiments');await idle();await page.getByText('待实验：尚无已复核训练数据。',{exact:true}).waitFor();
 await page.goto(fixture+'/Alarm_History');await page.getByRole('heading',{name:'报警日志',exact:true}).waitFor();await idle();
 await page.getByTestId('stMetric').filter({has:page.getByText('火烟占比变化率',{exact:true})}).first().waitFor();
 await page.locator('svg[aria-label="画面占比变化率趋势"]').first().waitFor();
 const video=page.locator('video').first();await video.waitFor();
 await video.evaluate(async e=>{e.muted=true;await e.play();});await page.waitForTimeout(1500);
 const playback=await video.evaluate(e=>({duration:e.duration,played:e.currentTime,error:e.error?.message||null}));
 assert(playback.duration>0&&playback.played>0&&!playback.error);
 const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}).click()]);
 const location=await download.path();assert((await fs.stat(location)).size>100);
 assert((await page.getByTestId('stImage').locator('img').first().evaluate(e=>e.naturalWidth))>0);
 assert((await page.locator('canvas').count())>0);
 await page.goto(fixture+'/Comparison');await page.getByRole('heading',{name:'YOLO与开放域模型对比',exact:true}).waitFor();await idle();await page.getByRole('button',{name:'下载完整对比记录',exact:true}).waitFor();
 await page.goto(fixture+'/Experiments');await page.getByRole('heading',{name:'优化对照实验',exact:true}).waitFor();await idle();await page.getByRole('button',{name:'下载成对实验结果',exact:true}).waitFor();
 assert.deepEqual(errors,[]);
 const result={status:'passed',verified_at:new Date().toISOString(),normal_workspace_pages:routes,formal_actions_pending_without_real_data:true,
  restarted_fixture_history_playback:playback,trend_csv_download:true,coverage_rate_metric_and_chart:true,history_map_and_snapshot:true,comparison_and_pair_history_after_restart:true,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/delivery_browser.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
}catch(error){await page.screenshot({path:path.join(root,'artifacts/release/screenshots/final_browser_failure.png')});throw error;}finally{await browser.close();}
