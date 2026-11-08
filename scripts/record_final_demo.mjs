// Actual browser recording against the completed real-data workspace.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const root=fileURLToPath(new URL('..',import.meta.url));
const workspace=path.resolve(process.env.FINAL_WORKSPACE||path.join(root,'artifacts/final_completion_20261108'));
const formal=JSON.parse(await fs.readFile(path.join(workspace,'formal_experiments.json'),'utf8'));
const output=path.join(workspace,'demo');await fs.mkdir(output,{recursive:true});
await fs.mkdir(path.join(output,'screenshots'),{recursive:true});
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({headless:true,channel:'msedge',args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1000},recordVideo:{dir:path.join(output,'raw'),size:{width:1440,height:1000}}});
const page=await context.newPage(),errors=[],chapters=[];let playback=null,pausedFrames=null;
page.on('pageerror',error=>errors.push(error.message));
const base=process.env.PRODUCTION_APP_URL||'http://127.0.0.1:8506';
const started=Date.now();
async function idle(){await page.waitForTimeout(500);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});assert.equal(await page.getByTestId('stException').count(),0);}
async function caption(value){chapters.push({seconds:(Date.now()-started)/1000,text:value});console.log(value);await page.evaluate(value=>{let e=document.getElementById('final-caption');if(!e){e=document.createElement('div');e.id='final-caption';e.style.cssText='position:fixed;bottom:10px;left:20px;right:20px;z-index:2147483647;padding:13px 18px;background:#152d46;color:white;font:20px Microsoft YaHei,sans-serif;pointer-events:none;';document.body.appendChild(e);}e.textContent=value;},value);}
async function dwell(ms){const until=Date.now()+ms;while(Date.now()<until){const ack=page.getByRole('button',{name:'确认并关闭',exact:true});if(await ack.isVisible())await ack.click();await page.waitForTimeout(Math.min(500,Math.max(1,until-Date.now())));}}
async function focus(locator){await locator.evaluate(e=>e.scrollIntoView({block:'center'}));await page.waitForTimeout(400);}
async function expand(label){const e=page.getByTestId('stExpander').filter({has:page.getByText(label,{exact:true})});if(!await e.locator('details').evaluate(e=>e.open))await e.locator('summary').click();}
async function choose(label,text){const box=page.getByTestId('stSelectbox').filter({has:page.getByText(label,{exact:true})}).getByRole('combobox');await box.click();await page.getByRole('option').filter({hasText:text}).click();await idle();}
async function toggle(label){await page.getByTestId('stCheckbox').filter({has:page.getByText(label,{exact:true})}).locator('label').click();await idle();}
async function screen(name){await page.screenshot({path:path.join(output,'screenshots',name+'.png')});}
try{
 await page.goto(base);await page.getByText('Select Model Type',{exact:true}).waitFor({timeout:120000});await idle();
 await choose('Select Model Size',formal.selection.training_id.slice(-8));
 await choose('Select demo image','image12.jpeg');
 await caption('01 选择本人训练模型 · 700幅已复核图像，两组各50轮，主模型由验证集选择');await dwell(7000);
 await toggle('开启火烟像素分割');await page.getByText('火焰画面占比',{exact:true}).waitFor({timeout:120000});await idle();
 await focus(page.getByTestId('stImage').last());await caption('02 当前帧检测框提示 SAM · 彩色掩码与面积占比，公开演示素材不计入正式精度');await screen('image_masks');await dwell(10000);
 await page.goto(base+'/Demo_Videos');await page.getByRole('heading',{name:'视频火焰与烟雾检测',exact:true}).waitFor();await idle();
 await page.getByText('命名通道',{exact:true}).click();await idle();await choose('检测模型',formal.selection.training_id.slice(-8));
 await expand('检测设置');await toggle('开启火烟像素分割');
 await caption('03 通道 A 实时逐帧处理 · 本人模型、SAM分割、连续确认与分级报警');
 await page.getByRole('button',{name:'开始检测',exact:true}).click();await dwell(22000);
 assert(await page.getByTestId('stSelectbox').filter({has:page.getByText('检测模型',{exact:true})}).getByRole('combobox').isDisabled());
 await page.getByRole('button',{name:'暂停检测',exact:true}).click();await idle();
 const frameMetric=page.getByTestId('stMetric').filter({has:page.getByText('当前帧',{exact:true})});pausedFrames=await frameMetric.innerText();
 await focus(page.getByTestId('stImage').last());await screen('video_masks');await dwell(5000);
 assert.equal(await frameMetric.innerText(),pausedFrames,'Pause must freeze the frame counter');
 await caption('04 暂停后不新增采样 · 原图像素并集、面积变化和等级历史可导出');
 await focus(page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}));await screen('area_trends');await dwell(10000);
 const [trendDownload]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}).click()]);
 assert((await fs.stat(await trendDownload.path())).size>100);
 await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();
 await choose('监控通道','监控通道 B');await caption('05 停止后切换通道 B · 位置和视角同步，旧通道确认状态不继承');await dwell(6000);
 await page.getByRole('button',{name:'开始检测',exact:true}).click();await dwell(9000);
 await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();
 await page.goto(base+'/Alarm_History');await page.getByRole('heading',{name:'报警日志',exact:true}).waitFor();await idle();
 await caption('06 报警历史 · 回溯模型与模拟位置，查看确认截图、实际前后片段和趋势');
 const video=page.locator('video').first();
 await video.waitFor({timeout:45000});await focus(video);await video.evaluate(async e=>{e.muted=true;await e.play();});await dwell(9000);
 playback=await video.evaluate(e=>({duration:e.duration,currentTime:e.currentTime,error:e.error?.message||null}));
 assert(playback.duration>0&&playback.currentTime>0&&!playback.error);
 await screen('history_media');
 await page.goto(base+'/Training');await page.getByRole('heading',{name:'YOLO模型训练与评估',exact:true}).waitFor();await idle();
 await choose('查看任务',formal.selection.training_id);
 await caption('07 真实50轮训练 · 环境、超参数、逐轮损失和验证mAP均可下载');
 await focus(page.getByTestId('stImage').first());await screen('training_curves');await dwell(10000);
 await page.goto(base+'/Comparison');await page.getByRole('heading',{name:'YOLO与开放域模型对比',exact:true}).waitFor();await idle();
 await choose('YOLO对比模型',formal.selection.training_id.slice(-8));
 await page.getByText('统一test评测',{exact:true}).click();await idle();
 await caption('08 YOLO 与未微调 GroundingDINO · 同一冻结test、同尺寸设备、统一固定阈值');
 await focus(page.getByRole('button',{name:'下载对比CSV',exact:true}));await screen('model_comparison');await dwell(14000);
 await page.goto(base+'/Experiments');await page.getByRole('heading',{name:'优化对照实验',exact:true}).waitFor();await idle();
 await caption('09 固定尺寸与多尺度 · 仅改变一个因素，保留实际改善或退化，不按test再选模');
 await focus(page.getByTestId('stImage').first());await screen('optimization');await dwell(10000);
 await focus(page.getByRole('button',{name:'下载成对实验结果',exact:true}));await dwell(8000);
 await page.goto(base+'/Annotations');await page.getByRole('heading',{name:'自动预标注与人工复核',exact:true}).waitFor();await idle();
 await caption('10 700幅人工复核均已保存 · 原Tiny、增强候选、A/B独立标注和最终标签分别保留');await screen('review_completed');await dwell(10000);
 await page.goto(base+'/Quality');await page.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();await idle();
 await page.getByText('面积误差',{exact:true}).click();await caption('11 面积分割误差待独立参考 · 工时按负责人实测约5秒每幅估算，非全量计时');await dwell(8000);
 await page.getByText('工时对照',{exact:true}).click();await dwell(7000);
 await caption('交付含真实数据、本人权重、训练评测、模板报告和复现说明；还需确认补足5幅自主火烟帧及7幅正常图');
 await dwell(Math.max(7000,195000-(Date.now()-started)));
 assert.deepEqual(errors,[]);const duration=(Date.now()-started)/1000;assert(duration>=180&&duration<300,`Duration ${duration}`);
 await context.close();const raw=await page.video().path();
 const checkContext=await browser.newContext({viewport:{width:1440,height:1000}});const checkPage=await checkContext.newPage();
 checkPage.on('pageerror',error=>errors.push(error.message));
 for(const route of ['Demo_Images','Dataset','Channels']){await checkPage.goto(base+'/'+route);await checkPage.getByTestId('stSidebarNav').waitFor({timeout:60000});await checkPage.waitForTimeout(500);await checkPage.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:60000});assert.equal(await checkPage.getByTestId('stException').count(),0);}
 await checkContext.close();
 const target=path.join(output,'系统演示.mp4');
 const rawProbe=spawnSync('ffprobe',['-v','error','-show_entries','format=duration','-of','json',raw],{encoding:'utf8',windowsHide:true});
 assert.equal(rawProbe.status,0,rawProbe.stderr);
 const lead=Math.max(0,Number(JSON.parse(rawProbe.stdout).format.duration)-duration+chapters[0].seconds);
 const conversion=spawnSync('ffmpeg',['-y','-ss',String(lead),'-i',raw,'-vf','fps=15,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22','-an','-movflags','+faststart',target],{encoding:'utf8',windowsHide:true});
 assert.equal(conversion.status,0,conversion.stderr);
 const encodedProbe=spawnSync('ffprobe',['-v','error','-show_entries','format=duration','-of','json',target],{encoding:'utf8',windowsHide:true});
 assert.equal(encodedProbe.status,0,encodedProbe.stderr);const encodedDuration=Number(JSON.parse(encodedProbe.stdout).format.duration);
 assert(encodedDuration>=180&&encodedDuration<=300);
 const firstCaption=chapters[0].seconds;
 const report={status:'passed',recorded_at:new Date().toISOString(),duration_seconds:encodedDuration,wall_clock_seconds:duration,removed_loading_lead_seconds:lead,
   chapters:chapters.map(c=>({...c,seconds:Math.max(0,c.seconds-firstCaption)})),page_errors:errors,
   source:'Actual production browser; real trained models and frozen formal experiment history. Public demo media; no personal camera recording.',
   selected_training_id:formal.selection.training_id,video:path.relative(root,target)};
 await fs.writeFile(path.join(root,'reports/final_demo_20261108.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 const browserReport={status:'passed',verified_at:new Date().toISOString(),routes:['Home','Demo_Images','Demo_Videos','Alarm_History','Dataset','Annotations','Training','Channels','Comparison','Experiments','Quality'],
   own_model_selected:formal.selection.training_id,own_model_video_controls:true,model_locked_while_running:true,pause_frame_counter_frozen:pausedFrames,trend_csv_downloaded:true,
   history_playback:playback,formal_comparison_and_pair_results_visible:true,page_errors:errors,scope:'Actual production browser, public demo video. This verifies UI operations and persistence visibility, not formal detection accuracy.'};
 await fs.writeFile(path.join(root,'reports/final_browser_20261108.json'),JSON.stringify(browserReport,null,2)+'\n');
}catch(error){if(!page.isClosed())await screen('recording_failure');throw error;}finally{await browser.close();}
