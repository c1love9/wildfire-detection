import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=fileURLToPath(new URL('..',import.meta.url));
const output=path.join(root,'artifacts/release');
await fs.mkdir(path.join(output,'screenshots'),{recursive:true});
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({headless:true,channel:'msedge',args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1000},recordVideo:{dir:path.join(output,'video_raw'),size:{width:1440,height:1000}}});
const page=await context.newPage();
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.ALARM_APP_URL||'http://127.0.0.1:8502';
const started=Date.now();const chapters=[];
async function idle(){await page.waitForTimeout(500);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});assert.equal(await page.getByTestId('stException').count(),0);}
async function caption(text){chapters.push({seconds:Math.round((Date.now()-started)/1000),text});console.log(text);await page.evaluate(text=>{let e=document.getElementById('delivery-caption');if(!e){e=document.createElement('div');e.id='delivery-caption';e.style.cssText='position:fixed;bottom:8px;left:260px;right:20px;z-index:2147483647;pointer-events:none;padding:12px 18px;background:#14293e;color:white;font:18px Microsoft YaHei,sans-serif;border-radius:6px;box-shadow:0 2px 8px #0005;';document.body.appendChild(e);}e.textContent=text;},text);}
async function dwell(ms){const until=Date.now()+ms;while(Date.now()<until){const ack=page.getByRole('button',{name:'确认并关闭',exact:true});if(await ack.isVisible())await ack.click();await page.waitForTimeout(Math.min(500,Math.max(1,until-Date.now())));}}
async function expand(label){const exp=page.getByTestId('stExpander').filter({has:page.getByText(label,{exact:true})});if(!await exp.locator('details').evaluate(e=>e.open))await exp.locator('summary').click();}
async function toggle(label){await page.getByTestId('stCheckbox').filter({has:page.getByText(label,{exact:true})}).locator('label').click();await idle();}
async function focus(locator){await locator.evaluate(e=>e.scrollIntoView({block:'center'}));await page.waitForTimeout(600);}
try{
 await page.goto(base);await page.getByText('Select Model Type',{exact:true}).waitFor({timeout:120000});await idle();
 await caption('01 图片检测与 SAM 分割 · 使用已有公开演示素材，以下不是正式精度结果');await dwell(8000);
 await toggle('开启火烟像素分割');await page.getByText('火焰画面占比',{exact:true}).waitFor({timeout:120000});await idle();
 const image=page.getByTestId('stImage').last();await focus(image);await image.locator('img').evaluate(e=>e.decode());
 await image.screenshot({path:path.join(output,'screenshots/image_masks.png')});await dwell(10000);
 await focus(page.getByRole('button',{name:'下载火焰画面占比掩码',exact:true}));await dwell(7000);

 await page.goto(base+'/Demo_Videos');await page.getByRole('heading',{name:'视频火焰与烟雾检测',exact:true}).waitFor();await idle();
 await page.getByText('命名通道',{exact:true}).click();await idle();
 await caption('02 命名通道 A · 模拟位置与视角随通道绑定，停止后才可切换');await dwell(7000);
 await expand('检测设置');await toggle('开启火烟像素分割');
 await page.getByRole('button',{name:'开始检测',exact:true}).click();
 await page.getByText('火焰画面占比',{exact:true}).waitFor({timeout:120000});
 await caption('03 逐帧分割与连续确认 · 面积辅助升级，弹窗确认，声音默认关闭');
 await focus(page.getByTestId('stImage').last());await dwell(18000);
 await page.getByRole('button',{name:'暂停检测',exact:true}).click();await idle();
 await page.screenshot({path:path.join(output,'screenshots/video_masks.png')});
 await caption('04 暂停后不新增采样 · 火焰、烟雾、并集占比及报警等级趋势可导出');
 await focus(page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}));await dwell(14000);
 await page.screenshot({path:path.join(output,'screenshots/area_trends.png')});
 await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();
 const channel=page.getByTestId('stSelectbox').filter({has:page.getByText('监控通道',{exact:true})}).getByRole('combobox');
 await channel.click();await page.getByRole('option').filter({hasText:'监控通道 B'}).click();await idle();
 await caption('05 停止后切换通道 B · 独立确认状态，历史保留原通道和位置快照');
 await page.getByRole('button',{name:'开始检测',exact:true}).click();await dwell(8000);
 await page.getByRole('button',{name:'停止检测',exact:true}).click();await idle();await dwell(4000);

 await page.goto(base+'/Alarm_History');await page.getByRole('heading',{name:'报警日志',exact:true}).waitFor();await idle();
 await caption('06 历史日志 · 按通道筛选，回溯事件发生时的位置与模型');await dwell(8000);
 await focus(page.getByText('报警截图与片段',{exact:true}));await dwell(5000);
 const video=page.locator('video').first();await video.waitFor();await video.evaluate(async e=>{e.muted=true;await e.play();});await dwell(10000);
 await page.screenshot({path:path.join(output,'screenshots/history_media.png')});
 await focus(page.getByRole('button',{name:'导出面积趋势 CSV',exact:true}));await dwell(6000);

 await page.goto(base+'/Comparison');await page.getByRole('heading',{name:'YOLO与开放域模型对比',exact:true}).waitFor();await idle();
 await caption('07 YOLO 与 GroundingDINO · 同尺寸同设备并排推理，提示词和协议可追溯');
 await page.getByRole('button',{name:'运行并排检测',exact:true}).click();
 await page.getByText('并排推理完成；没有独立参考标注，不计算准确率。',{exact:true}).waitFor({timeout:120000});
 await focus(page.getByTestId('stImage').first());await dwell(14000);
 await page.getByTestId('stHorizontalBlock').filter({has:page.getByTestId('stImage')}).last().screenshot({path:path.join(output,'screenshots/model_comparison.png')});
 await page.getByText('统一test评测',{exact:true}).click();await idle();
 await caption('08 冻结测试集与统一评测 · 合成流程记录仅用于工具验证，正式结果待补');await dwell(10000);

 await page.goto(base+'/Experiments');await page.getByRole('heading',{name:'优化对照实验',exact:true}).waitFor();await idle();
 await caption('09 固定尺寸与多尺度训练 · 同初始权重、同数据、同种子，仅改变声明因素');await dwell(8000);
 const curve=page.getByTestId('stImage').first();if(await curve.count()){await focus(curve);await dwell(10000);}
 await focus(page.getByRole('button',{name:'下载成对实验结果',exact:true}));await dwell(7000);
 await page.goto(base+'/Quality');await page.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();await idle();
 await caption('10 独立标注与质量工具 · 分开访问、空白画布、提交锁定，真人独立性仍需实际执行');await dwell(9000);
 await page.getByText('面积误差',{exact:true}).click();await dwell(7000);
 await caption('11 人工参考掩码与工时对照 · 实际误差、200幅双人标注和工时节省仍待正式数据');
 await page.getByText('工时对照',{exact:true}).click();await dwell(9000);
 await caption('交付状态：系统功能、实验工具和材料框架完成 · 正式数据与实验结论待补');
 await dwell(5000);
 const remaining=195000-(Date.now()-started);if(remaining>0)await dwell(remaining);
 assert.deepEqual(errors,[]);
 const duration=(Date.now()-started)/1000;assert(duration>=180&&duration<300,`Unexpected duration ${duration}`);
 await context.close();const raw=await page.video().path();
 const report={status:'passed',scope:'Actual browser recording on existing public demos and isolated synthetic records; no formal accuracy claim',duration_seconds:duration,raw_video:path.relative(root,raw),chapters,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/delivery_demo.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}catch(error){await page.screenshot({path:path.join(output,'screenshots/recording_failure.png')});throw error;}finally{await browser.close();}
