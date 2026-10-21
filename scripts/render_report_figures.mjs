import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1200,height:1000},deviceScaleFactor:2});
 const out=path.join(root,'artifacts/release/template_work/figures');await fs.mkdir(out,{recursive:true});
 for(const name of ['connections','pipeline']){
  await page.goto(pathToFileURL(path.join(root,'docs/figures',name+'.svg')).href);await page.locator('svg').screenshot({path:path.join(out,name+'.png')});
 }
}finally{await browser.close();}
