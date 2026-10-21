"""Package only the explicit public delivery allowlist, never local data/models."""
from pathlib import Path
import hashlib
import json
import subprocess
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1];RELEASE=ROOT/'artifacts/release'


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','-z','--name-only','HEAD'],cwd=ROOT).decode('utf-8').split('\0')
    code=[name for name in names if name and (Path(name).suffix in ('.py','.mjs','.js','.html','.ps1','.cmd','.toml','.svg') or
          name in ('LICENSE','.gitignore','requirements.txt','requirements-lock.txt') or
          name.startswith(('docs/','configs/','reports/')) and Path(name).suffix in ('.md','.json'))]
    source=RELEASE/'source_code.zip'
    subprocess.run(['git','archive','--format=zip','-o',str(source),'HEAD','--',*code],cwd=ROOT,check=True)
    with ZipFile(source) as archive:
        assert archive.testzip() is None
        assert all(not name.startswith(('artifacts/','data/','.venv/','fire-models/','demo-images/','demo-videos/')) for name in archive.namelist())
    files={p.name:p for p in [RELEASE/'智能火灾报警系统设计报告.docx',RELEASE/'智能火灾报警系统演示.mp4']}
    files['source_code.zip']=source
    for name in ('quick_start','completion','architecture','delivery','template_report','report_reference_check','report_template_contract','requirements_audit','environment','upstream'):
        files['docs/'+name+'.md']=ROOT/'docs'/(name+'.md')
    files['requirements-lock.txt']=ROOT/'requirements-lock.txt'
    for name in ('delivery_checks','delivery_demo','delivery_browser','template_report'):
        files['reports/'+name+'.json']=ROOT/'reports'/(name+'.json')
    manifest={'status':'System features, experiment tools and material framework completed; formal experiments pending',
              'repository':'https://github.com/c1love9/wildfire-detection','source_commit':commit,
              'content':{name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()},
              'excluded':'Models, datasets, physical camera images, annotation access codes, environments, temporary renders and synthetic fixtures'}
    manifest_path=RELEASE/'manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target=RELEASE/'智能火灾报警系统交付包.zip'
    with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in files.items():archive.write(path,name)
        archive.write(manifest_path,'manifest.json')
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,record in manifest['content'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
    print(json.dumps({'package':str(target),'bytes':target.stat().st_size,'files':len(files)+1,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=='__main__':main()
