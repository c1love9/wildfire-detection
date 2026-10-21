"""Use the skill renderer; opt into native Word export when LibreOffice is absent."""
from pathlib import Path
import argparse
import importlib.util
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--renderer',required=True,type=Path);parser.add_argument('--word',action='store_true');args,extra=parser.parse_known_args()
    spec=importlib.util.spec_from_file_location('document_skill_renderer',args.renderer)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    if args.word:
        def convert(doc_path,user_profile,convert_tmp_dir,stem,verbose):
            target=Path(convert_tmp_dir)/(stem+'.pdf')
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/export_word_pdf.ps1'),'-InputPath',str(doc_path),'-OutputPath',str(target)],capture_output=True,text=True,timeout=120)
            log=result.stdout+result.stderr
            if verbose:print(log)
            if result.returncode or not target.is_file():raise RuntimeError('Native Word export failed: '+log)
            return str(target),log
        renderer.convert_to_pdf=convert
    sys.argv=[str(args.renderer),*extra];renderer.main()


if __name__=='__main__':main()
