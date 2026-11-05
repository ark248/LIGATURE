"""Run the fixed comparison or H&E geometry control on a local machine."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def run(name, *args):
    subprocess.run([sys.executable, str(HERE/name), *map(str,args)], check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['fixed','he'],default='fixed')
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    if a.stage=='fixed':
        run('prepare.py')
        run('cheap_readouts.py')
        for fold in range(5):
            for seed in [271027,272027,273027]:
                for kind in ['dscore','dquery','ligature']:
                    run('worker.py','--fold',fold,'--seed',seed,'--kind',kind,'--threads',a.threads)
        run('reuse.py')
        run('evaluate.py')
        run('postprocess.py')
        run('audit_results.py')
    else:
        run('bootstrap_he.py')
        run('he_dataset.py')
        run('he_features.py')
        for fold in range(5):run('he_fit.py','--fold',fold)
        run('evaluate_he.py')
        run('audit_he.py')

if __name__=='__main__':main()
