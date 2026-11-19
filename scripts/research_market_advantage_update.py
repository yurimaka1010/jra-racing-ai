"""Independent amendment-2 commands. No fitting or production integration."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai import market_advantage_update as research

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','final-odds','analyze'))
    parser.add_argument('--run-id',required=True);args=parser.parse_args()
    try:
        value={'prepare':research.prepare,'final-odds':research.collect_final,'analyze':research.analyze}[args.stage](args.run_id)
        print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0
    except Exception as error:
        reason=str(error) if isinstance(error,ValueError) and str(error).replace('_','').isalnum() else 'local_analysis_error'
        print(json.dumps(dict(status='stopped',stage=args.stage,reason=reason)));return 1

if __name__=='__main__':raise SystemExit(main())
