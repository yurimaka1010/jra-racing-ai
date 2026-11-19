"""Keep local CSVs in sync with saved results without restarting live recorders."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai.win_sim.core import JST
from jra_racing_ai.d39_next import results_csv as export


def once(day):
    if day not in ('2026-10-24','2026-10-25'):raise ValueError('invalid_date')
    values={}
    for choice in export.CHOICES:
        try:value=export.refresh_saved(day,choice)
        except Exception:value={'status':'pending','reason':'saved_result_csv_unavailable'}
        values[choice]={k:value.get(k) for k in ('status','races','rows','summary_rows')}
    return {'status':'ok','date':day,'exports':values,'new_fits':0,'H_results_included':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--date',default='2026-10-25')
    parser.add_argument('--watch',action='store_true');args=parser.parse_args();previous=None
    root=export.safety.local_path(export.ROOT,root=export.ROOT);root.mkdir(parents=True,exist_ok=True)
    lock=root/'csv-writer.lock'
    try:stream=lock.open('x',encoding='utf-8')
    except FileExistsError:
        print(json.dumps({'status':'pending','reason':'csv_writer_already_running'}));return 1
    try:
        stream.write(str(os.getpid()));stream.flush()
        while True:
            try:value=once(args.date)
            except Exception:value={'status':'pending','reason':'saved_result_csv_unavailable','new_fits':0}
            if value!=previous:print(json.dumps(value),flush=True);previous=value
            if not args.watch:return 0 if value['status']=='ok' else 1
            now=datetime.now(JST)
            if str(now.date())!=args.date or now.hour>=18:return 0
            time.sleep(30)
    finally:stream.close();lock.unlink()


if __name__=='__main__':raise SystemExit(main())
