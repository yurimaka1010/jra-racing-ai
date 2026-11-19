"""Reuse saved LightGBM importance and the existing Windows chart renderer."""
from pathlib import Path
import sys,csv,json,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai.ml import safety
from jra_racing_ai.d39_next import contracts as original,learning as initial_model,importance
from jra_racing_ai.d39_enrichment import contracts as improved,learning as improved_model

ROOT=original.BASE/'model-diagnostics'
SOURCES=(('next_initial','次世代D39 初期版（47項目）',original.folder('b803140f84ac4ee39d9decc408e85f2a'),initial_model),
         ('next_improved','次世代D39 改善版（前走タイム接続・47項目）',improved.folder('f581701e5453474988425affca1b8faa'),improved_model))


def write_csv(path,rows):
    with safety.local_path(path,root=ROOT).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    protected={};models=[]
    directory=safety.new_artifact('evaluations',root=ROOT)
    for key,title,source,loader in SOURCES:
        freeze=original.read(source/'candidate_freeze.json');path=Path(freeze['model_path'])
        if original.sha(path/'model.json')!=freeze['model_sha256'] or original.sha(path/'booster.lgb.txt')!=freeze['tree_sha256']:
            raise safety.MLFailure('artifact_invalid')
        for p in [source/n for n in ('candidate_freeze.json','plan.json','data_manifest.json','training_report.json')]+[path/'model.json',path/'booster.lgb.txt']:
            protected[str(p)]=original.sha(p)
        booster,meta=loader.load(path);manifest=original.read(source/'data_manifest.json')
        if original.sha(source/'data_manifest.json')!=meta['data_manifest_sha256']:raise safety.MLFailure('artifact_invalid')
        train=manifest['parts']['final_train'];rows=importance.annotate(booster,meta,train)
        if abs(sum(r['gain_share_percent'] for r in rows)-100)>1e-10:raise safety.MLFailure('invalid_model')
        write_csv(directory/(key+'_feature_importance.csv'),rows)
        models.append({'key':key,'title':title,'model_id':meta['model_id'],'best_iteration':meta['best_iteration'],
            'temperature':meta['temperature'],'train_rows':train['rows'],'train_races':train['races'],
            'rows':rows,'total_gain':sum(r['gain'] for r in rows),'total_splits':sum(r['split_count'] for r in rows),
            'rest_features':importance.rest_summary(rows)})
    comparison=importance.compare(models[0]['rows'],models[1]['rows'])
    write_csv(directory/'importance_comparison.csv',comparison)
    write_csv(directory/'rest_feature_comparison.csv',[r for r in comparison if r['extra_rest_feature']])
    bundle={'feature_count':47,'models':models,'source_files_sha256':protected,'created_at':original.clock().isoformat(),
        'actual_new_fits':0,'database_connections':0,'individual_rows_or_predictions_read':False,
        'legend':'青＝既存39項目 / 橙＝追加の間隔・休養8項目 / ALL NA＝最終学習データが全欠損 / ZERO＝重要度0',
        'gain_note':'数値：Gain（小数4桁） | 総Gain構成比（%）。CSVは丸め前の値。全47項目を表示。',
        'split_note':'数値：実際の分割回数。全47項目を表示。',
        'interpretation_note':'保存済み木の学習時重要度。因果効果や未知レースでの改善量ではありません。再学習なし。',
        'ordering':'own_metric_descending_then_saved_feature_index;comparison_CSV_initial_gain_order',
        'gain_difference_sign':'improved_minus_initial','rank_improvement_sign':'initial_rank_minus_improved_rank;positive_is_up',
        'chart_common_axis_max':{'gain':max(r['gain'] for m in models for r in m['rows']),
                                 'split':max(r['split_count'] for m in models for r in m['rows'])}}
    safety.write_json(directory/'importance_summary.json',bundle,root=ROOT)
    args=[r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
        str(Path(__file__).resolve().parent/'render_next_importance.ps1'),'-BundlePath',str(directory/'importance_summary.json')]
    process=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',timeout=90,check=False)
    rendered=json.loads(process.stdout)
    if process.returncode or rendered.get('status')!='ok' or rendered.get('untruncated_labels') is not True:
        raise safety.MLFailure('io_failed')
    for model in models:
        with (directory/(model['key']+'_feature_importance.csv')).open(encoding='utf-8-sig',newline='') as stream:back=list(csv.DictReader(stream))
        if len(back)!=47 or any(float(a['gain'])!=b['gain'] or int(a['split_count'])!=b['split_count'] or a['japanese']!=b['japanese'] for a,b in zip(back,model['rows'])):
            raise safety.MLFailure('artifact_invalid')
    if any(original.sha(Path(p))!=h for p,h in protected.items()):raise safety.MLFailure('artifact_invalid')
    safety.write_json(directory/'validation.json',{'all_47_features_each_model':True,'bilingual_labels_complete':True,
        'full_precision_CSV_round_trip':True,'source_files_unchanged':len(protected),'actual_new_fits':0,
        'render':rendered,'output_sha256':{p.name:safety.digest(p,root=ROOT) for p in directory.iterdir() if p.is_file()}},root=ROOT)
    return {'status':'ok','output_dir':str(directory),'features_per_model':47,'PNG_count':4,'CSV_count':4,'new_fits':0,
        'models':[{'key':m['key'],'model_id':m['model_id'],'rest_features':m['rest_features']} for m in models]}


if __name__=='__main__':
    try:print(json.dumps(run(),ensure_ascii=True,allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'stopped','reason':error.code if isinstance(error,safety.MLFailure) else 'internal_error','stage':'next_importance'}));raise SystemExit(1)
