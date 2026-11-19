"""Sequential finite experiment. Baseline is persisted before place-input work."""
from pathlib import Path
from datetime import date
import json
import numpy as np
import pandas as pd
from ..ml import safety
from ..research import experiment as previous_exp
from ..stage4 import accuracy as old_accuracy,data as old_data,store as old_store,contracts as old_c
from . import contracts as c,store as s,metrics


def timing(year,split):
    first=f'{year}-01-01'
    for name in ('train_end','validation_end','calibration_end'):
        if not isinstance(split.get(name),str) or split[name]>=first:raise safety.MLFailure('invalid_split')
    if not split['train_end']<split['validation_start']<=split['validation_end']<split['calibration_start']<=split['calibration_end']:
        raise safety.MLFailure('invalid_split')


def source_times():
    out=[]
    for year in c.YEARS:
        for name,source in (('H',old_data.trial(c.S4,year,'H')),('ABCD',old_data.trial(c.S4,year,'A'))):
            path=c.S2/'models'/source['model_id']/'model.json';meta=safety.read_json(path,root=c.S2);timing(year,meta['split'])
            out.append({'year':year,'model':name,'model_id':source['model_id'],'split':meta['split'],'manifest_sha256':s.sha(path,c.S2)})
        choices=old_store.read(c.S4/'direct',f'{year}_selection.json')['selected']
        for name in ('Hcal','Hstack'):
            timing(year,choices[name]['split']);out.append({'year':year,'model':name,'split':choices[name]['split'],
                'selection_end':choices[name]['split']['validation_end'],'configuration_sha256':s.sha(c.S4/'direct'/f'{year}_selection.json',c.S4)})
        source=old_store.completed(c.S4/'H60',str(year))['summary'];path=c.S4/'models'/source['model_id']/'model.json'
        meta=safety.read_json(path,root=c.S4);timing(year,meta['split']);out.append({'year':year,'model':'H60','model_id':source['model_id'],'split':meta['split'],'manifest_sha256':s.sha(path,c.S4)})
        e=safety.read_json(c.S3/'E_trials'/f'{year}.json',root=c.S3)
        path=c.S3/'models'/e['model']['model_id']/'model.json';meta=safety.read_json(path,root=c.S3);timing(year,meta['split'])
        out.append({'year':year,'model':'ABCD_E','model_id':e['model']['model_id'],'split':meta['split'],'manifest_sha256':s.sha(path,c.S3)})
    return out


def evaluate(frame,predictions,*,expected=None):
    names=tuple(predictions)
    if 'H' not in names:raise safety.MLFailure('invalid_input')
    y,ph,races,weeks=metrics.assemble(frame,predictions['H']);bootstrap=metrics.WeekBootstrap(weeks)
    point={};draws={};reproduction=0.
    for name,values in predictions.items():
        old_accuracy.scoring.align(values,predictions['H'])
        labels,p,rs,ws=metrics.assemble(frame,values)
        if not np.array_equal(y,labels) or not np.array_equal(races,rs) or not np.array_equal(weeks,ws):raise safety.MLFailure('invalid_identity')
        value,draw=bootstrap.measure(y,p,races)
        legacy=old_accuracy.scoring.measure(values,previous_exp.winners(frame))
        for key,oldkey in (('race_log_loss','race_log_loss'),('horse_log_loss','log_loss'),('brier_score','brier_score'),('top1_accuracy_percent','top1_accuracy_percent')):
            reproduction=max(reproduction,abs(value[key]-legacy[oldkey]))
            if expected and name in expected:reproduction=max(reproduction,abs(value[key]-expected[name][oldkey]))
        point[name]=dict(value,roc_auc_95_percent_ci=metrics.bounds(draw['roc_auc']),races=int(len(np.unique(races))),
            runners=len(frame),positives=int(y.sum()),negatives=int((1-y).sum()))
        draws[name]=draw
    if reproduction>1e-12:raise safety.MLFailure('artifact_invalid')
    comparisons={n:metrics.compare(point[n],draws[n],point['H'],draws['H']) for n in names}
    winner=max(names,key=lambda name:point[name]['roc_auc'])
    best={'model':winner,'roc_auc':point[winner]['roc_auc'],'tied_models':[n for n in names if point[n]['roc_auc']==point[winner]['roc_auc']],
        'descriptive_only_not_formal_superiority':True}
    h=point['H'];ci=h['roc_auc_95_percent_ci']
    result={'metrics':point,'versus_H':comparisons,'highest_existing_AUC':best,'H_progress':{'roc_auc':h['roc_auc'],'95_percent_ci':ci,
        'point_estimate_above_080':h['roc_auc']>.80,'CI_lower_above_080':ci[0]>.80 if ci else False,
        'already_reached_by_existing_H':h['roc_auc']>.80,'not_80_percent_top1_accuracy':True},
        'bootstrap':bootstrap.metadata(),'legacy_metric_reproduction_max_difference':reproduction}
    if 'raw_market' in point:result['H_minus_raw_market']=metrics.compare(point['H'],draws['H'],point['raw_market'],draws['raw_market'])
    return result,draws


def baseline(directory):
    done=s.completed(directory,'baseline')
    if done:return done
    origins=source_times();frames,predictions,winners=old_accuracy.load(c.S4)
    expected=old_store.read(c.S4,'accuracy.json')['metrics'];annual={};draw_files={}
    for year in c.YEARS:
        annual[str(year)],draw=evaluate(frames[year],predictions[year],expected=expected['years'][str(year)])
        for name,values in draw.items():
            for metric,v in values.items():draw_files[f'{year}__{name}__{metric}']=v
    frame=pd.concat([frames[y] for y in c.YEARS],ignore_index=True)
    combined={name:[p for y in c.YEARS for p in predictions[y][name]] for name in predictions[c.YEARS[0]]}
    pooled,draw=evaluate(frame,combined,expected=expected['pooled'])
    for name,values in draw.items():
        for metric,v in values.items():draw_files[f'pooled__{name}__{metric}']=v
    h=pooled['metrics']['H']
    if (h['races'],h['runners'],h['positives'],h['negatives'])!=(6373,87601,6373,81228):raise safety.MLFailure('invalid_identity')
    expected_fingerprint=old_store.read(c.S4,'accuracy.json')['comparisons']['Hstack_minus_H']['pooled']['resample_indices_sha256']
    if pooled['bootstrap']['sample_indices_sha256']!=expected_fingerprint:raise safety.MLFailure('artifact_invalid')
    path=safety.local_path(directory/'baseline_bootstrap.local.npz',root=c.ROOT)
    with path.open('xb') as stream:np.savez_compressed(stream,**draw_files)
    result={'pooled':pooled,'years':annual,'source_timing':origins,'fixed_full_cohort':True,'LightGBM_fits':0,
        'no_probability_temperature_offset_normalization_reapplied':True,'baseline_saved_before_place_experiment':True}
    s.write(directory,'baseline.json',result)
    best=pooled['highest_existing_AUC'];summary={'H':pooled['H_progress'],'highest_existing_AUC':best,'models_measured':list(combined),
        'races':6373,'runners':87601,'LightGBM_fits':0,'old_metrics_reproduced':True}
    s.complete(directory,'baseline',[directory/'baseline.json',path],summary);return summary


def place(directory):
    if not s.completed(directory,'baseline'):raise safety.MLFailure('artifact_invalid')
    from .place import collect
    return collect(directory)


def fit(directory):
    from .candidate import run
    return run(directory)


def report(directory):
    from .report import run
    return run(directory)
