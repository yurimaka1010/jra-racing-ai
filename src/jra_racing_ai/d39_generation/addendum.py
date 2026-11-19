"""Zero-fit addendum: identities, diagnostic-only curves and sealed 21-week plan."""
from datetime import date,datetime,timedelta
from pathlib import Path
import hashlib
import json
import time

from ..ml import safety
from ..research import model as matrices
from ..tuner_pilot import experiment as tuner
from ..objective_pilot import model as calibration
from . import contracts as c,data,learning,control,forward
from .report import csv_file

VERSION='four_model_addendum_zero_fit_sealed_21weeks_v1'
PLAN='four_addendum_plan.json'
OUTPUT='forward-four-sealed'
PAIRS=(('new_D39','control_D39','Tuner_settings_including_rounds_and_temperature'),
       ('control_D39','old_D39','additional_data_including_data_derived_category_vocabulary'),
       ('new_D39','old_D39','total_change'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def model_paths(parent):
    return {'old_D39':c.OLD_MODEL,'new_D39':Path(c.read(parent/'candidate_freeze.json')['model_path']),
            'control_D39':Path(c.read(parent/'control_candidate_freeze.json')['model_path']),'H':c.h.MODEL}


def identity(parent):
    paths=model_paths(parent);new=c.read(paths['new_D39']/'model.json');ctrl=c.read(paths['control_D39']/'model.json')
    old=c.read(paths['old_D39']/'model.json');manifest=c.read(parent/'data_manifest.json')
    final=manifest['parts']['final_train'];file=parent/final['file'];csv_sha=c.sha(file)
    if (new['data_manifest_sha256']!=c.sha(parent/'data_manifest.json') or ctrl['data_manifest_sha256']!=new['data_manifest_sha256']
            or ctrl['same_final_train_sha256']!=csv_sha or final['sha256']!=csv_sha
            or new['features']!=ctrl['features'] or new['categories']!=ctrl['categories'] or new['feature_spec']!=ctrl['feature_spec']
            or ctrl['parameters']!=old['parameters'] or ctrl['temperature']!=old['temperature']
            or ctrl['best_iteration']!=old['best_iteration']):raise safety.MLFailure('artifact_invalid')
    # The existing CSV hash already proves all values and order. Add explicit
    # row-key/label and schema hashes without recreating any training data.
    frame=data.load_part(parent,'final_train');rows=hashlib.sha256()
    for key,day,y in zip(frame.entry_key,frame.date,frame.winner_label):
        rows.update((str(key)+'\t'+str(day)+'\t'+str(int(y))+'\n').encode())
    early=c.read(parent/'plan.json')['spans']['early_stopping']
    early_rows=int(((frame.date>=early['start'])&(frame.date<=early['end'])).sum())
    return {'status':'identical','final_train_CSV_sha256':csv_sha,'ordered_row_identity_and_labels_sha256':rows.hexdigest(),
        'feature_order_sha256':fingerprint(new['features']),'category_dictionary_with_order_sha256':fingerprint(new['categories']),
        'feature_spec_sha256':fingerprint(new['feature_spec']),'explicit_parameters_sha256':fingerprint(old['parameters']),
        'explicit_parameter_keys':sorted(old['parameters']),'seed':old['parameters'].get('seed'),
        'parameter_difference':{'added':[],'missing':[],'changed':[],'missing_filled_from_defaults':[]},
        'rows':len(frame),'races':int(frame.entry_key.str[:16].nunique()),'feature_count':len(new['features']),
        'final_procedure':{'new_D39':'one_fixed_round_refit_on_final_train','control_D39':'one_fixed_round_fit_on_final_train',
            'early_stopping_in_final_fit':False,'early_stop_window_is_merged_into_final_train':True,
            'early_window_rows_in_final_train':early_rows,'rounds_are_intentionally_different':{'new_D39':new['best_iteration'],'control_D39':ctrl['best_iteration']},
            'temperature_is_intentionally_different':{'new_D39':new['temperature'],'control_D39':ctrl['temperature']}},
        'evidence':'existing_hashed_final_train_and_saved_metadata;no_fit_or_data_regeneration'}


def prepare(parent):
    c.approved(parent)
    if (parent/PLAN).exists():return {'status':'ready','plan_sha256':c.sha(parent/PLAN),'reused':True,'new_fits':0}
    proof=c.read(parent/'addendum_identity.json') if (parent/'addendum_identity.json').exists() else identity(parent)
    paths=model_paths(parent);metas={n:c.read(p/'model.json') for n,p in paths.items()}
    references={str(p):c.sha(p) for root in paths.values() for p in root.iterdir()
                if p.name in ('model.json','booster.lgb.txt','win.lgb.txt')}
    baseline_refs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(forward.__file__),c.PROJECT/'src/jra_racing_ai/d39_generation/outcomes.py',
        parent/'forward_plan.json',parent/'start_three_model_comparison.cmd')}
    start=date(2026,10,24);monday=start-timedelta(days=start.weekday());end=monday+timedelta(weeks=21,days=-1)
    if not (parent/'addendum_identity.json').exists():c.write(parent/'addendum_identity.json',proof)
    plan={'version':VERSION,'created_at':c.clock().isoformat(),'start_date':str(start),'end_date':str(end),
        'week_slots':21,'week_definition':'fixed_Monday_starting_21_slots;empty_or_missing_weeks_retained;no_result_based_extension',
        'week_starts':[str(monday+timedelta(weeks=i)) for i in range(21)],
        'availability_deadline':'2026-10-24T07:00:00+09:00','decision_before_first_startup_required':True,
        'models':list(forward.FOUR_NAMES),'model_ids':{n:(c.h.H_ID if n=='H' else m['model_id']) for n,m in metas.items()},
        'model_paths':{n:str(p) for n,p in paths.items()},'temperatures':{n:(m['calibration']['temperature'] if n=='H' else m['temperature']) for n,m in metas.items()},
        'references':references,'base_three_unchanged_references':baseline_refs,'identity_sha256':c.sha(parent/'addendum_identity.json'),
        'output_directory':OUTPUT,'input_cutoff_minutes':10,'max_age_seconds':300,'save_deadline_seconds':60,
        'history_lag_days':7,'history_refresh':'read_only_full_JRA_population_to_target_day_minus_7;new_files_only;models_fixed',
        'control_failure':'write_missing_status_and_continue_base_three','startup_failure':'automatic_three_model_fallback;never_stop_base_three_for_control',
        'raw_probability_definition':{'ability':'booster_binary_probability_before_temperature_and_race_normalization',
            'H':'sigmoid(frozen_H_residual_raw_score+market_logit_offset);before_race_normalization;T1'},
        'policy':{'stake_yen':100,'threshold':'.05','max_points_per_race':1},
        'evaluation':{'only_once_after_window_end':True,'read_outcomes_before_end':False,'primary_metric':'race_log_loss',
            'difference_sign':'first_minus_second','pairs':[list(p) for p in PAIRS],
            'bootstrap_unit':'Monday_race_week','bootstrap_repetitions':5000,'seed':20261023,
            'primary_familywise_alpha':.05,'primary_pair_confidence':1-.05/3,'multiple_comparisons':'Bonferroni_3_primary_race_LL',
            'no_automatic_adoption':True,'Stage7_interim_results_opened':False},'additional_fits':0}
    c.write(parent/PLAN,plan)
    return {'status':'ready','start_date':str(start),'end_date':str(end),'weeks':21,'availability_deadline':plan['availability_deadline'],
            'identity':proof,'new_fits':0,'plan_sha256':c.sha(parent/PLAN)}


def approved(parent):
    c.approved(parent);plan=c.read(parent/PLAN)
    if plan.get('version')!=VERSION or plan.get('additional_fits')!=0 or plan.get('week_slots')!=21:
        raise safety.MLFailure('artifact_invalid')
    for path,digest in plan['base_three_unchanged_references'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise safety.MLFailure('artifact_invalid')
    if c.sha(parent/'addendum_identity.json')!=plan['identity_sha256']:raise safety.MLFailure('artifact_invalid')
    return plan


def diagnostics(parent):
    import numpy as np
    approved(parent)
    target=parent/'control_diagnostic_only.json'
    if target.exists():return {'status':'ok','reused':True,'new_fits':0,'diagnostic':c.read(target)}
    path=model_paths(parent)['control_D39'];booster,meta=learning.load(path)
    before={str(p):c.sha(p) for p in (path/'model.json',path/'booster.lgb.txt')}
    early=data.load_part(parent,'early_stopping');cal=data.load_part(parent,'calibration')
    x=matrices.matrix(early,meta['feature_spec'],meta['categories']);y=early.winner_label.to_numpy(int)
    curves=[];began=time.monotonic()
    for n in range(1,meta['best_iteration']+1):
        if time.monotonic()-began>120:raise safety.MLFailure('query_timeout')
        raw=np.asarray(booster.predict(x,num_iteration=n,num_threads=4),float);q=np.clip(raw,1e-15,1-1e-15)
        binary=float((-y*np.log(q)-(1-y)*np.log1p(-q)).mean())
        normalized=matrices.normalize(early,raw,meta['temperature'])
        curves.append({'rounds':n,'binary_log_loss':binary,'race_log_loss_at_fixed_T075':calibration.race_loss(early,normalized)})
    best=min(curves,key=lambda r:(r['binary_log_loss'],r['rounds']))
    raw=booster.predict(matrices.matrix(cal,meta['feature_spec'],meta['categories']),num_threads=4)
    temperature,grid=calibration.calibrate(cal,raw,{'objective':'binary','temperature_candidates':[.75,1.,1.25,2.]})
    for file,sha in before.items():
        if c.sha(Path(file))!=sha:raise safety.MLFailure('artifact_invalid')
    result={'status':'ok','model_id':meta['model_id'],'new_fits':0,'prediction_settings_unchanged':{'rounds':215,'temperature':.75},
        'new_early_window':{'start':str(early.date.min()),'end':str(early.date.max()),'races':int(early.entry_key.str[:16].nunique()),
            'rows':len(early),'overlaps_final_training':True,'role':'in_sample_prefix_diagnostic_NOT_validation_or_test'},
        'best_existing_tree_prefix_by_binary_LL':best,'searchable_rounds':[1,meta['best_iteration']],
        'right_censored_if_best_is_last':best['rounds']==meta['best_iteration'],'not_a_claim_of_global_optimal_tree_count':True,
        'calibration_window':{'start':str(cal.date.min()),'end':str(cal.date.max()),'rows':len(cal),'races':int(cal.entry_key.str[:16].nunique())},
        'diagnostic_only_temperature':temperature,'temperature_grid':grid,'used_for_future_predictions':False,
        'model_bytes_unchanged':True,'elapsed_seconds':time.monotonic()-began}
    csv_file(parent/'control_prefix_diagnostic.csv',curves);c.write(target,result)
    return {'status':'ok','new_fits':0,'diagnostic':result}


def verify_raw(parent):
    """Reuse the previously verified fictional files, exercising only the new sidecar."""
    import shutil
    from uuid import uuid4
    import numpy as np
    import pandas as pd
    from ..research import features as rf
    from ..win_sim import core
    from . import four_forward
    plan=dict(approved(parent),synthetic_verification=True)
    source=Path(c.read(parent/'four_model_completion.json')['saved_model_integration_verification']['saved_folder'])/'three'
    source_record=c.read(source/'prediction.json')
    if not source_record.get('synthetic_verification'):raise safety.MLFailure('artifact_invalid')
    target=forward.mkdir(parent/'raw-verify'/uuid4().hex);names=('prediction.json','predictions.csv','receipt.json')
    before={n:c.sha(source/n) for n in names}
    for name in names:shutil.copyfile(source/name,target/name)
    bundles,errors=four_forward.load_bundles(plan);now=core.instant(source_record['generated_at'])
    value=four_forward.sidecar(target,parent,plan,bundles,rf.Builder(),clock=lambda:now)
    if value['status']!='saved':raise safety.MLFailure('artifact_invalid')
    frame=pd.read_csv(target/'four_predictions.csv',dtype={'entry_key':str},float_precision='round_trip')
    if frame.raw_probability.isna().any() or set(frame.model)!=set(forward.FOUR_NAMES):raise safety.MLFailure('invalid_model')
    max_error=0.
    for name in forward.FOUR_NAMES:
        p=frame.loc[frame.model==name].sort_values('entry_key');raw=p.raw_probability.to_numpy(float)
        recovered=raw**(1/plan['temperatures'][name]);recovered/=recovered.sum()
        max_error=max(max_error,float(np.max(np.abs(recovered-p.probability.to_numpy(float)))))
    if max_error>1e-12:raise safety.MLFailure('invalid_model')
    if any(c.sha(target/n)!=before[n] or c.sha(source/n)!=before[n] for n in names):raise safety.MLFailure('artifact_invalid')
    result={'status':'ok','raw_saved_for_all_four_models':True,'rows':len(frame),'reconstruction_max_probability_error':max_error,
        'base_three_files_byte_identical':True,'original_three_source_files_unchanged':True,'model_ids':plan['model_ids'],
        'synthetic_inputs_reused':True,'results_read':False,'new_fits':0,'folder':str(target)}
    c.write(target/'verification.json',result);return result
