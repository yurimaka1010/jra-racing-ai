"""Bounded, local-only preparation; never calls train/fit or reads H outcomes."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date
import csv
import os
import time

from ..db_connection import read_only_session
from ..inspect_schema import COLUMNS_SQL
from ..win_sim_cli import connection_config
from ..ml import safety
from . import contracts as c


def preservation():
    # Reuse the existing finite checks: models, signed plans, protected code/tasks.
    value = c.enriched.prior.preservation()
    refs = {}
    for directory in (c.INITIAL, c.REFERENCE):
        for name in ('candidate_freeze.json', 'plan.json', 'data_manifest.json', 'training_report.json', 'source_manifest.json'):
            path = directory / name
            if path.exists(): refs[str(path)] = c.sha(path)
        freeze = c.read(directory / 'candidate_freeze.json')
        for name in ('model.json','booster.lgb.txt'):
            path = Path(freeze['model_path']) / name; refs[str(path)] = c.sha(path)
        for path in (directory / 'forward' / '2026-10-25' / 'races').glob('*/*'):
            if path.is_file() and path.name in ('prediction.json','predictions.csv','receipt.json','veto.json'):
                refs[str(path)] = c.sha(path)
    return {'existing': value, 'references': refs}


def verify(directory):
    before = c.read(directory / 'preservation_before.json')
    result = c.enriched.prior.verify_preservation(before['existing'])
    for path, sha in before['references'].items():
        if c.sha(Path(path)) != sha: raise safety.MLFailure('artifact_invalid')
    return dict(result, additional_immutable_files_unchanged=len(before['references']), real_fits=0)


def raw_days():
    from ..d39_generation.data import iter_days
    for rows in iter_days(c.enriched.RAW_PARENT):
        from ..research.features import legacy
        day = legacy.entry_identity(rows[0])[1]
        if day > date(2026,10,18): raise safety.MLFailure('invalid_split')
        yield rows


def compare_control():
    from ..d39_enrichment import data
    from ..objective_pilot.data import read_frame
    from ..research.contracts import spec
    parent = c.enriched.RAW_PARENT
    freeze = c.read(parent / 'control_candidate_freeze.json')
    control = c.read(Path(freeze['model_path']) / 'model.json')
    current = c.read(Path(c.read(c.REFERENCE / 'candidate_freeze.json')['model_path']) / 'model.json')
    old_manifest = c.read(parent / 'data_manifest.json')
    rows = {}
    for name in ('final_train','calibration'):
        new = data.load_part(c.REFERENCE,name)
        old = read_frame(parent / old_manifest['parts'][name]['file'])
        common = sorted(set(new.entry_key) & set(old.entry_key))
        fields = spec(('A','B','C','D'))['features']
        a = new.set_index('entry_key').loc[common,fields]
        b = old.set_index('entry_key').loc[common,fields]
        different = {f: int((~((a[f] == b[f]) | (a[f].isna() & b[f].isna()))).sum()) for f in fields}
        rows[name] = {'reference47_rows':len(new),'existing_control_rows':len(old), 'common_rows':len(common),
                      'identical_keys':list(new.entry_key)==list(old.entry_key), 'different_values_by_feature':different}
    return {'reusable':False, 'reason':'cohort_common_values_history_parameters_calibration_not_identical',
            'parts':rows, 'parameters_equal':control['parameters']==current['parameters'],
            'rounds':{'reference47':current['best_iteration'],'existing_control':control['best_iteration']},
            'temperature_policy_equal':False, 'categories_equal':control['categories']=={k:v for k,v in current['categories'].items() if k!='e_rest_stage'},
            'required_new_control_fits':1, 'pedigree_fits':1, 'reference47_refits_needed':0,
            'planned_total_fits':2, 'fits_authorized':False}


def input_replay(directory):
    """Predictions only: never call saved()/evaluate() that can join outcomes."""
    import pandas as pd
    from ..d39_next import forward, learning as initial_learning
    from ..d39_enrichment import forward as enriched_forward, learning as enriched_learning, data
    from ..research import model as matrices
    from ..win_sim.core import instant
    day = date(2026,10,25); report = {}
    for name, root, load, history, matrix in (
        ('initial47', c.INITIAL, initial_learning.load, lambda:forward.history(day), __import__('jra_racing_ai.d39_next.data',fromlist=['matrix']).matrix),
        ('time_connected47', c.REFERENCE, enriched_learning.load, lambda:enriched_forward.history(c.REFERENCE,day), data.matrix)):
        meta_path = Path(c.read(root/'candidate_freeze.json')['model_path'])
        bundle = load(meta_path); booster, meta = bundle; builder = history()
        sums=Counter(); missing=Counter(); unique=defaultdict(set); max_raw=max_prob=0.; schema_types=None
        preserved_inputs=[]
        for receipt_path in sorted((root/'forward'/'2026-10-25'/'races').glob('*/receipt.json')):
            receipt = c.read(receipt_path)
            if receipt.get('status')!='saved' or (receipt_path.parent/'veto.json').exists(): continue
            path=receipt_path.parent/'prediction.json'
            if c.sha(path)!=receipt['prediction_sha256']: raise safety.MLFailure('artifact_invalid')
            record=c.read(path)
            if record['result_accessed'] is not False or record['model_id']!=meta['model_id']: raise safety.MLFailure('artifact_invalid')
            source=record['entry_snapshot']; entry_path=Path(source['path'])
            safety.local_path(entry_path,root=c.BASE,exists=True)
            if c.sha(entry_path)!=source['sha256'] or instant(source['first_seen_at'])>instant(record['input_cutoff']):
                raise safety.MLFailure('artifact_invalid')
            entries=c.read(entry_path)['payload']
            frame=forward.live_frame(builder,entries)
            if set(frame.entry_key)!=set(r['entry_key'] for r in record['rows']): raise safety.MLFailure('invalid_identity')
            encoded=matrix(frame,meta['feature_spec'],meta['categories'])
            if list(encoded.columns)!=meta['features'] or booster.feature_name()!=meta['features']: raise safety.MLFailure('invalid_model')
            dtypes={k:str(v) for k,v in encoded.dtypes.items()}
            if schema_types is not None and schema_types!=dtypes: raise safety.MLFailure('invalid_input')
            schema_types=dtypes
            for f in meta['features']:
                missing[f]+=int(frame[f].isna().sum());unique[f].update(map(str,frame[f].dropna().unique()))
                if isinstance(encoded[f].dtype,pd.CategoricalDtype) and list(encoded[f].cat.categories)!=meta['categories'][f]:
                    raise safety.MLFailure('invalid_model')
            scores=booster.predict(encoded,num_threads=4)
            probabilities=matrices.normalize(frame,scores,meta['temperature'])
            stored={r['entry_key']:r for r in record['rows']}
            max_raw=max(max_raw,max(abs(float(v)-stored[k]['raw_probability']) for k,v in zip(frame.entry_key,scores)))
            max_prob=max(max_prob,max(abs(float(v)-stored[k]['probability']) for k,v in zip(frame.entry_key,probabilities)))
            sums.update(races=1,rows=len(frame))
            if len(entries)!=len(record['roster']) or any(r.get('finish_position') not in (None,'') for r in entries):
                raise safety.MLFailure('artifact_invalid')
            preserved_inputs.append({'prediction_sha256':receipt['prediction_sha256'],'entry_sha256':source['sha256'],
                                     'regenerated_matrix_sha256':__import__('hashlib').sha256(encoded.to_csv(index=False).encode()).hexdigest()})
        final=data.load_part(c.REFERENCE,'final_train') if name=='time_connected47' else __import__('jra_racing_ai.d39_next.data',fromlist=['load_part']).load_part(c.INITIAL,'final_train')
        cats=matrices.schema(final,meta['feature_spec'])
        constants={f: int(final[f].nunique(dropna=True)) for f in meta['features'] if final[f].nunique(dropna=True)<=1}
        report[name]={'races':sums['races'],'rows':sums['rows'],'model_id':meta['model_id'],
            'max_raw_probability_difference':max_raw,'max_final_probability_difference':max_prob,
            'saved_predictions_reproduced':bool(sums['races'] and max_raw<1e-12 and max_prob<1e-12),
            'columns_order_types_checked':True,'dtypes':schema_types,'training_category_dictionary_matches':cats==meta['categories'],
            'feature_missing_in_replay':dict(missing),'training_constant_or_all_missing_unique_counts':constants,
            'raw_47_column_matrix_saved_at_T10':False,
            'full_original_matrix_value_equality':'unverifiable_original_engineered_matrix_not_saved',
            'historical_data_identity':'pinned_source_cache_hashes_reused;no_current_DB_values_substituted',
            'historical_PC_receipt_and_later_source_corrections':'unverified',
            'current_actual_starters_asof':'not_input;declared_runners_is_registration_count;history_actual_count_validated_by_clean_mask',
            'history_lag_days':7,'current_results_read':False,'replay_evidence':preserved_inputs}
    c.write(directory/'input_replay.json',report)
    return {n:{k:v[k] for k in ('races','rows','saved_predictions_reproduced','max_raw_probability_difference','max_final_probability_difference')} for n,v in report.items()}


def session():
    return read_only_session(connection_config(10000), os.environ.get('PGPASSWORD'), gate_timeout_ms=3000)


def inventory(directory):
    if (directory / 'schema.json').exists():
        schema = c.read(directory / 'schema.json')
    else:
        with session() as db:
            tables = db.aggregate('pedigree.tables',
                "SELECT c.relname AS name FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p') AND c.relname IN ('jvd_um','jvd_hn','jvd_sk','jvd_we') ORDER BY 1", max_rows=4)
            schema = {r['name']: db.aggregate('pedigree.columns', COLUMNS_SQL, ('public', r['name']), max_rows=1000) for r in tables}
        c.write(directory / 'schema.json', schema)
    return {'status': 'ok', 'kind': 'pedigree_schema', 'run_id': directory.name,
            'real_fits': 0, 'tables': {t: [v['name'] for v in rows] for t, rows in schema.items()}}


def prepare(directory):
    from . import sources
    from .features import Builder, stable_id, cell
    from ..research import features as legacy
    began=time.monotonic()
    if (directory/'preparation.json').exists(): raise safety.MLFailure('artifact_exists')
    if not (directory/'preservation_before.json').exists(): c.write(directory/'preservation_before.json',preservation())
    if not (directory/'feature_spec.json').exists(): c.write(directory/'feature_spec.json',c.spec())
    inventory(directory)
    horses=set();raw_counts=Counter()
    for rows in raw_days():
        horses.update(h for r in rows if (h:=stable_id(r.get('horse_id'))))
        raw_counts['rows']+=len(rows);raw_counts['days']+=1
    c.write(directory/'cache_scope.json',dict(raw_counts,unique_horses=len(horses),results_scope='existing_authorized_training_history_only;through2026-10-18'))
    print(__import__('json').dumps({'status':'running','stage':'identity','unique_horses':len(horses)}),flush=True)
    lineage,links=sources.collect(directory,horses,session)
    c.write(directory/'link_summary.json',links)
    print(__import__('json').dumps({'status':'running','stage':'input_replay'}),flush=True)
    replays=input_replay(directory)
    control=compare_control();c.write(directory/'control_reuse.json',control)
    b=Builder(lineage);summaries=defaultdict(Counter); thickness=defaultdict(Counter);missing=Counter();parts={}
    # Match the reference row cohort exactly; no lineage or optional-value filtering.
    from ..d39_enrichment.data import load_part
    targets={}
    for part in ('final_train','calibration'):
        frame=load_part(c.REFERENCE,part);targets[part]=set(frame.entry_key)
        parts[part]={'expected_rows':len(frame),'expected_races':int(frame.entry_key.str[:16].nunique())}
    files={};writers={};observed=defaultdict(set)
    try:
        for part in targets:
            path=directory/(part+'.pedigree.local.csv');files[part]=path.open('x',encoding='utf-8',newline='')
            writers[part]=csv.DictWriter(files[part],fieldnames=['entry_key',*c.FEATURES]);writers[part].writeheader()
        for rows in raw_days():
            if time.monotonic()-began>900:raise safety.MLFailure('query_timeout')
            values=b.transform_day(rows)
            for row,x in zip(rows,values):
                key=''.join(legacy.legacy.entry_identity(row)[0])
                for part,wanted in targets.items():
                    if key not in wanted:continue
                    writers[part].writerow(dict(entry_key=key,**x));observed[part].add(key)
                    condition=cell(row.get('surface'),row.get('distance'))
                    for role in ('sire','damsire'):
                        stats=summaries[(part,role,condition or 'unsupported')]
                        support=x[f'p_{role}_support'];stats['target_rows']+=1;stats['with_support']+=int(support>0)
                        stats['support_sum']+=support;stats['support_lt5']+=int(support<5)
                        thickness[(part,role)][support]+=1
                        missing[part+':'+role]+=int(lineage.get(row.get('horse_id'),{}).get(role) is None)
        for part in targets:
            if observed[part]!=targets[part]:raise safety.MLFailure('artifact_invalid')
    finally:
        for stream in files.values():stream.close()
    cells=[]
    for (part,role,condition),counts in sorted(summaries.items()):
        cells.append(dict(part=part,role=role,cell=condition,**counts))
    def quantile(counts,q):
        threshold=sum(counts.values())*q;passed=0
        for value,n in sorted(counts.items()):
            passed+=n
            if passed>=threshold:return value
        return None
    summary={'condition_cells':cells,'support_quantiles':{part+':'+role:{str(q):quantile(counts,q) for q in (0,.25,.5,.75,1)} for (part,role),counts in thickness.items()},
        'unknown_parent_rows':dict(missing),'history_counts':dict(b.counts),
        'parts':{part:dict(v,sha256=c.sha(directory/(part+'.pedigree.local.csv')),actual_rows=len(observed[part])) for part,v in parts.items()},
        'all_target_rows_retained':True,'future_or_current_race_results_in_features':False,
        'identity_master_historical_revisions':'not_reconstructable_from_current_master;research_identity_only'}
    c.write(directory/'pedigree_summary.json',summary)
    result={'status':'prepared_awaiting_training_approval','run_id':directory.name,'real_fits':0,
        'link_summary':links,'input_replay':replays,'control_reusable':control['reusable'],
        'next_stage_fits':control['planned_total_fits'],'pedigree_rows':{k:v['actual_rows'] for k,v in summary['parts'].items()},
        'preservation':verify(directory),'elapsed_seconds':time.monotonic()-began}
    c.write(directory/'preparation.json',result)
    return result
