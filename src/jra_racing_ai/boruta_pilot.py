"""Training-only Boruta preparation. No selection is permitted without new approval."""
from contextlib import contextmanager
from datetime import date
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from unittest.mock import patch

from .ml import safety
from .research import contracts as features, model as matrices

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
ROOT = BASE / 'boruta-pilot'
VERSION = 'training_only_boruta_prepare_20261022_v1'
TRAIN_START, TRAIN_END = '2022-01-01', '2024-08-31'


def proposal():
    return {'version': VERSION, 'target': 'existing_win_ABCD39_feature_selection_not_new_research_family',
        'feature_spec': features.spec(('A','B','C','D')), 'train_start':TRAIN_START, 'train_end':TRAIN_END,
        'estimator':{'kind':'RandomForestClassifier','n_estimators':64,'max_depth':5,'n_jobs':4,'random_state':42},
        'BorutaPy':{'n_estimators':64,'max_iter':4,'perc':100,'alpha':.05,'two_step':True,'random_state':42,'verbose':0},
        'actual_forest_fit_max':3,'failed_attempt_consumes_budget':True,'automatic_retry':False,
        'while_contract':'installed_Boruta0.4.3_starts_iter1_and_loops_while_iter_less_than_max_iter',
        'selection_seconds':180,'maximum_train_rows':130000,'maximum_encoded_columns':1024,
        'preprocessing':'train_only_categories;all_missing_raw_columns_excluded;numeric_training_median;one_hot',
        'confirmed_and_tentative_kept_separate':True,'empty_confirmed_never_falls_back_to_all_features':True,
        'downstream_LightGBM_fits_proposed':1,'downstream_fit_in_this_module':False,
        'sealed_2026_used':False,'Stage7_results_used':False,'automatic_adoption':False}


def proposal_hash():
    return hashlib.sha256(json.dumps(proposal(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def compatibility():
    import inspect
    from boruta import BorutaPy
    version=importlib.metadata.version('Boruta')
    signature=inspect.signature(BorutaPy)
    source=inspect.getsource(BorutaPy._fit)
    if version!='0.4.3' or 'max_iter' not in signature.parameters or '_iter < self.max_iter' not in source:
        raise safety.MLFailure('missing_library')
    return {'status':'prepared_not_selected','Boruta_version':version,'actual_forest_fits':0,
            'LightGBM_fits':0,'plan':proposal(),'proposal_sha256':proposal_hash()}


def training_dates(frame):
    if frame.empty or len(frame)>130000 or 'date' not in frame:
        raise safety.MLFailure('input_limit')
    try:
        days=[date.fromisoformat(str(v)) for v in frame.date]
    except (ValueError,TypeError):
        raise safety.MLFailure('invalid_split') from None
    if any(not TRAIN_START<=d.isoformat()<=TRAIN_END for d in days):
        raise safety.MLFailure('invalid_split')


def make_preprocessor(train):
    """Only training rows may contribute category dictionaries or medians."""
    import numpy as np
    training_dates(train)
    spec=features.spec(('A','B','C','D')); categories=matrices.schema(train,spec)
    try:matrix=matrices.matrix(train,spec,categories)
    except safety.MLFailure:raise
    except Exception:raise safety.MLFailure('invalid_input') from None
    definitions=[];excluded=[];medians={}
    for name in spec['features']:
        if train[name].isna().all():
            excluded.append(name);continue
        if name in categories:
            offset=len(definitions)
            definitions.extend({'name':f'x{offset+i:04d}','source':name,'kind':'one_hot','category':category}
                               for i,category in enumerate(categories[name]))
        else:
            values=matrix[name].to_numpy(float)
            if np.isinf(values).any():raise safety.MLFailure('invalid_input')
            medians[name]=float(np.nanmedian(values))
            definitions.append({'name':f'x{len(definitions):04d}','source':name,'kind':'numeric'})
    if not definitions or len(definitions)>1024:raise safety.MLFailure('input_limit')
    return {'version':VERSION,'feature_spec':spec,'categories':categories,'numeric_training_medians':medians,
        'excluded_all_missing':excluded,'encoded_columns':definitions,'fit_start':str(train.date.min()),
        'fit_end':str(train.date.max()),'category_dictionary_scope':'training_only',
        'no_target_or_outcome_used':True}


def transform(frame,preprocessor):
    import numpy as np
    import pandas as pd
    if (preprocessor.get('version')!=VERSION or preprocessor['feature_spec']!=features.spec(('A','B','C','D'))
            or not TRAIN_START<=preprocessor['fit_start']<=preprocessor['fit_end']<=TRAIN_END):
        raise safety.MLFailure('invalid_mapping')
    if 'date' in frame and any(str(d)>='2026-01-01' for d in frame.date):
        raise safety.MLFailure('invalid_split')
    try:x=matrices.matrix(frame,preprocessor['feature_spec'],preprocessor['categories'])
    except safety.MLFailure:raise
    except Exception:raise safety.MLFailure('invalid_input') from None
    out={}
    for column in preprocessor['encoded_columns']:
        source=column['source']
        if column['kind']=='one_hot':
            out[column['name']]=(x[source]==column['category']).astype('float32').to_numpy()
        elif column['kind']=='numeric':
            values=x[source].to_numpy(float)
            if np.isinf(values).any():raise safety.MLFailure('invalid_input')
            out[column['name']]=np.where(np.isnan(values),preprocessor['numeric_training_medians'][source],values).astype('float32')
        else:raise safety.MLFailure('invalid_mapping')
    return pd.DataFrame(out,index=frame.index)


def result_masks(selector,preprocessor):
    import numpy as np
    n=len(preprocessor['encoded_columns']);strong=np.asarray(selector.support_,bool);weak=np.asarray(selector.support_weak_,bool)
    if strong.shape!=(n,) or weak.shape!=(n,):raise safety.MLFailure('invalid_mapping')
    # Boruta's rough fix may overlap support_ and support_weak_; never promote weak to confirmed.
    confirmed=strong & ~weak
    names=[column['name'] for column in preprocessor['encoded_columns']]
    sources=[column['source'] for column in preprocessor['encoded_columns']]
    return {'version':VERSION,'encoded_columns':names,'encoded_to_source':dict(zip(names,sources)),
        'library_raw_support':strong.tolist(),'library_raw_support_weak':weak.tolist(),
        'confirmed_mask':confirmed.tolist(),'tentative_mask':weak.tolist(),
        'confirmed_columns':[name for name,keep in zip(names,confirmed) if keep],
        'tentative_columns':[name for name,keep in zip(names,weak) if keep],
        'confirmed_source_features':sorted({source for source,keep in zip(sources,confirmed) if keep}),
        'tentative_source_features':sorted({source for source,keep in zip(sources,weak) if keep}),
        'source_feature_interpretation':'at_least_one_encoded_level;not_a_whole_raw_feature_adoption',
        'automatic_adoption':False}


def selected_matrix(frame,preprocessor,selection,*,include_tentative=False):
    if type(include_tentative) is not bool:raise safety.MLFailure('invalid_input')
    matrix=transform(frame,preprocessor)
    if selection['encoded_columns']!=list(matrix.columns):raise safety.MLFailure('invalid_mapping')
    names=list(selection['confirmed_columns'])
    if include_tentative:names=list(dict.fromkeys(names+selection['tentative_columns']))
    if not names:raise safety.MLFailure('insufficient_history')
    return matrix.loc[:,names]


def save_preparation(directory,preprocessor,selection=None):
    safety.write_json(directory/'preprocessor.json',preprocessor,root=ROOT)
    if selection is not None:safety.write_json(directory/'selection_mask.json',selection,root=ROOT)


@contextmanager
def training_lock(directory):
    lock=safety.local_path(BASE/'research-training.lock',root=BASE);token=uuid4().hex
    try:stream=lock.open('x',encoding='utf-8')
    except FileExistsError:raise safety.MLFailure('input_limit') from None
    try:
        json.dump({'owner':token,'pid':os.getpid(),'run_id':directory.name,'purpose':VERSION},stream)
        stream.flush();os.fsync(stream.fileno());yield
    finally:
        stream.close()
        try:
            if json.loads(lock.read_text(encoding='utf-8'))['owner']==token:lock.unlink()
        except (OSError,ValueError,KeyError):pass


def select_training(train,labels,*,directory,authorization):
    """Future approved selection API. Preparation/tests never call a real forest fit."""
    if (not isinstance(authorization,dict) or authorization.get('source')!='human_user'
            or authorization.get('proposal_sha256')!=proposal_hash() or authorization.get('allow_selection') is not True
            or authorization.get('actual_forest_fit_max')!=3):
        raise safety.MLFailure('explicit_execution_required')
    directory=safety.local_path(directory,root=ROOT)
    lock=safety.read_json(BASE/'research-training.lock',root=BASE)
    if lock.get('pid')!=os.getpid() or lock.get('run_id')!=directory.name or lock.get('purpose')!=VERSION:
        raise safety.MLFailure('explicit_execution_required')
    compatibility();training_dates(train)
    safety.write_json(ROOT/('execution_'+proposal_hash()+'.json'),{'run_id':directory.name,'repeat_forbidden':True},root=ROOT)
    preprocessor=make_preprocessor(train);x=transform(train,preprocessor)
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    y=np.asarray(labels)
    if y.shape!=(len(x),) or set(y)!={0,1}:raise safety.MLFailure('invalid_input')
    forest=RandomForestClassifier(**{k:v for k,v in proposal()['estimator'].items() if k!='kind'})
    selector=BorutaPy(forest,**proposal()['BorutaPy'])
    original=forest.fit;calls=0;deadline=time.monotonic()+180
    def counted(*args,**kwargs):
        nonlocal calls
        if calls>=3 or time.monotonic()>=deadline:raise safety.MLFailure('input_limit')
        calls+=1;number=calls
        safety.write_json(directory/f'forest_{number}.started.json',{'attempt':number,'consumes_budget':True},root=ROOT)
        state='failed'
        try:
            value=original(*args,**kwargs)
            if time.monotonic()>=deadline:raise safety.MLFailure('query_timeout')
            state='complete';return value
        finally:
            safety.write_json(directory/f'forest_{number}.{state}.json',{'attempt':number,'state':state},root=ROOT)
    try:
        with patch.object(forest,'fit',counted):selector.fit(x.to_numpy(),y.astype(int))
    except Exception:
        raise safety.MLFailure('training_failed') from None
    selection=result_masks(selector,preprocessor)
    selection.update(actual_forest_fit_calls=calls,maximum_forest_fit_calls=3,downstream_LightGBM_fits=0,
                     preprocessor_source='training_only',proposal_sha256=proposal_hash())
    save_preparation(directory,preprocessor,selection)
    return preprocessor,selection
