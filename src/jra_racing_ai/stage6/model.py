"""Six reserved fit calls at most; serialized residuals always use external offsets."""
from uuid import uuid4
import numpy as np
from ..ml import safety,model as legacy
from ..research import model as shared
from . import contracts as c,data,store as s


def matrix(frame,config):
    data.identities(frame)
    if config not in c.CONFIGS:raise safety.MLFailure('invalid_model')
    out=frame.loc[:,list(c.CONFIGS[config])].astype(float)
    if np.isinf(out.to_numpy()).any():raise safety.MLFailure('invalid_input')
    return out


def predict(bundle,frame):
    meta=bundle['metadata'];residual=bundle['boosters']['win'].predict(matrix(frame,meta['config']),raw_score=True,num_threads=4)
    return shared.normalize(frame,shared.sigmoid(residual+shared.offset(frame.market_probability.to_numpy(float))),1.)


def load(directory):
    import lightgbm as lgb
    meta=safety.read_json(directory/'model.json',root=c.ROOT);config=meta.get('config');year=meta.get('evaluation_year')
    if (config not in c.CONFIGS or year not in c.YEARS or meta.get('format_version')!=6 or meta.get('model_kind')!='stage6_binary_market_offset'
            or meta.get('features')!=list(c.CONFIGS[config]) or meta.get('categories')!={} or meta.get('formula_version')!=c.VERSION
            or meta.get('source_formula_version')!=c.SOURCE_FORMULA or meta.get('input_format')!='saved_T10_H8_plus_lag7_corner_proxy_v1'
            or meta.get('calibration')!={'temperature':1.,'method':'identity_after_external_offset'} or meta.get('history_lag_days')!=7
            or s.old.comparable(meta.get('split'))!=s.old.comparable(c.split(year)) or meta.get('training_parameters')!=c.PARAMETERS
            or meta.get('production_hr_settlement_enabled') is not False):raise safety.MLFailure('invalid_model')
    path=directory/'win.lgb.txt'
    if s.sha(path)!=meta['artifacts']['win.lgb.txt']:raise safety.MLFailure('artifact_invalid')
    return {'metadata':meta,'boosters':{'win':lgb.Booster(model_str=path.read_text(encoding='utf-8'))}}


def reserve(directory,config,year):
    if config not in c.CONFIGS or year not in c.YEARS:raise safety.MLFailure('invalid_input')
    inventory=s.inventory(directory)
    for row in inventory['records']:
        if (row['config'],row['year'])==(config,year):return row,False
    if inventory['fits_started']>=6:raise safety.MLFailure('input_limit')
    row={'config':config,'year':year,'model_id':uuid4().hex,'status':'started','started_at_jst':s.now(),'fit_calls_reserved':1}
    # Six fixed, exclusive reservations also forbid concurrent duplicate fit calls.
    safety.write_json(directory/f'fit_{config}_{year}.started.json',row,root=c.ROOT)
    inventory['records'].append(row);inventory['fits_started']+=1;s.inventory(directory,inventory)
    return row,True


def update(directory,record,**values):
    inventory=s.inventory(directory)
    matches=[r for r in inventory['records'] if r['model_id']==record['model_id']]
    if len(matches)!=1:raise safety.MLFailure('artifact_invalid')
    matches[0].update(values);s.inventory(directory,inventory);return matches[0]


def fit(directory,config,year,parts):
    plan_hash=s.verify_plan(directory);split=c.split(year)
    for name,f in parts.items():
        data.identities(f)
        if name=='evaluation':a,b=split['evaluation'][0]
        else:a,b=split[name+'_start'],split[name+'_end']
        if str(f.date.min())<a or str(f.date.max())>b:raise safety.MLFailure('invalid_split')
    record,new=reserve(directory,config,year);path=directory/'models'/record['model_id']
    try:
        if not new:
            if record['status']=='failed' or not (path/'model.json').exists():raise safety.MLFailure('training_failed')
            bundle=load(path)
            if (bundle['metadata']['config'],bundle['metadata']['evaluation_year'],bundle['metadata']['plan_sha256'])!=(config,year,plan_hash):
                raise safety.MLFailure('artifact_invalid')
            if record['status']=='complete' and (s.sha(path/'model.json')!=record['model_sha256'] or s.sha(path/'win.lgb.txt')!=record['tree_sha256']):
                raise safety.MLFailure('artifact_invalid')
        else:
            import lightgbm as lgb
            train,validation=parts['train'],parts['validation'];path=safety.local_path(path,root=c.ROOT);path.mkdir(parents=True,exist_ok=False)
            t=lgb.Dataset(matrix(train,config),label=train.winner_label.to_numpy(),init_score=shared.offset(train.market_probability))
            v=lgb.Dataset(matrix(validation,config),label=validation.winner_label.to_numpy(),reference=t,init_score=shared.offset(validation.market_probability))
            booster=lgb.train(c.PARAMETERS,t,num_boost_round=300,valid_sets=[v],callbacks=[lgb.early_stopping(20,verbose=False)])
            meta={'format_version':6,'model_kind':'stage6_binary_market_offset','config':config,'evaluation_year':year,'split':split,
                'features':list(c.CONFIGS[config]),'categories':{},'formula_version':c.VERSION,'source_formula_version':c.SOURCE_FORMULA,
                'input_format':'saved_T10_H8_plus_lag7_corner_proxy_v1','history_lag_days':7,
                'calibration':{'temperature':1.,'method':'identity_after_external_offset'},'training_parameters':c.PARAMETERS,
                'offset':'external_logit(q)_train_early_and_predict;sigmoid_then_race_normalize','best_iteration':booster.best_iteration,
                'max_rounds':300,'early_stopping_rounds':20,'plan_sha256':plan_hash,'inputs_sha256':s.sha(directory/'inputs.json'),
                'library_versions':legacy.versions(),'production_hr_settlement_enabled':False,'run_id':directory.name,
                'feature_definitions':s.read(directory,'plan.json')['formulas']}
            bundle={'metadata':meta,'boosters':{'win':booster}};legacy.save_models(bundle,path,root=c.ROOT)
        before=predict(bundle,parts['evaluation']);loaded=load(path);after=predict(loaded,parts['evaluation'])
        difference=float(np.max(np.abs(before-after)))
        if difference>1e-12:raise safety.MLFailure('artifact_invalid')
        row=update(directory,record,status='complete',completed_at_jst=s.now(),best_iteration=loaded['metadata']['best_iteration'],
            reload_max_probability_difference=difference,model_sha256=s.sha(path/'model.json'),tree_sha256=s.sha(path/'win.lgb.txt'))
        return loaded,row,after
    except Exception:
        if record['status']!='complete':update(directory,record,status='failed',failed_at_jst=s.now(),reason='training_or_artifact_validation_failed_no_retry')
        raise
