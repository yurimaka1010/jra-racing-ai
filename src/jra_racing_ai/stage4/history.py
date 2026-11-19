from datetime import date
import json
import numpy as np
from ..ml import safety
from ..research import experiment as exp
from . import contracts as c,store as s,data,models,quotes,direct


def useful(frame):
    for name in c.EXTRA[:2]:
        values=frame[name].to_numpy(float);finite=values[np.isfinite(values)]
        if len(finite)>1 and np.ptp(finite)>1e-12:return True
    return False


def run(directory):
    if not s.completed(directory,'quotes60'):raise safety.MLFailure('artifact_invalid')
    done=s.completed(directory,'h60')
    if done:return done['summary']
    folder=directory/'H60';folder.mkdir(exist_ok=True);files=[];catalog=[]
    for year in c.YEARS:
        done=s.completed(folder,str(year))
        if done:catalog.append(done['summary']);files.append(folder/f'{year}.complete.json');continue
        split=c.h_split(year);parts=models.parts(directory,split,augment=lambda f:quotes.augment(directory,f))
        if not useful(parts[0]):
            summary={'year':year,'status':'skipped','reason':'new_history_all_missing_or_no_numeric_variation','LightGBM_fits':0}
            s.write(folder,f'{year}_skipped.json',summary);s.complete(folder,str(year),[folder/f'{year}_skipped.json'],summary)
        else:
            bundle,origin=models.fit(directory,'H60',year,split,parts)
            f=data.market_frame(directory,date(year,1,1),date(year,9,30));augmented=quotes.augment(directory,f)
            p=models.predict(bundle,augmented);s.write(folder,f'{year}_predictions.local.json',direct.document(exp.predictions(f,p)))
            summary=dict(origin,status='complete',LightGBM_fits=1,races=int(f.entry_key.str[:16].nunique()),runners=len(f),
                missing_new_history_runners=int(augmented[c.EXTRA[2]].sum()),
                train_new_history_available_runners=int((parts[0][c.EXTRA[2]]==0).sum()),
                same_T10_primary_population=True,temperature=1.,no_Hstack_combination=True)
            s.complete(folder,str(year),[folder/f'{year}_predictions.local.json',directory/'fits'/f'{year}_H60.complete.json'],summary)
        catalog.append(summary);files.append(folder/f'{year}.complete.json')
        print(json.dumps(dict(stage='h60',**summary)),flush=True)
    result={'years':catalog,'LightGBM_fits':sum(x['LightGBM_fits'] for x in catalog),'skipped':sum(x['status']=='skipped' for x in catalog)}
    s.complete(directory,'h60',files,result);return result
