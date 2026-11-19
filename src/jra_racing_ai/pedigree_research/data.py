"""Read-only 39/45-column projections of the exact frozen 47-column cohort."""
from . import contracts as c
from ..ml.safety import MLFailure
from ..research.contracts import spec as legacy_spec
from ..d39_enrichment import data as reference
from ..objective_pilot.data import read_frame


def spec(pedigree=False):
    old=legacy_spec(('A','B','C','D'))
    return dict(old, format_version=9, formula_version=c.VERSION,
        input_format='exact_time_connected47_projection_39_plus_optional_pedigree6',
        features=[*old['features'],*(c.FEATURES if pedigree else ())],
        reference_formula_version=c.enriched.VERSION,
        history_policy='same_clean_race_builder_and_7day_lag_as_frozen47',
        pedigree_spec=c.spec() if pedigree else None)


def merge(reference_frame, extra, *, pedigree):
    old=spec()['features']
    if (reference_frame.entry_key.duplicated().any() or extra.entry_key.duplicated().any()
            or set(reference_frame.entry_key)!=set(extra.entry_key)):
        raise MLFailure('invalid_identity')
    base=reference_frame.loc[:,['entry_key','date','winner_label',*old]].copy()
    if pedigree:
        base=base.merge(extra.loc[:,['entry_key',*c.FEATURES]],on='entry_key',how='left',validate='one_to_one',sort=False)
        if base[list(c.FEATURES)].isna().any().any():raise MLFailure('invalid_input')
    if list(base.entry_key)!=list(reference_frame.entry_key):raise MLFailure('invalid_identity')
    return base


def load_part(directory,part,*,pedigree=False):
    if part not in ('final_train','calibration'):raise MLFailure('invalid_input')
    manifest=c.read(directory/'pedigree_summary.json')
    path=directory/(part+'.pedigree.local.csv')
    if c.sha(path)!=manifest['parts'][part]['sha256']:raise MLFailure('artifact_invalid')
    frame=reference.load_part(c.REFERENCE,part)
    return merge(frame,read_frame(path),pedigree=pedigree)


def matrix(frame,feature_spec,categories):
    # Reuse the existing categorical encoder unchanged; only add six float columns.
    import pandas as pd
    from ..research.model import matrix as legacy_matrix
    blood=feature_spec==spec(True)
    if feature_spec!=spec(blood):raise MLFailure('invalid_model')
    out=legacy_matrix(frame,legacy_spec(('A','B','C','D')),categories)
    if blood:
        for f in c.FEATURES:out[f]=pd.to_numeric(frame[f],errors='raise').astype('float64')
    return out.loc[:,feature_spec['features']]
