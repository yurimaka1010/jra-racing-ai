"""Reuse saved 2023-25 rosters, O1 pay regimes, predictions, and selected labels."""
from collections import Counter
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re

from ..ml import safety
from ..model_diagnostics import load_sources, S4
from ..objective_pilot import contracts as pilot_contract
from ..objective_pilot.data import KEY_FIELDS, read_frame
from . import contracts as c


def prepare(directory):
    import numpy as np
    references = {}
    def reference(path, expected=None):
        path = safety.local_path(path, root=c.BASE, exists=True); got = safety.digest(path, root=c.BASE)
        if expected is not None and got != expected:
            raise safety.MLFailure('artifact_invalid')
        references[str(path)] = got
        return path
    frame, models, _, _ = load_sources(reference)
    frame = frame.sort_values('entry_key').reset_index(drop=True)
    for value in frame.date.unique(): c.day(value)
    # Already saved predictions: no model fit, no new selection from results.
    model_info = {m['label']: {'model_id': m['model_id'], 'kind': 'saved_annual_prediction'}
                  for m in models if m['label'].startswith(('H_', 'ABCD_'))}
    morning = {}
    for name in pilot_contract.NAMES:
        manifest = safety.read_json(reference(c.PILOT / f'fit_{name}.result.json'), root=c.BASE)
        path = reference(c.PILOT / f'{name}.predictions.local.csv', manifest['prediction_sha256'])
        import pandas as pd
        f = pd.read_csv(path, dtype={'entry_key': 'string', 'date': 'string'}, float_precision='round_trip')
        if (len(f) != 29879 or f.entry_key.duplicated().any() or not f.date.str.startswith('2025-').all()
                or set(f.entry_key) != set(frame.loc[frame.date.str.startswith('2025-'), 'entry_key'])):
            raise safety.MLFailure('invalid_identity')
        morning[name] = dict(zip(f.entry_key, f.probability))
        model_info[name] = {'model_id': manifest['model_id'], 'kind': 'saved_2025_objective_pilot_prediction',
                            'source_sha256': manifest['prediction_sha256']}
    # An old place model is an explicit rank-proxy reference, not an official target model.
    from ..ml import model as legacy_model
    legacy_dir = c.BASE / 'ml-v01/models/f1cce38718a4473ca1967b2252cb7997'
    legacy_meta = safety.read_json(reference(legacy_dir / 'model.json'), root=c.BASE)
    for name, sha in legacy_meta['artifacts'].items(): reference(legacy_dir / name, sha)
    bundle = legacy_model.load_models(legacy_dir, root=c.BASE)
    fm = safety.read_json(reference(c.S2 / 'feature_manifest.json'), root=c.BASE)
    feature = next(row for row in fm['years'] if row['year'] == 2025)
    f = read_frame(reference(c.S2 / 'features' / feature['file'], feature['sha256']))
    wanted = set(frame.loc[frame.date > '2025-01-19', 'entry_key'])
    f = f.loc[f.entry_key.isin(wanted)].sort_values('entry_key')
    if set(f.entry_key) != wanted:
        raise safety.MLFailure('invalid_identity')
    inputs = [{name: None if pd.isna(value) else value for name, value in row.items()}
              for row in f[legacy_meta['features']].to_dict('records')]
    p = bundle['boosters']['place'].predict(legacy_model.frame(inputs, legacy_meta['categories']), num_threads=4)
    legacy_values = dict(zip(f.entry_key, map(float, p)))
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise safety.MLFailure('invalid_model')
    model_info['legacy_place_proxy'] = {'model_id': legacy_dir.name, 'kind': 'saved_proxy_model_reference_inference_0fit',
        'label_contract': legacy_meta['label_contract'], 'after_fit_only': '2025-01-19',
        'not_claimed_official_place_model': True}
    # First commit immutable prediction references. Outcomes are joined afterwards.
    c.write(directory, 'prediction_commit.json', {'models': model_info, 'source_references': dict(references),
        'generated_at': datetime.now(timezone.utc).isoformat(), 'prediction_kind': 'retrospective_saved_prediction_replay',
        'selection': 'all_saved_6373_races_before_result_masks', 'new_fits': 0, 'results_used_for_prediction': False,
        '2026_rows_read': 0, 'Stage7_results_read': 0})
    sm = safety.read_json(reference(c.S2 / 'source_manifest.json'), root=c.BASE)
    wanted = set(frame.entry_key); months = {key[:6] for key in wanted}; found = {}; rosters = {}
    for key in wanted: rosters.setdefault(key[:16], set()).add(key)
    observed = {key: set() for key in rosters}
    for item in sm['months']:
        if item['file'][:6] not in months: continue
        path = reference(c.S2 / 'source' / item['file'], item['sha256'])
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = ''.join(row[name] for name in KEY_FIELDS)
                if key[:16] in observed:
                    if key in observed[key[:16]]: raise safety.MLFailure('duplicate_entry')
                    observed[key[:16]].add(key)
                if key not in wanted: continue
                if key in found or any(not re.fullmatch('[0-9]{2}', row[n]) for n in ('finish_position', 'actual_runners')):
                    raise safety.MLFailure('invalid_identity')
                found[key] = (int(row['finish_position']), int(row['actual_runners']))
    if set(found) != wanted or observed != rosters:
        raise safety.MLFailure('invalid_identity')
    # Reuse Stage 5 same-record checks, not a new raw-O1 parser.
    from ..stage5 import place
    from ..stage4 import data as market_data
    start, end = date(2023, 1, 1), date(2025, 9, 30)
    place_manifest = safety.read_json(reference(c.S5 / 'place.complete.json'), root=c.BASE)
    for path, sha in place_manifest['files'].items():
        if Path(path).name[:4] in ('2023', '2024', '2025'): reference(Path(path), sha)
    raw = place.inputs(c.S5, start, end); packets = market_data.market(S4, start, end)
    counts = Counter(); output = directory / 'races.local.jsonl'
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        for race, f in frame.groupby(frame.entry_key.str[:16], sort=True):
            c.identity(race, str(f.date.iloc[0])); keys = f.entry_key.tolist(); n = len(f)
            if any(found[k][1] != n or not 1 <= found[k][0] <= n for k in keys):
                raise safety.MLFailure('invalid_identity')
            if not all(int(found[k][0] == 1) == int(y) for k, y in zip(keys, f.winner_label)):
                raise safety.MLFailure('invalid_identity')
            pieces = (race[:4], race[4:8], race[8:10], race[10:12], race[12:14], race[14:16])
            record = raw.get(pieces); paid = None; reason = 'same_asof_place_regime_unavailable'
            if record and not record.get('read_error'):
                values, reason, pay = place.decode(pieces, record['rows'], dict(zip(f.entry_key.str[-2:], f.raw_market)), packets[pieces])
                if pay in ('2', '3') and (values is not None or reason == 'place_capped_or_ambiguous'):
                    paid = int(pay)
            counts['races'] += 1; counts['runners'] += n
            counts['place_pay_' + str(paid)] += 1
            predictions = {name + '_PL': dict(zip(f.entry_key.str[-2:], f[name].astype(float))) for name in ('H', 'raw_market', 'ABCD')}
            if str(f.date.iloc[0]).startswith('2025-'):
                for name, value in morning.items(): predictions[name + '_PL'] = {k[-2:]: float(value[k]) for k in keys}
            item = {'race_id': race, 'date': str(f.date.iloc[0]), 'paid_positions': paid,
                'pay_regime_reason': reason, 'finishes': {k[-2:]: found[k][0] for k in keys},
                'predictions': predictions, 'legacy_place_proxy': {k[-2:]: legacy_values[k] for k in keys} if all(k in legacy_values for k in keys) else None,
                'official_HR_context': 'not_connected;rank_references_only',
                'timing_basis': 'T_minus_10_JRA_publication_time_replay_original_PC_receipt_unknown'}
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    c.write(directory, 'data_manifest.json', {'version': c.VERSION, 'source_references': references,
        'file': output.name, 'sha256': safety.digest(output, root=c.ROOT), 'counts': dict(counts),
        'official_multiwager_HR_labels': 0, 'official_multiwager_payouts': 0, 'new_real_fits': 0,
        'sealed_2026_rows_read': 0, 'Stage7_results_read': 0, 'database_connections': 0,
        'legacy_place_proxy_reference_only': True, 'price_missing_does_not_imply_pay_regime_missing': True})
    return dict(counts)
