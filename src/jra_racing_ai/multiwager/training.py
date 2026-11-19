"""Explicitly approved, serialized execution on separately verified training files."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4

from ..ml import safety
from ..objective_pilot.data import read_frame
from . import contracts as c, models

PROJECT = Path(__file__).resolve().parents[3]


def approved(directory):
    if not (directory / 'training_authorization.json').exists():
        raise safety.MLFailure('explicit_execution_required')
    auth = c.read(directory, 'training_authorization.json')
    digest = safety.digest(directory / 'training_plan.json', root=c.ROOT)
    if (auth.get('source') != 'human_user' or auth.get('allow_real_fits') is not True
            or auth.get('plan_sha256') != digest or auth.get('real_fits_max') != 2):
        raise safety.MLFailure('explicit_execution_required')
    if c.read(directory, 'training_plan.json') != c.training_plan():
        raise safety.MLFailure('artifact_invalid')
    return digest


@contextmanager
def lock(directory):
    path = safety.local_path(c.TRAINING_LOCK, root=c.BASE); token = uuid4().hex
    try: stream = path.open('x', encoding='utf-8')
    except FileExistsError: raise safety.MLFailure('input_limit') from None
    try:
        json.dump({'owner': token, 'run_id': directory.name, 'purpose': c.VERSION, 'pid': os.getpid()}, stream)
        stream.flush(); os.fsync(stream.fileno()); yield
    finally:
        stream.close()
        if safety.read_json(path, root=c.BASE).get('owner') == token: path.unlink()


def part(manifest_path, name, family):
    """Only caller-supplied, verified normalized HR labels; no raw payout parsing."""
    import pandas as pd
    if name not in ('train', 'early_stopping', 'calibration', 'evaluation'):
        raise safety.MLFailure('invalid_input')
    m = safety.read_json(manifest_path, root=c.BASE)
    if (m.get('version') != c.VERSION or m.get('formula_version') != models.existing_contracts.FORMULA_VERSION
            or m.get('history_lag_days') != 7 or m.get('sealed_2026_used') is not False
            or m.get('label_contract') != 'verified_ordinary_HR_place_targets'
            or not re.fullmatch('[0-9a-f]{64}', str(m.get('official_HR_source_sha256', '')))):
        raise safety.MLFailure('mapping_not_reviewed')
    item = m['parts'][name]; span = c.training_plan()[name]
    c.day(item['start']); c.day(item['end'])
    if not span[0] <= item['start'] <= item['end'] <= span[1]:
        raise safety.MLFailure('invalid_split')
    path = safety.local_path(Path(item['file']), root=c.BASE, exists=True)
    if safety.digest(path, root=c.BASE) != item['sha256']: raise safety.MLFailure('artifact_invalid')
    identifiers = pd.read_csv(path, usecols=['entry_key', 'date'], dtype='string')
    for day in identifiers.date.unique(): c.day(day)
    if not ((identifiers.date >= span[0]) & (identifiers.date <= span[1])).all():
        raise safety.MLFailure('invalid_split')
    cap = 100000 if name == 'train' else 30000 if name == 'evaluation' else 10000
    if len(identifiers) > cap or len(identifiers) != item['rows']: raise safety.MLFailure('input_limit')
    f = read_frame(path); models.validate_frame(f, family)
    return f


def reserve(directory, family):
    approved(directory)
    if len(list(directory.glob('fit_*.started.json'))) >= 2 or family not in c.FAMILIES:
        raise safety.MLFailure('input_limit')
    c.write(directory, f'fit_{family}.started.json', {'family': family, 'counts_as_attempt': True, 'retry': False})


def worker(directory, manifest_path, family):
    fingerprint = approved(directory)
    ownership = safety.read_json(c.TRAINING_LOCK, root=c.BASE)
    if ownership.get('run_id') != directory.name or ownership.get('purpose') != c.VERSION:
        raise safety.MLFailure('explicit_execution_required')
    if not (directory / 'training_started.json').exists(): raise safety.MLFailure('explicit_execution_required')
    c.write(directory, f'worker_{family}.entered.json', {'automatic_retry': False})
    frames = [part(manifest_path, name, family) for name in ('train', 'early_stopping', 'calibration')]
    result = models.fit_family(directory, family, *frames, approved_plan_sha256=fingerprint,
                              reserve_fit=lambda name: reserve(directory, name))
    evaluation = part(manifest_path, 'evaluation', family)
    from .scoring import Scores
    from .outcomes import rank_reference
    bundle = models.load(directory / 'models' / result['model_id']); predicted = models.predict(bundle, evaluation)
    scores = {w: Scores() for w in (('place',) if family == c.FAMILIES[0] else c.WAGERS)}
    with (directory / f'{family}.predictions.local.jsonl').open('x', encoding='utf-8') as stream:
        for race, f in evaluation.groupby(evaluation.entry_key.str[:16], sort=True):
            k = int(f.place_paid_positions.iloc[0]); ref, why = rank_reference(dict(zip(f.entry_key.str[-2:], f.finish_position.astype(int))), k)
            if why: raise safety.MLFailure('outcomes_incomplete')
            labels = {'place': dict(zip(f.entry_key.str[-2:], f.official_place_label.astype(int)))} if family == c.FAMILIES[0] else ref
            for wager in scores: scores[wager].add(predicted[race][wager], labels[wager])
            stream.write(json.dumps({'race_id': race, 'date': str(f.date.iloc[0]), 'probabilities': predicted[race],
                                    'model_id': result['model_id'], 'evaluation_kind': 'development_reuse'}, allow_nan=False) + '\n')
    measured = {w: s.result() for w, s in scores.items()}
    if family == c.FAMILIES[0]:
        for value in measured.values(): value['label_basis'] = 'verified_ordinary_official_HR_place_targets'
    c.write(directory, f'{family}.evaluation.json', {'model': result,
        'metrics': measured, 'ROI': None, 'automatic_adoption': False,
        'direct_place_labels': 'verified_official_HR' if family == c.FAMILIES[0] else 'conditional_order_rank_reference'})
    return result


def execute(directory, manifest_path):
    fingerprint = approved(directory)
    if manifest_path is None: raise safety.MLFailure('mapping_not_reviewed')
    manifest_path = safety.local_path(manifest_path, root=c.BASE, exists=True)
    from ..objective_pilot.runner import memory_bytes, tree_memory, terminate_worker
    results = []; began = time.monotonic()
    with lock(directory):
        c.write(c.ROOT, 'execution_' + fingerprint + '.json', {'run_id': directory.name, 'max_fits': 2})
        c.write(directory, 'training_started.json', {'utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': safety.digest(manifest_path, root=c.BASE), 'automatic_retry': False})
        for family in c.FAMILIES:
            args = [sys.executable, '-B', str(PROJECT / 'scripts/multiwager_research.py'), '_worker',
                    '--run-id', directory.name, '--training-manifest', str(manifest_path), '--family', family]
            start = time.monotonic(); reason = None; peak = 0
            env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with subprocess.Popen(args, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding='utf-8') as child:
                while child.poll() is None:
                    used = tree_memory(child.pid); own = memory_bytes(os.getpid())
                    if used is None or own is None:
                        if child.poll() is not None: break
                        reason = 'input_limit'
                    else:
                        peak = max(peak, used + own)
                        if used + own > 12 * 1024 ** 3: reason = 'input_limit'
                    if time.monotonic() - start >= 600 or time.monotonic() - began >= 1800: reason = 'query_timeout'
                    if reason: terminate_worker(child); break
                    time.sleep(.25)
                output, _ = child.communicate(timeout=10)
            try: outcome = json.loads(output.strip())
            except ValueError: outcome = {'status': 'stopped', 'reason': 'training_failed'}
            result = {'family': family, 'status': 'complete' if child.returncode == 0 and not reason and outcome.get('status') == 'ok' else 'stopped',
                'reason': reason or outcome.get('reason'), 'wall_seconds': time.monotonic() - start,
                'peak_parent_and_worker_private_bytes': peak}
            c.write(directory, f'worker_{family}.complete.json', result); results.append(result)
            if time.monotonic() - began >= 1800: break
    result = {'status': 'complete' if len(results) == 2 and all(x['status'] == 'complete' for x in results) else 'stopped',
        'fits_consumed': len(list(directory.glob('fit_*.started.json'))), 'results': results, 'automatic_retry': False}
    c.write(directory, 'training_complete.json', result)
    return result
