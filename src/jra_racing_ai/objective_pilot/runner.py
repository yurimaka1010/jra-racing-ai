"""Bounded local workers, exclusive fit ledger, and descriptive development report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..ml import safety
from . import contracts as c

PROJECT = Path(__file__).resolve().parents[3]


def memory_bytes(process_id):
    """Windows private commit, including memory not currently in the working set."""
    import ctypes as ct
    from ctypes import wintypes as w
    class Counters(ct.Structure):
        _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [
            (n, ct.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
             'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
             'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    api = ct.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; api.OpenProcess.restype = w.HANDLE
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.OpenProcess(0x1000 | 0x10, False, process_id)
    if not handle:
        return None
    try:
        ps = ct.WinDLL('psapi', use_last_error=True)
        ps.GetProcessMemoryInfo.argtypes = [w.HANDLE, ct.POINTER(Counters), w.DWORD]
        data = Counters(); data.cb = ct.sizeof(data)
        if not ps.GetProcessMemoryInfo(handle, ct.byref(data), data.cb):
            return None
        return int(data.PrivateUsage)
    finally:
        api.CloseHandle(handle)


def process_parents():
    """PID/parent pairs only. Do not read process command lines or credentials."""
    import ctypes as ct
    from ctypes import wintypes as w
    class Entry(ct.Structure):
        _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('th32ProcessID', w.DWORD),
            ('th32DefaultHeapID', ct.c_size_t), ('th32ModuleID', w.DWORD), ('cntThreads', w.DWORD),
            ('th32ParentProcessID', w.DWORD), ('pcPriClassBase', w.LONG), ('dwFlags', w.DWORD),
            ('szExeFile', w.WCHAR * 260)]
    api = ct.WinDLL('kernel32', use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    api.CreateToolhelp32Snapshot.restype = w.HANDLE
    api.Process32FirstW.argtypes = [w.HANDLE, ct.POINTER(Entry)]
    api.Process32NextW.argtypes = [w.HANDLE, ct.POINTER(Entry)]
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.CreateToolhelp32Snapshot(2, 0)
    if handle == ct.c_void_p(-1).value:
        raise safety.MLFailure('input_limit')
    try:
        entry = Entry(); entry.dwSize = ct.sizeof(entry); out = {}
        if not api.Process32FirstW(handle, ct.byref(entry)):
            raise safety.MLFailure('input_limit')
        while True:
            out[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not api.Process32NextW(handle, ct.byref(entry)):
                break
        return out
    finally:
        api.CloseHandle(handle)


def descendants(root, parents):
    seen = {root}; frontier = {root}
    while frontier:
        frontier = {pid for pid, parent in parents.items() if parent in frontier and pid not in seen}
        seen.update(frontier)
    return seen


def tree_memory(root):
    # Windows venv python.exe is a redirector; the real interpreter is a descendant.
    values = [memory_bytes(pid) for pid in descendants(root, process_parents())]
    return sum(v for v in values if v is not None) if any(v is not None for v in values) else None


def terminate_worker(child):
    # Stop the interpreter too, not just the Windows venv redirector.
    subprocess.run([str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/taskkill.exe'),
                    '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=10, check=False)
    child.wait(timeout=10)


def bounded_worker(directory, phase, seconds, name=None):
    """Only one worker at a time; parent+child private commit <= approved 12 GiB."""
    args = [sys.executable, '-B', str(PROJECT / 'scripts/run_objective_pilot.py'), '_worker',
            '--run-id', directory.name, '--phase', phase]
    if name is not None:
        args.extend(['--name', name])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    begin = time.monotonic(); peak = 0; status = 'complete'; reason = None
    with subprocess.Popen(args, cwd=str(PROJECT), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, encoding='utf-8') as child:
        while child.poll() is None:
            used = tree_memory(child.pid); own = memory_bytes(os.getpid())
            if used is not None and own is not None:
                peak = max(peak, used + own)
                if used + own > 12 * 1024 ** 3:
                    status = 'failed'; reason = 'input_limit'; terminate_worker(child); break
            elif child.poll() is None:
                status = 'failed'; reason = 'input_limit'; terminate_worker(child); break
            if time.monotonic() - begin >= seconds:
                status = 'failed'; reason = 'query_timeout'; terminate_worker(child); break
            time.sleep(.25)
        stdout, _ = child.communicate(timeout=10)
        if status == 'complete':
            try:
                result = json.loads(stdout.strip())
                if child.returncode != 0 or result.get('status') != 'ok':
                    status = 'failed'; reason = result.get('reason', 'training_failed')
            except (json.JSONDecodeError, TypeError):
                status = 'failed'; reason = 'training_failed'
        return {'status': status, 'reason': reason if reason in safety.REASONS else ('training_failed' if reason else None),
                'wall_seconds': time.monotonic() - begin, 'peak_parent_and_worker_private_bytes': peak,
                'phase': phase, 'name': name, 'stdout_contains_individual_data': False}


def verify_protection(directory):
    from ..stage4.store import task_configuration
    before = c.read(directory, 'protection_before.json')
    changed = sum(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
                  for p, h in before['project_files'].items())
    assets = sum(not Path(p).is_file() or safety.digest(Path(p), root=c.BASE) != h
                 for p, h in before['protected_assets'].items())
    tasks = task_configuration()
    same_tasks = tasks == before['tasks'] and tasks.get('status') == 'observed'
    result = {'original_project_files': len(before['project_files']), 'changed_original_files': changed,
        'protected_assets': len(before['protected_assets']), 'changed_protected_assets': assets,
        'task_configuration_unchanged': same_tasks, 'task_mutations': 0,
        'sealed_2026_used': False, 'Stage7_results_used': False,
        'old_Stage7_exact_source_set_report_audit': 'new_modules_added;recorder_scope_unchanged;existing_report_mode_limitation'}
    c.write(directory, 'preservation.json', result)
    if changed or assets or not same_tasks:
        raise safety.MLFailure('artifact_invalid')
    return result


def reference_ability(frame):
    from ..model_diagnostics import S4, S5
    refs = safety.read_json(S4 / 'references.json', root=c.BASE)
    path = c.S2 / 'trials/2025_annual_ability.predictions.local.json'
    if safety.digest(path, root=c.BASE) != refs[str(path)]:
        raise safety.MLFailure('artifact_invalid')
    document = safety.read_json(path, root=c.BASE); wanted = set(frame.entry_key.str[:16]); matched = set(); values = {}
    for part in document:
        for row in part:
            race = ''.join(row['race_key'])
            if race not in wanted:
                continue
            if race in matched:
                raise safety.MLFailure('duplicate_entry')
            matched.add(race)
            for horse, p in row['probabilities'].items():
                values[race + horse] = p
    if set(values) != set(frame.entry_key):
        raise safety.MLFailure('invalid_identity')
    frame['saved_ABCD'] = frame.entry_key.map(values).astype(float)
    old = safety.read_json(S5 / 'baseline.json', root=c.BASE)['years']['2025']['metrics']
    return old, {'ABCD_prediction_source': str(path), 'sha256': refs[str(path)],
                 'baseline_report_sha256': safety.digest(S5 / 'baseline.json', root=c.BASE)}


def calibration_table(y, p):
    import numpy as np
    edges = (0, .01, .02, .05, .1, .2, .3, .5, 1.)
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        result.append({'lower_inclusive': lo, 'upper': hi, 'runners': int(mask.sum()),
            'mean_prediction': float(p[mask].mean()) if mask.any() else None,
            'observed_win_rate': float(y[mask].mean()) if mask.any() else None})
    return result


def report(directory):
    import numpy as np
    import pandas as pd
    from ..model_diagnostics import predictions, binary_metrics, top1_decisions
    from ..stage5 import metrics
    from .data import load_part
    f = load_part(directory, 'evaluation'); expected, refs = reference_ability(f)
    trials = []; names = ['H', 'raw_market', 'saved_ABCD']; boot = None; points = {}; draws = {}; tables = {}
    for name in c.NAMES:
        result = directory / f'fit_{name}.result.json'
        if not result.exists():
            continue
        item = c.read(directory, result.name)
        path = directory / f'{name}.predictions.local.csv'
        if safety.digest(path, root=c.ROOT) != item['prediction_sha256']:
            raise safety.MLFailure('artifact_invalid')
        got = pd.read_csv(path, dtype={'entry_key': 'string', 'date': 'string'}, float_precision='round_trip')
        if (not np.array_equal(got.entry_key, f.entry_key) or not np.array_equal(got.date, f.date)
                or not np.array_equal(got.winner_label.to_numpy(float), f.winner_label.to_numpy(float))):
            raise safety.MLFailure('invalid_identity')
        f[name] = got.probability.to_numpy(float); names.append(name); trials.append(item)
    max_reference_delta = 0.
    for name in names:
        y, p, codes, weeks = metrics.assemble(f, predictions(f, name))
        if boot is None:
            boot = metrics.WeekBootstrap(weeks, repetitions=1000)
        points[name], draws[name] = boot.measure(y, p, codes)
        if name in ('H', 'raw_market', 'saved_ABCD'):
            old_name = 'ABCD' if name == 'saved_ABCD' else name
            delta = max(abs(points[name][k] - expected[old_name][k]) for k in metrics.NAMES)
            max_reference_delta = max(delta, max_reference_delta)
            if delta > 1e-12:
                raise safety.MLFailure('artifact_invalid')
        tables[name] = {'probability_bands': calibration_table(y, p),
            'probability_ge_0_5': binary_metrics(y, p >= .5),
            'top1_horse_number_tie_break': binary_metrics(y, top1_decisions(f.entry_key.tolist(), p))}
    comparisons = {}
    base = 'paired_binary_control'
    if base in points:
        for name in c.NAMES[1:]:
            if name in points:
                comparisons[name] = {k: {'difference_new_minus_binary_control': points[name][k] - points[base][k],
                    '95_percent_ci': metrics.bounds(draws[name][k] - draws[base][k])} for k in metrics.NAMES}
    quarters = {}
    for lo, hi, label in (('2025-01-01', '2025-03-31', 'Jan_Mar'), ('2025-04-01', '2025-06-30', 'Apr_Jun'),
                           ('2025-07-01', '2025-09-30', 'Jul_Sep')):
        part = f.loc[(f.date >= lo) & (f.date <= hi)]
        quarters[label] = {name: metrics.point_only(part, predictions(part, name)) for name in names}
    selected = min((n for n in c.NAMES if n in points), key=lambda n: points[n]['race_log_loss'], default=None)
    frozen = {'version': c.VERSION, 'proposal_sha256': c.PROPOSAL_HASH,
        'trial_results': trials, 'descriptive_lowest_race_LL': selected, 'automatic_adoption': False,
        'production_H_unchanged': True, 'evaluation_is_reused_development': True,
        'more_training_or_search_authorized': False, 'purchase_rules_changed': False}
    c.write(directory, 'candidate_frozen.json', frozen)
    value = {'version': c.VERSION, 'status': 'complete' if len(trials) == 4 else 'partial_or_failed',
        'races': int(f.entry_key.str[:16].nunique()), 'runners': len(f), 'models': points,
        'new_minus_paired_binary_control': comparisons, 'bootstrap': boot.metadata(),
        'quarters': quarters, 'calibration_and_classification': tables, 'trials': trials,
        'reference_reproduction_max_difference': max_reference_delta, 'reference_sources': refs,
        'matched_training_windows_only_for_four_new_models': True,
        'H_market_saved_ABCD_are_reference_not_matched_training_controls': True,
        'reused_development_not_independent_test': True, 'ROI_computed': False,
        'sealed_2026_used': False, 'Stage7_results_used': False, 'automatic_adoption': False}
    c.write(directory, 'development_report.json', value)
    lines = ['# 目的関数4構成の有限比較', '', '2025年1〜9月の既存開発評価。独立した最終テストではありません。',
             '4構成は同じABCD 39列・同じ学習期間。H・市場・旧ABCDは学習期間が異なる参考比較です。', '',
             f'対象：{value["races"]:,}レース・{value["runners"]:,}頭。追加探索・自動採用・購入条件変更は行いません。', '',
             '|モデル|レースLL|馬LL|Brier|AUC|レース内AUC|1位的中率|', '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in points.items():
        lines.append(f'|{name}|{m["race_log_loss"]:.6f}|{m["horse_log_loss"]:.6f}|{m["brier_score"]:.6f}|'
                     f'{m["roc_auc"]:.6f}|{m["within_race_auc"]:.6f}|{m["top1_accuracy_percent"]:.2f}%|')
    lines += ['', '## 主比較：新モデル − 同条件二値対照', '', '負のレースLL差が改善方向です。95%週クラスタ区間は探索用の記述値です。']
    for name, comparison in comparisons.items():
        v = comparison['race_log_loss']; bounds = v['95_percent_ci']
        lines.append(f'- {name}: {v["difference_new_minus_binary_control"]:+.6f} [{bounds[0]:+.6f}, {bounds[1]:+.6f}]')
    lines += ['', '## 学習と較正', '']
    for t in trials:
        lines.append(f'- {t["name"]}: best_iteration={t["best_iteration"]}, trees={t["trees"]}, T={t["temperature"]}, model_id={t["model_id"]}')
    lines += ['', f'参考モデルの既存指標との差の最大値：{max_reference_delta:.3g}。',
              '全特徴量欠損率・較正帯・四半期別成績・補助分類指標はJSONに保存。',
              '回帰・ランキングの生スコアを勝率とは扱わず、較正期間で固定したSoftmax温度を適用。',
              'going・previous_time_secondsの全欠損は既存契約どおり維持。特徴量追加の効果を検証した実験ではありません。',
              '2026年1〜9月およびStage 7の途中成績は使用していません。ROI・収益改善は未評価です。',
              '既存Hを維持します。候補は今回の開発評価内での比較記録であり、本番への昇格を意味しません。']
    text = '\n'.join(lines) + '\n'
    with (directory / 'development_report.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    with (directory / 'development_report.html').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                     '<title>目的関数比較</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:32px auto;padding:16px}'
                     'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>有限4構成比較</h1><pre>' + escape(text) + '</pre></html>')
    return {'models_complete': len(trials), 'races': value['races'], 'runners': value['runners'],
            'descriptive_lowest_race_LL': selected, 'reference_max_difference': max_reference_delta}


def execute(directory):
    p = c.plan(); authorization = c.read(directory, 'authorization.json')
    if authorization['proposal_sha256'] != c.PROPOSAL_HASH or authorization['fits_max'] != 4:
        raise safety.MLFailure('explicit_execution_required')
    # One approved execution across directories. A rerun cannot silently receive four more fits.
    safety.write_json(c.ROOT / ('execution_' + c.PROPOSAL_HASH + '.json'),
                      {'run_id': directory.name, 'approved_fits_max': 4}, root=c.ROOT)
    c.write(directory, 'execution_started.json', {'utc': datetime.now(timezone.utc).isoformat(),
            'script_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in Path(__file__).parent.glob('*.py')}})
    began = time.monotonic(); records = []; fitted_seconds = 0
    status = 'stopped'; reason = None
    try:
        preflight = bounded_worker(directory, 'preflight', 30)
        c.write(directory, 'resource_preflight.json', preflight)
        if preflight['status'] != 'complete':
            raise safety.MLFailure(preflight['reason'])
        prep = bounded_worker(directory, 'prepare', 300)
        c.write(directory, 'preparation_status.json', prep)
        if prep['status'] != 'complete':
            raise safety.MLFailure(prep['reason'])
        print(json.dumps({'status': 'running', 'stage': 'labels_and_rosters_validated', 'real_fits': 0}), flush=True)
        for name in c.NAMES:
            remaining = min(600, 1800 - fitted_seconds, 2700 - (time.monotonic() - began))
            if remaining <= 0:
                break
            c.reserve(directory, name)
            item = bounded_worker(directory, 'fit', remaining, name)
            fitted_seconds += item['wall_seconds']; records.append(item)
            c.write(directory, f'fit_{name}.worker.json', item)
            print(json.dumps({'status': 'running', 'stage': 'fit_finished', 'name': name,
                              'fit_status': item['status'], 'fits_consumed': len(records)}), flush=True)
        remaining = 2700 - (time.monotonic() - began)
        if remaining <= 0:
            raise safety.MLFailure('query_timeout')
        final = bounded_worker(directory, 'report', remaining)
        c.write(directory, 'report_status.json', final)
        if final['status'] != 'complete':
            raise safety.MLFailure(final['reason'])
        status = 'complete' if all(x['status'] == 'complete' for x in records) and len(records) == 4 else 'partial'
    except safety.MLFailure as error:
        reason = error.code
    finally:
        preservation = verify_protection(directory)
        completion = {'status': status, 'reason': reason, 'run_id': directory.name,
            'fits_consumed': len(list(directory.glob('fit_*.started.json'))),
            'successful_fits': len(list(directory.glob('fit_*.result.json'))), 'fit_workers': records,
            'fit_wall_seconds': fitted_seconds, 'whole_job_seconds': time.monotonic() - began,
            'preservation': preservation, 'more_training_permitted': False}
        c.write(directory, 'completion.json', completion)
    return {k: completion[k] for k in ('status', 'reason', 'run_id', 'fits_consumed', 'successful_fits', 'whole_job_seconds')}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Approved maximum four-fit, local-only objective pilot')
    parser.add_argument('command', choices=('execute', '_worker'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=('preflight', 'prepare', 'fit', 'report'))
    parser.add_argument('--name', choices=c.NAMES)
    args = parser.parse_args(argv)
    try:
        directory = c.folder(args.run_id)
        if args.command == 'execute':
            result = execute(directory)
        else:
            p = c.plan()
            if args.phase == 'preflight':
                import lightgbm
                if memory_bytes(os.getpid()) is None or tree_memory(os.getpid()) is None:
                    raise safety.MLFailure('input_limit')
                result = {'LightGBM_import': True, 'memory_monitor': True, 'fits': 0}
            elif args.phase == 'prepare':
                from .data import prepare
                result = prepare(directory, p)
            elif args.phase == 'fit':
                from .model import fit
                result = fit(directory, p, args.name)
            elif args.phase == 'report':
                result = report(directory)
            else:
                raise safety.MLFailure('invalid_input')
            result = {'status': 'ok', 'result': result}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result.get('status') in ('ok', 'complete', 'partial') else 1
    except safety.MLFailure as error:
        print(json.dumps({'status': 'stopped', 'reason': error.code, 'stage': args.phase or args.command}), flush=True)
        return 1
    except Exception:
        print(json.dumps({'status': 'stopped', 'reason': 'internal_error', 'stage': args.phase or args.command}), flush=True)
        return 1
