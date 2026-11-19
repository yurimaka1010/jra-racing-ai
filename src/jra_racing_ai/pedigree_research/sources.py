"""Reviewed UM/SK ancestry identifiers only. Never select cumulative statistics."""
from collections import Counter
from .features import stable_id
from . import contracts as c
from ..ml.safety import MLFailure, identifier

FIELDS = ('data_kubun', 'data_sakusei_nengappi', 'ketto_toroku_bango',
          'ketto_joho_01a', 'ketto_joho_02a', 'ketto_joho_05a')
HN_FIELDS = ('data_kubun', 'data_sakusei_nengappi', 'hanshoku_toroku_bango',
             'ketto_toroku_bango', 'ketto_joho_01a', 'ketto_joho_02a')


def query(table, ids):
    if table not in ('jvd_um', 'jvd_sk', 'jvd_hn') or not 1 <= len(ids) <= 1000 or any(not stable_id(i) for i in ids):
        raise MLFailure('invalid_input')
    key = 'hanshoku_toroku_bango' if table == 'jvd_hn' else 'ketto_toroku_bango'
    fields = HN_FIELDS if table == 'jvd_hn' else FIELDS
    sql = ('SELECT ' + ','.join(identifier(k) + '::text AS ' + identifier(k) for k in fields)
           + ' FROM ONLY public.' + identifier(table) + ' WHERE ' + identifier(key) + '=ANY(%s) LIMIT 3001')
    return sql, (list(ids),)


def index(rows, *, hn=False):
    """Identical snapshots deduplicate; conflicting or deleted links remain unknown."""
    key = 'hanshoku_toroku_bango' if hn else 'ketto_toroku_bango'
    groups = {}
    for row in rows:
        k = stable_id(row.get(key))
        if k:
            groups.setdefault(k, []).append(row)
    out = {}; stats = Counter()
    for k, items in groups.items():
        signatures = {tuple(r.get(f) for f in ('ketto_joho_01a', 'ketto_joho_02a', 'ketto_joho_05a')) for r in items}
        allowed = ('1', '2') if hn else ('1', '2', '3', '4', '9')
        if any(r.get('data_kubun') not in allowed for r in items) or len(signatures) != 1:
            stats['conflicting_or_deleted_ids'] += 1
        else:
            out[k] = items[0]
            stats['usable_ids'] += 1
        stats['extra_records'] += len(items) - 1
    return out, dict(stats)


def link(horses, um_rows, sk_rows, hn_rows):
    um, us = index(um_rows); sk, ss = index(sk_rows); hn, hs = index(hn_rows, hn=True)
    blocked = ({r.get('ketto_toroku_bango') for r in um_rows} - set(um)) | ({r.get('ketto_toroku_bango') for r in sk_rows} - set(sk))
    out = {}; counts = Counter(); reasons = Counter()
    for horse in sorted(horses):
        a, b = um.get(horse), sk.get(horse)
        if horse in blocked:
            a = b = None
        if a and b:
            counts['UM_and_SK_overlap'] += 1
            if any(a.get(k) != b.get(k) for k in ('ketto_joho_01a', 'ketto_joho_02a', 'ketto_joho_05a')):
                counts['UM_SK_conflicting_horses'] += 1
                a = b = None
        row = a or b
        result = {'sire': None, 'damsire': None}
        if row:
            sire = stable_id(row.get('ketto_joho_01a'))
            damsire = stable_id(row.get('ketto_joho_05a'))
            mother = stable_id(row.get('ketto_joho_02a'))
            mrow = hn.get(mother)
            if mrow and stable_id(mrow.get('ketto_joho_01a')):
                counts['maternal_path_checkable'] += 1
                if stable_id(mrow['ketto_joho_01a']) != damsire:
                    reasons['maternal_path_conflict'] += 1
                    damsire = None
                else:
                    counts['maternal_path_agrees'] += 1
            for role, value in (('sire', sire), ('damsire', damsire)):
                if value and value in hn:
                    result[role] = value
                    counts[role + '_HN_linked'] += 1
                else:
                    reasons[role + '_unknown_or_HN_missing'] += 1
        else:
            reasons['horse_master_missing_or_conflicting'] += 1
        out[horse] = result
    return out, {'target_horses': len(horses), 'counts': dict(counts), 'reasons': dict(reasons),
                 'UM': us, 'SK': ss, 'HN': hs,
                 'identity_basis': 'current_master_static_ancestry_only;no_cumulative_performance',
                 'historical_master_revision_asof': 'unverified',
                 'duplicate_breeding_ids_for_same_horse': 'not_name_merged;exact_declared_breeding_id_groups'}


def collect(directory, horses, session):
    schema = c.read(directory / 'schema.json')
    wanted = {'jvd_um': FIELDS, 'jvd_sk': FIELDS, 'jvd_hn': HN_FIELDS}
    for table, fields in wanted.items():
        names = {r['name'] for r in schema.get(table, ()) if r['type_is_builtin']}
        if not set(fields) <= names:
            raise MLFailure('invalid_schema')
    target = directory / 'identity-source'; target.mkdir(exist_ok=True)
    tables = {}; manifest = []
    import time
    began = time.monotonic()
    for table in ('jvd_um', 'jvd_sk', 'jvd_hn'):
        if table == 'jvd_hn':
            ids = sorted({v for rows in tables.values() for r in rows for k in FIELDS[3:] if (v := stable_id(r.get(k)))})
        else:
            ids = sorted(horses)
        if len(ids) > 150000:
            raise MLFailure('input_limit')
        all_rows = []
        for start in range(0, len(ids), 1000):
            if time.monotonic() - began > 300:
                raise MLFailure('query_timeout')
            path = target / f'{table}_{start // 1000:03d}.json'
            if path.exists():
                value = c.read(path)
                if value['ids'] != ids[start:start+1000]:
                    raise MLFailure('artifact_invalid')
            else:
                sql, args = query(table, ids[start:start+1000])
                with session() as db:
                    rows = db.aggregate('pedigree.identity', sql, args, max_rows=3000)
                value = {'ids': ids[start:start+1000], 'observed_at': c.clock().isoformat(), 'rows': rows}
                c.write(path, value)
            all_rows.extend(value['rows'])
            manifest.append({'file': str(path.relative_to(directory)), 'sha256': c.sha(path), 'rows': len(value['rows'])})
        tables[table] = all_rows
    lineage, summary = link(horses, tables['jvd_um'], tables['jvd_sk'], tables['jvd_hn'])
    c.write(directory / 'lineage.local.json', lineage)
    c.write(directory / 'identity_manifest.json', {'parts': manifest, 'lineage_sha256': c.sha(directory / 'lineage.local.json'), 'summary': summary})
    return lineage, summary
