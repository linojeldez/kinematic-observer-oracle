"""Auditoría final desde crudos: identidades, tablas e integridad de originales."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT.parents[1]

def read(path):
    return json.loads((ROOT / path).read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    analysis = read('resultados/RESULTADOS_MEMORIA_GANANCIA.json')
    assert analysis['new_response_runs'] == 5760
    groups = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    maximum = defaultdict(float)
    identities = set()
    for path in (ROOT / 'resultados/corridas').glob('*.json'):
        z = json.loads(path.read_text())
        j, r = z['job'], z['result']
        identity = (j['protocol'], j['tauW'], j['burn'], j['variant']['name'], j['rep'], j['angle_index'])
        assert identity not in identities
        identities.add(identity)
        assert j['seed'] == 920000 + 1000 * j['angle_index'] + j['rep']
        assert 0 <= j['angle_index'] < 18
        assert abs(j['phi'] - math.radians(10 * (j['angle_index'] + .5))) < 1e-14
        assert r['params']['seed'] == j['seed']
        assert r['variant'] == j['variant']
        assert r['params']['dt'] == .01 and j['t_R'] == 15.8
        assert r['burn'] == j['burn'] == 12
        counts[j['phase']] += 1
        series = r['projection_mean_every_0_2s']
        assert len(series) == 79 and all(math.isfinite(v) for v in series)
        # Todas las cajas tienen 20 pasos; los intervalos de 4 s son 20 cajas.
        checks = {'R': statistics.fmean(series),
                  'R_first4': statistics.fmean(series[:20]),
                  'R_tail': statistics.fmean(series[-20:])}
        for name, expected in checks.items():
            err = abs(r[name] - expected)
            maximum[name] = max(maximum[name], err)
            assert err < 2e-12, (path.name, name, err)
        assert r['R'] == r['r']
        if r['diagnostics']:
            d = r['diagnostics']
            e = abs(d['total_nonself_weight_per_receiver'] - r['params']['k'] * d['nonself_pair_weight_mean'])
            maximum['edge_mean_identity'] = max(maximum['edge_mean_identity'], e)
            assert e < 2e-12
            e = abs(d['nonleader_weight_per_receiver'] - (d['total_nonself_weight_per_receiver'] - d['leader_weight_per_receiver']))
            maximum['edge_leader_partition'] = max(maximum['edge_leader_partition'], e)
            assert e < 2e-12
            e = abs(r['leader_posterior_time_above_half'] - 15.8 * d['leader_posterior_above_half'])
            maximum['duration_identity'] = max(maximum['duration_identity'], e)
            assert e < 2e-12
        for name, sha in z['source_hashes'].items():
            assert digest(Path(name)) == sha
        groups[identity[:4]][j['rep']].append(r['R'])
    assert counts == {'fixed': 3744, 'matched': 2016}, counts
    assert len(groups) == 50 and len(analysis['conditions']) == 50
    assert Counter(len(reps) for reps in groups.values()) == {8: 30, 4: 20}
    for condition in analysis['conditions']:
        key = tuple(condition[k] for k in ['protocol', 'tauW', 'burn', 'variant'])
        reps = groups[key]
        assert len(reps) == condition['n_rep']
        assert all(len(vals) == 18 for vals in reps.values())
        r = statistics.fmean(statistics.fmean(vals) for vals in reps.values())
        err = abs(r - condition['R'])
        maximum['raw_to_summary_R'] = max(maximum['raw_to_summary_R'], err)
        assert err < 2e-12
        if condition['R_oracle'] > condition['R_base']:
            expected = (condition['R'] - condition['R_base']) / (condition['R_oracle'] - condition['R_base'])
            assert abs(expected - condition['recovery']) < 1e-14
        else:
            assert condition['recovery'] is None
    summary = {(c['protocol'], c['tauW'], c['variant']): c for c in analysis['conditions']}
    for contrast in analysis['primary_contrasts']:
        suffix = 'ag0' if contrast['scheme'] == 'fixed' else contrast['scheme']
        a = summary[(contrast['protocol'], 4., 'full_t0_' + suffix)]
        b = summary[(contrast['protocol'], 4., 'full_t1_' + suffix)]
        assert abs(contrast['difference_R'] - (b['R'] - a['R'])) < 2e-12
    with (ROOT / 'resultados/contrastes_principales.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for row, contrast in zip(rows, analysis['primary_contrasts']):
        assert row['protocol'] == contrast['protocol'] and row['scheme'] == contrast['scheme']
        for name, value in contrast.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(row[name]) == value, name
        for name in ['pointwise_CI95', 'simultaneous_CI95', 'difference_recovery_CI95']:
            if contrast[name] is None:
                assert row[name+'_lo'] == row[name+'_hi'] == ''
            else:
                assert [float(row[name+'_lo']), float(row[name+'_hi'])] == contrast[name]
    cal = read('calibracion/calibracion_final.json')
    frozen = ROOT / 'calibracion/parametros_congelados_antes_validacion.json'
    assert digest(frozen) == cal['frozen_parameters_sha256']
    assert cal['no_responsivity_used'] and not cal['validation_used_for_parameter_refit']
    before = read('evidencia/originales_antes.json')
    unchanged, changed, missing = [], [], []
    for relative, old in before.items():
        path = SIMULATION / relative
        if not path.exists():
            missing.append(relative)
        elif digest(path) != old:
            changed.append({'path': relative, 'before': old, 'after': digest(path)})
        else:
            unchanged.append(relative)
    result = {'date_utc': datetime.now(timezone.utc).isoformat(), 'passed_numerical_checks': True,
        'response_runs': len(identities), 'phase_counts': dict(counts),
        'conditions': len(groups), 'maximum_absolute_errors': dict(maximum),
        'frozen_calibration_hash_unchanged': True,
        'original_files': {'baseline_count': len(before), 'unchanged_count': len(unchanged),
                           'changed': changed, 'missing': missing},
        'scope': 'Compara los 409 archivos preexistentes inventariados; no atribuye cambios concurrentes a un autor. Todas las escrituras de esta campaña apuntan a prioridad3.'}
    (ROOT / 'evidencia/VERIFICACION_FINAL.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
