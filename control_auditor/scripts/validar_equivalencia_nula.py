from pathlib import Path
import json
import csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'calibracion/calibracion_final.json').read_text())
assert data['declaration']['fit_estimand']=='directed non-self edges, equal weight per sampled edge'
assert data['declaration']['validation_seeds']==list(range(911000,911008))
rng=np.random.default_rng(20261112)
indices=rng.integers(0,8,(10000,8))
pairs=[]
for scheme in ['fpr','mean','joint']:
    values={v['tau']:v for v in data['variants'] if v['scheme']==scheme}
    assert set(values)=={0.,1.}
    a,b=values[0.],values[1.]
    assert a['validation_seeds']==b['validation_seeds']==list(range(911000,911008))
    row={'scheme':scheme,'tau0':a['validation_summary'],'tau1':b['validation_summary'],'differences':{}}
    for metric,tol in [('fpr',.01),('mean_attention',.015)]:
        x=np.array([v[metric] for v in a['validation_summary']['replicates']])
        y=np.array([v[metric] for v in b['validation_summary']['replicates']])
        d=y-x;ci=np.percentile(d[indices].mean(1),[2.5,97.5])
        row['differences'][metric]={'mean':float(d.mean()),'CI95':ci.tolist(),'tolerance':tol,
                                  'CI95_within_tolerance':bool(ci[0]>=-tol and ci[1]<=tol)}
    needed=['fpr'] if scheme=='fpr' else ['mean_attention'] if scheme=='mean' else ['fpr','mean_attention']
    row['pair_equivalence']=all(row['differences'][m]['CI95_within_tolerance'] for m in needed)
    row['each_condition_target_equivalence']=a['validation_equivalence'] and b['validation_equivalence']
    pairs.append(row)
out={'seed':20261112,'bootstrap':10000,'validation_seeds':list(range(911000,911008)),
     'estimator':'directed_nonself_edges','unit':'paired null seed','pairs':pairs,
     'all_required_pair_equivalence':all(r['pair_equivalence'] for r in pairs),
     'no_responsivity_used':True,'no_validation_refit':True}
(ROOT/'calibracion/equivalencia_pareada.json').write_text(json.dumps(out,indent=2))
rows=[]
for v in data['variants']:
    s=v['validation_summary']
    rows.append(dict(scheme=v['scheme'],tau=v['tau'],gain=v['gain'],intercept=v['intercept'],
        fpr=s['fpr'],fpr_low=s['ci95_fpr'][0],fpr_high=s['ci95_fpr'][1],
        attention=s['mean_attention'],attention_low=s['ci95_mean_attention'][0],attention_high=s['ci95_mean_attention'][1]))
with (ROOT/'calibracion/resumen_validacion.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Calibración y equivalencia nula','',
       'Parámetros ajustados sólo en cuatro semillas de entrenamiento sin individuos informados. Validación en ocho semillas nuevas, sin reajuste. Cada observación es una arista dirigida sin autoenlace; los intervalos remuestrean semillas completas, no aristas. FPR es fracción de pesos mayores de0.5, no episodios de alarma por segundo.','',
       '| Esquema | Memoria s | Ganancia | Intercepto total | FPR [IC95] | Atención media [IC95] | Objetivo requerido cumplido |',
       '|---|---:|---:|---:|---|---|---|']
for v in data['variants']:
    s=v['validation_summary']
    lines.append(f"| {v['scheme']} | {v['tau']:g} | {v['gain']:.6f} | {v['intercept']:.6f} | {s['fpr']:.5f} [{s['ci95_fpr'][0]:.5f}, {s['ci95_fpr'][1]:.5f}] | {s['mean_attention']:.5f} [{s['ci95_mean_attention'][0]:.5f}, {s['ci95_mean_attention'][1]:.5f}] | {v['validation_equivalence']} |")
lines+=['','Los esquemas FPR y mean equiparan sólo su magnitud objetivo. Sus otras magnitudes pueden diferir mucho. El esquema joint busca las dos simultáneamente. Las tolerancias son operacionales, fijadas antes de evaluar respuestas.','',
        '| Esquema | Diferencia FPR memoria−instantáneo [IC95] | Diferencia atención [IC95] | Equivalencia de la pareja en métricas exigidas |',
        '|---|---|---|---|']
for r in pairs:
    p=r['differences']['fpr'];m=r['differences']['mean_attention']
    lines.append(f"| {r['scheme']} | {p['mean']:+.5f} [{p['CI95'][0]:+.5f}, {p['CI95'][1]:+.5f}] | {m['mean']:+.5f} [{m['CI95'][0]:+.5f}, {m['CI95'][1]:+.5f}] | {r['pair_equivalence']} |")
lines+=['','El intercepto está calibrado como parte de una regla de atención; estos pesos logísticos no quedan validados automáticamente como probabilidades posteriores de información privada. Los intervalos son condicionales al entrenamiento y al motor.']
(ROOT/'CALIBRACION_Y_EQUIVALENCIA.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'all_pairs_equivalent':out['all_required_pair_equivalence'],
                  'pairs':[{k:r[k] for k in ['scheme','pair_equivalence','each_condition_target_equivalence']} for r in pairs]},indent=2))
