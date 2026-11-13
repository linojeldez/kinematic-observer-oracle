"""Diagnostico pareado posterior a congelar parametros; nunca reajusta parametros."""
from pathlib import Path
import hashlib
import json
import numpy as np
from calibrar_controles import ROOT, TARGET_FPR, TARGET_MEAN, TOL_FPR, TOL_MEAN, save_json


def f(v): return f'{v:.5f}'
def interval(v): return '['+'; '.join(f(q) for q in v)+']'


def main():
    source=ROOT/'calibracion/calibracion_final.json'
    x=json.loads(source.read_text())
    if x['declaration'].get('fit_estimand')!='directed non-self edges, equal weight per sampled edge':
        raise ValueError('Final directed-edge calibration required')
    rng=np.random.default_rng(202611123)
    paired={}
    for scheme in ('fpr','mean','joint'):
        a=next(r for r in x['variants'] if r['scheme']==scheme and r['tau']==0)
        b=next(r for r in x['variants'] if r['scheme']==scheme and r['tau']==1)
        if a['validation_seeds']!=b['validation_seeds']: raise ValueError('Unpaired validation seeds')
        aa=a['validation_summary']['replicates'];bb=b['validation_summary']['replicates']
        C=rng.integers(0,len(aa),(10000,len(aa)))
        rr={}
        for metric,tolerance in [('fpr',TOL_FPR),('mean_attention',TOL_MEAN)]:
            delta=np.array([rb[metric]-ra[metric] for ra,rb in zip(aa,bb)])
            ci=np.percentile(delta[C].mean(1),[2.5,97.5]).tolist()
            rr[metric]=dict(delta_tau1_minus_tau0=float(delta.mean()),ci95=ci,
                tolerance=tolerance,ci95_inside_equivalence=bool(ci[0]>=-tolerance and ci[1]<=tolerance),
                per_seed_delta=delta.tolist())
        required=['fpr'] if scheme=='fpr' else ['mean_attention'] if scheme=='mean' else ['fpr','mean_attention']
        paired[scheme]=dict(n_pairs=len(aa),seeds=a['validation_seeds'],metrics=rr,
                     required_metrics=required,paired_equivalence=all(rr[m]['ci95_inside_equivalence'] for m in required),
                     both_tau_target_equivalence=bool(a['validation_equivalence'] and b['validation_equivalence']))
    result=dict(estimator='directed_nonself_edges',pairs=paired,bootstrap=10000,seed=202611123,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),no_parameters_modified=True,
        frozen_parameters_sha256=x['frozen_parameters_sha256'])
    save_json(ROOT/'calibracion/validacion_pareada.json',result)
    config_path=ROOT/'calibracion/variantes_para_respuesta.json'
    config=json.loads(config_path.read_text())
    for v in config['variants']:
        v['paired_equivalence']=paired[v['scheme']]['paired_equivalence']
        v['both_tau_target_equivalence']=paired[v['scheme']]['both_tau_target_equivalence']
        v['full_matching_claim_allowed']=bool(v['paired_equivalence'] and v['both_tau_target_equivalence'])
    config['paired_validation_file']='calibracion/validacion_pareada.json'
    save_json(config_path,config)
    L=['# Calibración nula de memoria y ganancia','',
       'Posterior = logistic(α·score + b), con b total, incluido el prior. El score es el log-cociente completo instantáneo o su EMA, antes de multiplicar por α. No se usó responsividad para seleccionar parámetros. Los objetivos son operacionales: FPR = 0,05 para posterior > 0,5 y media del posterior = 0,173; el segundo coincide con el prior del banco y no estima prevalencia biológica.','',
       'El ajuste y la validación ponderan las aristas dirigidas hacia los k vecinos, excluido el propio individuo, en la topología que consume el score antes de la acción. La media del posterior así medida es una media de atención inferida por arista; no es el peso normalizado final de la ecuación social. Este último se registra por el motor para diagnóstico separado.','',
       '## Diseño y separación de datos','',
       '- N=250, k=15, ruido de referencia, sin líder ni falsos estados informados; 12 s de asentamiento y 12 s de evaluación a intervalos de 0,1 s.',
       '- Inicialización en scores de ViSt de entrenamiento, filtrados a resolución original antes de reducir el muestreo; ajuste definitivo sobre nulos WiSt con realimentación del observador.',
       '- Entrenamiento: semillas 900000–900003. Máximo de cuatro iteraciones y amortiguación 0,7; parada por error puntual FPR ≤ 0,001 y/o media ≤ 0,002, según objetivo. Se conserva el último parámetro realmente evaluado, nunca una propuesta sin simular.',
       '- Ganancia fija 12,048 para esquemas FPR y media. El esquema conjunto ajusta α y b con α entre 0,01 y 10000, usando el cuantil 95 y la media logística; no se aleatorizan empates.',
       '- Los parámetros se congelaron antes de validar y antes de medir respuesta. Validación final independiente: semillas 911000–911007; no se reajustó usando sus resultados. La corrida anterior por emisores está archivada en `calibracion/preliminar_emisores/` y no aporta variantes de respuesta. Sus semillas de validación 910000–910007 no se reutilizaron en la validación final.',
       '- IC95 por bootstrap de ocho réplicas independientes, no por millones de aristas/fotogramas. Equivalencia operacional requiere que el IC95 completo esté dentro de objetivo ±0,01 FPR y/o ±0,015 media. Además se prueba la diferencia pareada entre τ=1 y τ=0 con las mismas tolerancias.','',
       '## Parámetros congelados y validación','',
       '| Esquema | τ (s) | α | b | Iteraciones / estado entrenamiento | FPR [IC95] | Media [IC95] | IC al objetivo |',
       '|---|---:|---:|---:|---|---:|---:|---|']
    for r in x['variants']:
        v=r['validation_summary']
        L.append(f"| {r['scheme']} | {r['tau']:g} | {f(r['gain'])} | {f(r['intercept'])} | {r['iterations']} / {r['status']} | {f(v['fpr'])} {interval(v['ci95_fpr'])} | {f(v['mean_attention'])} {interval(v['ci95_mean_attention'])} | {'sí' if r['validation_equivalence'] else 'no'} |")
    L+=['','«IC al objetivo» exige sólo la magnitud ajustada en cada esquema: FPR en fpr, media en mean y ambas en joint. Las otras magnitudes se muestran y no se supone que estén igualadas.','',
        '## Diferencia pareada τ=1 menos τ=0','',
        '| Esquema | Δ FPR [IC95] | Δ media [IC95] | Equivalencia pareada requerida | Ambos τ al objetivo |',
        '|---|---:|---:|---|---|']
    for scheme,p in paired.items():
        a=p['metrics']['fpr'];b=p['metrics']['mean_attention']
        L.append(f"| {scheme} | {f(a['delta_tau1_minus_tau0'])} {interval(a['ci95'])} | {f(b['delta_tau1_minus_tau0'])} {interval(b['ci95'])} | {'sí' if p['paired_equivalence'] else 'no'} | {'sí' if p['both_tau_target_equivalence'] else 'no'} |")
    allowed=[s for s,p in paired.items() if p['paired_equivalence'] and p['both_tau_target_equivalence']]
    failed=[s for s in paired if s not in allowed]
    L+=['',f"Esquemas que permiten la afirmación de igualación según los criterios fijados: {', '.join(allowed) or 'ninguno'}.",
        f"Esquemas restantes: {', '.join(failed) or 'ninguno'}. Si se miden sus respuestas, son análisis de sensibilidad y deben acompañarse de su desajuste; no autorizan una afirmación de igualdad de FPR/atención.",
        '', 'La equivalencia en el vuelo nulo no implica que la distribución de pesos permanezca igual durante una maniobra informada. Esa alteración es parte del mecanismo y debe registrarse al analizar las respuestas, sin recalibrar los parámetros con ellas. Tampoco se equipara la capacidad sensorial o la información temporal disponible al observador.',
        '', '## Evidencia y límites','',
        '- `calibracion/diseno_calibracion.json`: objetivos, semillas, versiones y convenciones.',
        '- `calibracion/parametros_congelados_antes_validacion.json`: parámetros realmente evaluados y congelados antes de validación/R.',
        '- `calibracion/calibracion_final.json`: historial completo de entrenamiento, ajustes fuera de línea, errores, empates y validación por réplica.',
        '- `calibracion/validacion_pareada.json`: diferencias pareadas y sus intervalos.',
        '- `calibracion/variantes_para_respuesta.json`: configuraciones y etiquetas de igualdad permitida.',
        '- `calibracion/corridas_nulas/*.npz` y metadatos JSON: scores, tiempos, fuentes y checks de fórmula; los caches finales se identifican por estimador y hash del motor.',
        '- `calibracion/calibration_math_smoke.json`: verificación de objetivos continuos, empates binarios, scores constantes y estabilidad logística; no son resultados simulados de la bandada.',
        '',f"Hash del motor final: `{x['declaration']['motor_sha256']}`.",
        f"SHA-256 de los parámetros congelados: `{x['frozen_parameters_sha256']}`.",
        '', 'El presupuesto de cuatro iteraciones es un criterio de cierre, no una garantía de convergencia. Un estado no alcanzado o un IC fuera del margen permanece reportado, sin ajustar con validación ni buscar una comparación favorable de responsividad.','']
    (ROOT/'CALIBRACION_NULA.md').write_text('\n'.join(L))
    print(ROOT/'CALIBRACION_NULA.md')


if __name__=='__main__':main()
