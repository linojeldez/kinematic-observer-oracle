"""Calibra exclusivamente en nulos; congela parametros antes de validar y de medir R.

Requiere motor_controles.py en el mismo directorio. Maximo cuatro trabajadores.
Los caches guardan scores de cada corrida; ningun resultado de respuesta se lee.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from calibrar_controles import (ROOT, FIXED_GAIN, TARGET_FPR, TARGET_MEAN, logistic,
    calibrate_scores, closed_loop_fit, replicate_metrics, target_status, save_json)

TRAIN_SEEDS=list(range(900000,900004))
VALIDATION_SEEDS=list(range(911000,911008))
TAUS=(0.,1.)
BURN=12.
DURATION=12.
SAMPLE_EVERY=.1
MAX_ITERATIONS=4
WORKERS=4
CACHE=ROOT/'calibracion/corridas_nulas'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def serializable(x):
    if isinstance(x,dict): return {str(k):serializable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [serializable(v) for v in x]
    if isinstance(x,np.ndarray): return serializable(x.tolist())
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if math.isfinite(x) else None
    return x


def run_job(job):
    import motor_controles as motor
    if sha(Path(motor.__file__))!=job['motor_sha256']: raise RuntimeError('Motor changed after job declaration')
    key=hashlib.sha256(json.dumps(serializable(job),sort_keys=True).encode()).hexdigest()[:20]
    npz=CACHE/(key+'.npz'); meta=CACHE/(key+'.json')
    if npz.exists() and meta.exists():
        r=np.load(npz)
        return dict(scores=r['scores'],metadata=json.loads(meta.read_text()),cache=str(npz.relative_to(ROOT)))
    t0=time.time()
    if job['kind']=='replay':
        r=motor.run_replay_source(job['seed'],burn=BURN,duration=DURATION,sample_every=None)
        s=np.asarray(r['full'],float)
        np.savez_compressed(npz,scores=s,times=r['times'],evaluation_mask=r['evaluation_mask'],dt=r['dt'])
        summary=dict(job=job,params=r['params'],seconds=time.time()-t0,shape=s.shape)
    else:
        r=motor.run_null_scores(job['seed'],burn=BURN,duration=DURATION,
                                variant=job['variant'],sample_every=SAMPLE_EVERY)
        s=np.asarray(r['edge_scores'],float)
        error=float(np.max(np.abs(np.asarray(r['edge_posterior'])-
                          logistic(job['variant']['gain']*s+job['variant']['intercept']))))
        if error>1e-10: raise RuntimeError('Posterior and affine score disagree: '+str(error))
        np.savez_compressed(npz,scores=s,times=r['times'])
        summary=dict(job=job,params=r['params'],seconds=time.time()-t0,shape=s.shape,
                     posterior_formula_max_error=error)
    save_json(meta,serializable(summary))
    return dict(scores=s,metadata=serializable(summary),cache=str(npz.relative_to(ROOT)))


def ema(stream,tau,dt):
    if tau==0: return np.asarray(stream,float).copy()
    a=math.exp(-dt/tau); state=np.zeros(stream.shape[1]); out=np.empty_like(stream,dtype=float)
    for i,s in enumerate(stream): state=a*state+(1-a)*s; out[i]=state
    return out


def main():
    import motor_controles as motor
    t0=time.time(); CACHE.mkdir(parents=True,exist_ok=True)
    motor_hash=sha(Path(motor.__file__))
    declaration=dict(date=datetime.now(timezone.utc).isoformat(),targets=dict(fpr=TARGET_FPR,mean_attention=TARGET_MEAN),
          fixed_gain=FIXED_GAIN,taus=list(TAUS),train_seeds=TRAIN_SEEDS,validation_seeds=VALIDATION_SEEDS,
          burn=BURN,duration=DURATION,sample_every=SAMPLE_EVERY,maximum_iterations=MAX_ITERATIONS,
          workers=WORKERS,motor_sha256=motor_hash,calibration_sha256=sha(Path(__file__).with_name('calibrar_controles.py')),
          fit_estimand='directed non-self edges, equal weight per sampled edge',warm_estimand='emitter scores in ViSt',
          train_tolerances=dict(fpr=.001,mean_attention=.002),validation_tolerances=dict(fpr=.01,mean_attention=.015),
          no_responsivity_used=True,posterior='logistic(gain * raw_score + total_intercept)',
          selection='fit only 4 training seeds; freeze before 8 validation seeds; no validation refit')
    save_json(ROOT/'calibracion/diseno_calibracion.json',declaration)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        replay=list(pool.map(run_job,[dict(kind='replay',seed=s,motor_sha256=motor_hash) for s in TRAIN_SEEDS]))
        warm={tau:[] for tau in TAUS}
        for r in replay:
            arrays=np.load(ROOT/r['cache']); dt=float(arrays['dt'])
            sample=np.flatnonzero(arrays['evaluation_mask'])[::max(1,round(SAMPLE_EVERY/dt))]
            for tau in TAUS: warm[tau].append(ema(arrays['scores'],tau,dt)[sample])
        training=[]
        jobs_seen=[]
        def batch(tau,gain,intercept,seeds):
            variant=dict(stat='full',tau=float(tau),gain=float(gain),intercept=float(intercept))
            jobs=[dict(kind='closed_null',seed=s,variant=variant,motor_sha256=motor_hash,
                       estimator='directed_nonself_edges') for s in seeds]
            rr=list(pool.map(run_job,jobs)); jobs_seen.extend([r['cache'] for r in rr])
            return [r['scores'] for r in rr]
        for tau in TAUS:
            pooled=np.concatenate([s.reshape(-1) for s in warm[tau]])
            for scheme in ('fpr','mean','joint'):
                initial=calibrate_scores(pooled,scheme,gain=FIXED_GAIN)
                fitted,_=closed_loop_fit(initial,lambda *args:None,TRAIN_SEEDS,tau,
                            max_iterations=MAX_ITERATIONS,evaluate_batch=batch)
                fitted['initial_offline']=initial
                fitted['name']=f'matched_{scheme}_tau{tau:g}'
                fitted['estimand']='directed_nonself_edges'
                training.append(fitted)
                save_json(ROOT/'calibracion/entrenamiento_parcial.json',serializable(dict(declaration=declaration,variants=training)))
                print('TRAIN',fitted['name'],'gain',fitted['gain'],'b',fitted['intercept'],
                      fitted['status'],'iterations',fitted['iterations'],flush=True)
        frozen=dict(declaration=declaration,variants=[{k:v for k,v in r.items() if k not in ('history','initial_offline')} for r in training],
                    frozen_utc=datetime.now(timezone.utc).isoformat(),before_validation=True,before_responsivity=True)
        freeze_path=ROOT/'calibracion/parametros_congelados_antes_validacion.json'
        save_json(freeze_path,serializable(frozen))
        frozen_hash=sha(freeze_path)
        validated=[]
        for fitted in training:
            arrays=batch(fitted['tau'],fitted['gain'],fitted['intercept'],VALIDATION_SEEDS)
            summary=replicate_metrics(arrays,fitted['gain'],fitted['intercept'],bootstrap=10000)
            status=target_status(summary,fitted['scheme'])
            variant={**fitted,'validation_summary':summary,'validation_status':status,
                     'validation_seeds':VALIDATION_SEEDS,
                     'validation_equivalence':status['all_required_ci95_within_tolerance']}
            validated.append(variant)
            save_json(ROOT/'calibracion/calibracion_validacion_parcial.json',serializable(dict(declaration=declaration,variants=validated)))
            print('VALIDATION',variant['name'],'FPR',summary['fpr'],summary['ci95_fpr'],
                  'mean',summary['mean_attention'],summary['ci95_mean_attention'],
                  'equivalent',variant['validation_equivalence'],flush=True)
    result=dict(declaration=declaration,variants=validated,all_validation_targets_met=all(r['validation_equivalence'] for r in validated),
                frozen_parameters_sha256=frozen_hash,seconds=time.time()-t0,
                null_cache_files=sorted(set(jobs_seen)),warm_replay_files=[r['cache'] for r in replay],
                no_responsivity_used=True,validation_used_for_parameter_refit=False)
    save_json(ROOT/'calibracion/calibracion_final.json',serializable(result))
    configs=[]
    for r in validated:
        configs.append({k:r[k] for k in ('name','stat','tau','gain','intercept','scheme','estimand','validation_seeds','validation_equivalence')})
    save_json(ROOT/'calibracion/variantes_para_respuesta.json',dict(variants=configs,
          frozen_parameters_sha256=frozen_hash,calibration_file='calibracion/calibracion_final.json',
          estimator='directed_nonself_edges',validation_seeds=VALIDATION_SEEDS,motor_sha256=motor_hash,
          parameters_frozen_before_validation_and_responsivity=True,
          warning='Unmatched variants remain sensitivity analyses; do not claim equal FPR/attention'))
    print('DONE',time.time()-t0,'seconds',flush=True)


if __name__=='__main__': main()
