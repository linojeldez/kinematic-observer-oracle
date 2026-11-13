"""Calibracion sobre scores nulos, sin consultar responsividad ni etiquetas de respuesta.

Convencion: posterior = logistic(gain * score + intercept).
intercept es TOTAL: incluye el prior; no se debe sumarle logit(prior) otra vez.
Todas las funciones matematicas aceptan scores antes de gain/intercept.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
from typing import Callable
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET_FPR = .05
TARGET_MEAN = .173
FIXED_GAIN = 12.048
TOL_FPR = .01
TOL_MEAN = .015


def logistic(z):
    a = np.asarray(z, dtype=float)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1 / (1 + np.exp(-a[pos]))
    ez = np.exp(a[~pos]); out[~pos] = ez / (1 + ez)
    return out


def finite_scores(scores):
    raw = np.asarray(scores, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if not len(finite): raise ValueError('No finite null scores')
    return finite, dict(n_total=len(raw), n_finite=len(finite), n_nonfinite=len(raw)-len(finite))


def score_diagnostics(scores, target_fpr=TARGET_FPR):
    s, n = finite_scores(scores)
    values, counts = np.unique(s, return_counts=True)
    q = float(np.quantile(s, 1-target_fpr, method='higher'))
    gt, eq = float(np.mean(s > q)), float(np.mean(s == q))
    # A one-observation rounding error is not a structural tie obstruction.
    tolerance = 1 / len(s) + 1e-12
    return dict(**n, n_unique=len(values), binary_or_constant=bool(len(values) <= 2),
                maximum_atom=float(counts.max()/len(s)), minimum=float(s.min()), maximum=float(s.max()),
                mean=float(s.mean()), sd=float(s.std()), q95=q,
                quantile_method='higher', fpr_strict_at_quantile=gt, quantile_tie_mass=eq,
                fpr_if_all_ties_positive=gt+eq,
                target_obstructed_by_ties=bool(abs(gt-target_fpr)>tolerance and eq>tolerance),
                empirical_resolution=1/len(s))


def metrics(scores, gain, intercept):
    s, n = finite_scores(scores)
    z = gain*s+intercept
    p = logistic(z)
    return dict(**n, fpr=float(np.mean(z > 0)), mean_attention=float(np.mean(p)),
                posterior_ties_at_half=float(np.mean(z == 0)),
                posterior_min=float(p.min()), posterior_max=float(p.max()),
                gain=float(gain), intercept=float(intercept))


def mean_intercept(scores, gain, target_mean=TARGET_MEAN, mean_tolerance=1e-10, max_iter=120):
    s, _ = finite_scores(scores)
    if not gain > 0 or not 0 < target_mean < 1: raise ValueError('Invalid gain or mean target')
    u = gain*s
    lo, hi = -float(u.max())-45., -float(u.min())+45.
    for it in range(max_iter):
        mid = (lo+hi)/2
        got = float(np.mean(logistic(u+mid)))
        if abs(got-target_mean) <= mean_tolerance: return mid, it+1
        if got < target_mean: lo=mid
        else: hi=mid
    return (lo+hi)/2, max_iter


def calibrate_scores(scores, scheme, gain=FIXED_GAIN, target_fpr=TARGET_FPR,
                     target_mean=TARGET_MEAN, gain_bounds=(.01, 1e4)):
    """Devuelve parametros y errores en el MISMO array; no implica validacion cerrada.

    scheme: fpr (gain fijo), mean (gain fijo), joint (gain+intercept).
    El FPR es estricto posterior>.5; no se aleatorizan empates.
    """
    s, _ = finite_scores(scores)
    if scheme not in ('fpr', 'mean', 'joint'): raise ValueError(scheme)
    if not gain > 0 or not 0 < target_fpr < 1: raise ValueError('Invalid gain/FPR target')
    diagnostics = score_diagnostics(s, target_fpr)
    q = diagnostics['q95']
    status, iterations = 'offline_calibrated', 0
    bracket = None
    if scheme == 'mean':
        intercept, iterations = mean_intercept(s, gain, target_mean)
    elif scheme == 'fpr':
        intercept = -gain*q
        if diagnostics['target_obstructed_by_ties']: status='fpr_unattainable_due_to_ties'
    else:
        lo, hi = map(float, gain_bounds)
        if lo <= 0 or hi <= lo: raise ValueError('Invalid positive gain bounds')
        def objective(alpha): return float(logistic(alpha*(s-q)).mean())-target_mean
        flo, fhi = objective(lo), objective(hi)
        bracket=dict(gain_low=lo, gain_high=hi, mean_low=flo+target_mean, mean_high=fhi+target_mean)
        if diagnostics['binary_or_constant'] or diagnostics['target_obstructed_by_ties']:
            status='joint_fpr_unattainable_due_to_ties'
        if flo*fhi > 0:
            gain=lo if abs(flo)<abs(fhi) else hi
            status='joint_target_outside_gain_bracket'
        else:
            # Bisection requires a sign bracket, not an unproved monotonicity claim.
            for iterations in range(1, 121):
                mid=math.sqrt(lo*hi); fm=objective(mid)
                if abs(fm)<1e-10: lo=hi=mid; break
                if flo*fm <= 0: hi=mid; fhi=fm
                else: lo=mid; flo=fm
            gain=math.sqrt(lo*hi)
        intercept=-gain*q
    got=metrics(s,gain,intercept)
    errors=dict(fpr=got['fpr']-target_fpr, mean_attention=got['mean_attention']-target_mean)
    objectives=(('fpr',) if scheme=='fpr' else ('mean_attention',) if scheme=='mean' else ('fpr','mean_attention'))
    return dict(scheme=scheme,gain=float(gain),intercept=float(intercept),status=status,
                targets=dict(fpr=target_fpr,mean_attention=target_mean),objectives=list(objectives),
                achieved_offline=got,errors_offline=errors,score_diagnostics=diagnostics,
                gain_bracket=bracket,solver_iterations=iterations,
                posterior_convention='logistic(gain * score + total_intercept)',
                no_responsivity_used=True)


def replicate_metrics(score_arrays, gain, intercept, seed=20261112, bootstrap=10000):
    rows=[metrics(s,gain,intercept) for s in score_arrays]
    if not rows: raise ValueError('No replicate scores')
    v=np.array([[r['fpr'],r['mean_attention']] for r in rows])
    rng=np.random.default_rng(seed)
    b=rng.integers(0,len(rows),(bootstrap,len(rows)))
    means=v[b].mean(1)
    return dict(n_replicates=len(rows),replicates=rows,
                fpr=float(v[:,0].mean()),mean_attention=float(v[:,1].mean()),
                ci95_fpr=np.percentile(means[:,0],[2.5,97.5]).tolist(),
                ci95_mean_attention=np.percentile(means[:,1],[2.5,97.5]).tolist(),
                bootstrap=bootstrap,bootstrap_seed=seed,unit='independent seed replicate')


def target_status(summary, scheme, target_fpr=TARGET_FPR,target_mean=TARGET_MEAN,
                  tol_fpr=TOL_FPR,tol_mean=TOL_MEAN):
    targets={'fpr':(target_fpr,tol_fpr),'mean_attention':(target_mean,tol_mean)}
    use=('fpr',) if scheme=='fpr' else ('mean_attention',) if scheme=='mean' else ('fpr','mean_attention')
    result={}
    for name in use:
        target,tol=targets[name]; low,high=summary['ci95_'+name]
        result[name]=dict(target=target,tolerance=tol,mean=summary[name],ci95=[low,high],
                          point_in_tolerance=abs(summary[name]-target)<=tol,
                          ci95_within_tolerance=bool(low>=target-tol and high<=target+tol))
    return dict(metrics=result,all_required_points_in_tolerance=all(r['point_in_tolerance'] for r in result.values()),
                all_required_ci95_within_tolerance=all(r['ci95_within_tolerance'] for r in result.values()))


def closed_loop_fit(initial, evaluate: Callable, seeds, tau, max_iterations=4,
                    damping=.7, target_fpr=TARGET_FPR,target_mean=TARGET_MEAN,evaluate_batch=None,
                    train_tol_fpr=.001,train_tol_mean=.002):
    """evaluate(tau,gain,intercept,seed)->scores ARRAY, cada llamada simula WiSt nulo.

    Cada iteracion evalua parametros ACTUALES y propone ajustes sobre sus scores.
    Se devuelve el ultimo conjunto realmente evaluado, nunca una propuesta sin correr.
    No usa semillas de validacion ni respuesta.
    """
    current=dict(initial)
    scheme=current['scheme']; history=[]; final_arrays=[]
    for iteration in range(1,max_iterations+1):
        arrays=(evaluate_batch(tau,current['gain'],current['intercept'],seeds) if evaluate_batch
                else [evaluate(tau,current['gain'],current['intercept'],seed) for seed in seeds])
        summary=replicate_metrics(arrays,current['gain'],current['intercept'],bootstrap=2000)
        achieved=target_status(summary,scheme,target_fpr,target_mean,train_tol_fpr,train_tol_mean)
        pooled=np.concatenate([finite_scores(s)[0] for s in arrays])
        proposal=calibrate_scores(pooled,scheme,gain=FIXED_GAIN,
                                  target_fpr=target_fpr,target_mean=target_mean)
        history.append(dict(iteration=iteration,parameters=dict(gain=current['gain'],intercept=current['intercept']),
                            train_summary=summary,train_status=achieved,proposal=proposal))
        final_arrays=arrays
        if achieved['all_required_points_in_tolerance']: break
        if iteration<max_iterations:
            next_gain=math.exp((1-damping)*math.log(current['gain'])+damping*math.log(proposal['gain']))
            next_b=(1-damping)*current['intercept']+damping*proposal['intercept']
            if scheme!='joint': next_gain=FIXED_GAIN
            current={**current,'gain':next_gain,'intercept':next_b}
    return dict(scheme=scheme,tau=tau,gain=current['gain'],intercept=current['intercept'],
                stat='full',status='train_point_matched' if history[-1]['train_status']['all_required_points_in_tolerance'] else 'train_tolerance_not_reached',
                train_seeds=list(seeds),iterations=len(history),maximum_iterations=max_iterations,
                damping=damping,history=history,targets=dict(fpr=target_fpr,mean_attention=target_mean),
                training_tolerances=dict(fpr=train_tol_fpr,mean_attention=train_tol_mean),
                no_responsivity_used=True),final_arrays


def smoke_checks():
    rng=np.random.default_rng(3491); s=rng.normal(-.7,.4,20000)
    rows=[calibrate_scores(s,k) for k in ('fpr','mean','joint')]
    assert abs(rows[0]['errors_offline']['fpr'])<=1/len(s)+1e-12
    assert abs(rows[1]['errors_offline']['mean_attention'])<1e-8
    assert abs(rows[2]['errors_offline']['fpr'])<=1/len(s)+1e-12
    assert abs(rows[2]['errors_offline']['mean_attention'])<1e-8
    binary=calibrate_scores(np.r_[np.zeros(16000),np.ones(4000)],'joint')
    constant=calibrate_scores(np.ones(1000),'joint')
    assert binary['status']!='offline_calibrated'
    assert constant['status']!='offline_calibrated'
    assert np.array_equal(logistic([-1000,0,1000]),[0,.5,1])
    return dict(status='passed',continuous=rows,binary=binary,constant=constant,
                tests=['continuous target errors','structural binary ties recorded','constant scores unattainable',
                       'stable logistic tails'],no_simulation_run=True)


def save_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Output must remain inside prioridad3')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--scores',type=Path)
    parser.add_argument('--field',default='scores')
    parser.add_argument('--scheme',choices=('fpr','mean','joint'),default='joint')
    parser.add_argument('--output',type=Path,default=ROOT/'calibracion/calibration_math_smoke.json')
    args=parser.parse_args()
    if args.smoke: result=smoke_checks()
    else:
        if not args.scores: parser.error('--scores or --smoke required')
        source=np.load(args.scores)
        result=calibrate_scores(source[args.field],args.scheme)
    save_json(args.output,result)
    print(args.output)
