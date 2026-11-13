"""Campaña nueva con semillas retenidas; sólo escribe dentro de prioridad3."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from motor_controles import run_trial, DEFAULT_INTERCEPT

ROOT=Path(__file__).resolve().parents[1]
GAIN=12.048
HORIZON=15.8
ENGINE_FILES=[ROOT/'fuentes/flocking.py',ROOT/'scripts/motor_controles.py']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def variant(name,stat='full',tau=0.,gain=GAIN,intercept=DEFAULT_INTERCEPT,**extra):
    return dict(name=name,stat=stat,tau=float(tau),gain=float(gain),intercept=float(intercept),**extra)

def build_jobs(phase):
    variants=[]
    if phase=='fixed':
        for tau in [0.,.25,1.,4.]:
            for g,label in [(1.,'1'),(GAIN,'g0'),(48.,'48')]:
                core=tau in (0.,1.) and label=='g0'
                variants.append(variant(f'full_t{tau:g}_a{label}',tau=tau,gain=g,
                                        scheme='fixed',replicates=8 if core else 4,primary=core))
        for tau in [0.,1.]:
            variants.append(variant(f'binary_t{tau:g}_ag0',stat='binary',tau=tau,
                                    scheme='binary_fixed',replicates=8,primary=False))
    else:
        path=ROOT/'calibracion/variantes_para_respuesta.json'
        c=json.loads(path.read_text())
        assert c.get('estimator')=='directed_nonself_edges', 'Calibración final por aristas no disponible'
        assert c.get('parameters_frozen_before_validation_and_responsivity') is True
        assert c.get('validation_seeds')==list(range(911000,911008))
        assert c.get('motor_sha256')==sha(ROOT/'scripts/motor_controles.py')
        assert len(c['variants'])==6
        for v in c['variants']:
            v=dict(v);v.update(name=f"full_t{v['tau']:g}_{v['scheme']}",replicates=8,primary=True)
            variants.append(v)
    jobs=[]
    burncheck=ROOT/'piloto/asentamiento_memoria.json'
    if not burncheck.exists():raise RuntimeError('Falta el control nulo de asentamiento predeclarado')
    burninfo=json.loads(burncheck.read_text())
    long_burn=bool(burninfo['use_common_burn24_for_tau4_block_and_references'])
    for protocol in ['forzado','maniobra']:
        for v in variants:
            burn=24. if v['tau']==4 and long_burn else 12.
            for rep in range(v['replicates']):
                for angle in range(18):
                    jobs.append(dict(phase=phase,variant=v,protocol=protocol,tauW=4.,rep=rep,
                        angle_index=angle,phi=float(np.deg2rad(10*(angle+.5))),
                        seed=920000+1000*angle+rep,burn=burn,t_R=HORIZON))
    if phase=='fixed':
        for protocol in ['forzado','maniobra']:
            for burn in ([12.,24.] if long_burn else [12.]):
                for name,stat in [('base','ViSt' if protocol=='forzado' else 'CoLd'),('oracle','oracle')]:
                    v=variant(name,stat=stat,gain=1.,scheme='reference',replicates=8,primary=False)
                    for rep in range(8):
                        for angle in range(18):
                            jobs.append(dict(phase=phase,variant=v,protocol=protocol,tauW=4.,rep=rep,
                              angle_index=angle,phi=float(np.deg2rad(10*(angle+.5))),
                              seed=920000+1000*angle+rep,burn=burn,t_R=HORIZON))
        validation=[v for v in variants if v['stat']=='full' and v['tau'] in (0.,1.) and v['gain']==GAIN]
        validation += [variant('base',stat='CoLd',gain=1.,scheme='reference',replicates=8),
                       variant('oracle',stat='oracle',gain=1.,scheme='reference',replicates=8)]
    else:
        validation=[v for v in variants if v['scheme']=='joint']
    for v in validation:
        for rep in range(8):
            for angle in range(18):
                jobs.append(dict(phase=phase,variant=v,protocol='maniobra',tauW=1.,rep=rep,
                    angle_index=angle,phi=float(np.deg2rad(10*(angle+.5))),
                    seed=920000+1000*angle+rep,burn=12.,t_R=HORIZON))
    return jobs,variants,burninfo

def filename(j):
    return (f"{j['protocol']}_w{j['tauW']:g}_b{j['burn']:g}_{j['variant']['name']}"
            f"_r{j['rep']:02}_a{j['angle_index']:02}.json")

def worker(job,hashes):
    for name,h in hashes.items():
        if sha(Path(name))!=h:raise RuntimeError('El código cambió durante la campaña')
    path=ROOT/'resultados/corridas'/filename(job)
    if path.exists():
        old=json.loads(path.read_text())
        if old['job']!=job or old['source_hashes']!=hashes:raise RuntimeError('Checkpoint incompatible')
        return {'cached':True,'seconds':0,'file':path.name}
    result=run_trial(job['variant'],job['seed'],job['phi'],protocol=job['protocol'],
         tauW=job['tauW'],t_R=job['t_R'],burn=job['burn'],diagnostics=True,
         diagnostics_every=10,collect_scores=False,score_every=.1)
    proj=result.pop('proj')
    step=20
    result['projection_mean_every_0_2s']=[float(proj[i:i+step].mean()) for i in range(0,len(proj),step)]
    record=clean({'job':job,'source_hashes':hashes,'result':result})
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(record,separators=(',',':'),allow_nan=False))
    temp.replace(path)
    return {'cached':False,'seconds':result['seconds'],'file':path.name}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['fixed','matched'],required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    (ROOT/'resultados/corridas').mkdir(parents=True,exist_ok=True)
    jobs,variants,burninfo=build_jobs(args.phase)
    hashes={str(p):sha(p) for p in ENGINE_FILES}
    plan={'phase':args.phase,'jobs':len(jobs),'variants':variants,'hashes':hashes,
          'burn_rule':{'use_burn24_for_tau4':burninfo['use_common_burn24_for_tau4_block_and_references']},
          'seed_rule':'920000+1000*angle+rep',
          'replicates':8,'exploratory_replicates':4,'angles':18,
          'R_resolution_seconds':.01,'diagnostic_resolution_seconds':.1,
          'predeclaration_sha256':sha(ROOT/'PREDECLARACION.md'),
          'estimand_addendum_sha256':sha(ROOT/'ADENDA_ESTIMANDO.md')}
    (ROOT/f'resultados/diseno_respuestas_{args.phase}.json').write_text(json.dumps(clean(plan),indent=2))
    if args.limit:jobs=jobs[:args.limit]
    pending=[];cached=0
    for j in jobs:
        path=ROOT/'resultados/corridas'/filename(j)
        if path.exists():
            old=json.loads(path.read_text())
            if old['job']!=j or old['source_hashes']!=hashes:raise RuntimeError('Checkpoint incompatible')
            cached+=1
        else:pending.append(j)
    started=time.time();done=0;last=0;cpu_seconds=0
    print(f"FASE {args.phase}: {len(jobs)} corridas, {cached} ya completas, {len(pending)} pendientes, {args.workers} procesos",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(worker,j,hashes) for j in pending]
        for fu in as_completed(fs):
            r=fu.result();done+=1;cpu_seconds+=r['seconds']
            elapsed=time.time()-started
            if elapsed-last>=20 or done==len(pending):
                print(f"{args.phase}: {done}/{len(pending)} nuevas; total {cached+done}/{len(jobs)}; {elapsed:.0f}s; media CPU {cpu_seconds/max(done,1):.2f}s/corrida",flush=True)
                last=elapsed
                (ROOT/f'logs/progreso_{args.phase}.json').write_text(json.dumps({'completed':cached+done,
                     'planned':len(jobs),'new':done,'elapsed':elapsed,'mean_worker_seconds':cpu_seconds/max(done,1)}))
    final={'phase':args.phase,'completed':len(jobs),'cached':cached,'new':done,'elapsed':time.time()-started,
           'cpu_seconds':cpu_seconds,'finished':True,'sources_unchanged':all(sha(Path(p))==h for p,h in hashes.items())}
    (ROOT/f'logs/resumen_respuestas_{args.phase}.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final),flush=True)

if __name__=='__main__':main()
