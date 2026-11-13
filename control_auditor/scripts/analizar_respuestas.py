"""Analiza la campaña completa; no elige parámetros usando sus respuestas."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=20261112
B=10000

def J(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(x):return [float(v) for v in np.percentile(x,[2.5,97.5])]
def jsonout(path,value):
    (ROOT/path).write_text(json.dumps(value,indent=2,allow_nan=False))
def csvout(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def group_key(j):return (j['protocol'],float(j['tauW']),float(j['burn']),j['variant']['name'])

def main():
    for phase in ['fixed','matched']:
        summary=J(f'logs/resumen_respuestas_{phase}.json')
        assert summary['finished'] and summary['sources_unchanged']
    rows=[];groups={};configs={}
    for p in sorted((ROOT/'resultados/corridas').glob('*.json')):
        x=json.loads(p.read_text());j=x['job'];r=x['result'];key=group_key(j)
        assert np.isfinite(r['R']) and np.isfinite(r['R_first4'])
        assert j['seed']==920000+1000*j['angle_index']+j['rep']
        for name,h in x['source_hashes'].items():assert sha(Path(name))==h
        groups.setdefault(key,{}).setdefault(j['rep'],{})[j['angle_index']]=r
        configs[key]=j['variant']
        row={k:j[k] for k in ['protocol','tauW','burn','rep','angle_index','phi','seed']}
        row.update(variant=j['variant']['name'],stat=j['variant']['stat'],tau=j['variant']['tau'],
                   gain=j['variant']['gain'],intercept=j['variant']['intercept'],
                   R=r['R'],R_first4=r['R_first4'],pol=r['pol'],release_time=r['release_time'],
                   seconds=r['seconds'])
        if r['diagnostics']:
            row.update({'diag_'+k:v for k,v in r['diagnostics'].items()})
        rows.append(row)
    expected=sum(J(f'resultados/diseno_respuestas_{p}.json')['jobs'] for p in ['fixed','matched'])
    assert len(rows)==expected,(len(rows),expected)
    replicate={};rep_rows=[]
    for key,reps in groups.items():
        n_expected=configs[key].get('replicates',8)
        assert set(reps)==set(range(n_expected)),(key,set(reps),n_expected)
        d={}
        for rep,angles in sorted(reps.items()):
            assert set(angles)==set(range(18)),(key,rep)
            vals=[angles[i] for i in range(18)]
            rec={'R':float(np.mean([v['R'] for v in vals])),
                 'R_first4':float(np.mean([v['R_first4'] for v in vals]))}
            if vals[0]['diagnostics']:
                rec.update({'diag_'+name:float(np.mean([v['diagnostics'][name] for v in vals]))
                    for name in vals[0]['diagnostics'] if all(v['diagnostics'][name] is not None for v in vals)})
            d[rep]=rec
            rep_rows.append(dict(protocol=key[0],tauW=key[1],burn=key[2],variant=key[3],rep=rep,**rec))
        replicate[key]=d
    conditions=[]
    for key,reps in sorted(replicate.items()):
        protocol,tauw,burn,name=key
        base=replicate[(protocol,tauw,burn,'base')]
        oracle=replicate[(protocol,tauw,burn,'oracle')]
        ids=sorted(set(reps)&set(base)&set(oracle));n=len(ids)
        x=np.array([reps[i]['R'] for i in ids]);v=np.array([base[i]['R'] for i in ids]);o=np.array([oracle[i]['R'] for i in ids])
        ix=np.random.default_rng(SEED).integers(0,n,(B,n))
        xb=x[ix].mean(1);vb=v[ix].mean(1);ob=o[ix].mean(1);den=ob-vb
        meanx=float(x.mean());meanv=float(v.mean());meano=float(o.mean())
        den_ci=ci(den)
        ratio_stable=bool(den_ci[0]>0 and np.all(den!=0))
        config=configs[key]
        row=dict(protocol=protocol,tauW=tauw,burn=burn,variant=name,stat=config['stat'],tau=config['tau'],
                 gain=config['gain'],intercept=config['intercept'],scheme=config.get('scheme'),n_rep=n,
                 R=meanx,R_CI95=ci(xb),R_base=meanv,R_oracle=meano,
                 absolute_gain=meanx-meanv,absolute_gain_CI95=ci(xb-vb),
                 oracle_advantage=meano-meanv,oracle_advantage_CI95=den_ci,
                 recovery=(meanx-meanv)/(meano-meanv) if meano>meanv else None,
                 recovery_CI95=ci((xb-vb)/den) if ratio_stable else None,
                 recovery_interval_stable=ratio_stable,
                 bootstrap_denominator_nonpositive_fraction=float(np.mean(den<=0)),
                 difference_from_oracle=meanx-meano,difference_from_oracle_CI95=ci(xb-ob))
        row['diagnostics']={k:{'mean':float(np.mean([reps[i][k] for i in ids])),
                     'CI95':ci(np.array([reps[i][k] for i in ids])[ix].mean(1))}
                     for k in reps[ids[0]] if k.startswith('diag_')}
        conditions.append(row)

    # Eight contrasts decided before response measurements.
    contrast_data=[]
    for protocol in ['forzado','maniobra']:
        for scheme in ['fixed','fpr','mean','joint']:
            suffix='ag0' if scheme=='fixed' else scheme
            a=replicate[(protocol,4.,12.,f'full_t0_{suffix}')]
            b=replicate[(protocol,4.,12.,f'full_t1_{suffix}')]
            v=replicate[(protocol,4.,12.,'base')];o=replicate[(protocol,4.,12.,'oracle')]
            ids=sorted(set(a)&set(b)&set(v)&set(o));assert ids==list(range(8))
            x=np.array([a[i]['R'] for i in ids]);y=np.array([b[i]['R'] for i in ids])
            ve=np.array([v[i]['R'] for i in ids]);oe=np.array([o[i]['R'] for i in ids])
            contrast_data.append((protocol,scheme,x,y,ve,oe))
    ix=np.random.default_rng(SEED).integers(0,8,(B,8))
    estimates=np.array([float((y-x).mean()) for _,_,x,y,_,_ in contrast_data])
    boots=np.stack([(y[ix]-x[ix]).mean(1) for _,_,x,y,_,_ in contrast_data],axis=1)
    critical=float(np.percentile(np.max(np.abs(boots-estimates[None,:]),axis=1),95))
    eq=J('calibracion/equivalencia_pareada.json')
    eqmap={r['scheme']:r for r in eq['pairs']}
    contrasts=[]
    for j,(protocol,scheme,x,y,v,o) in enumerate(contrast_data):
        d=estimates[j];den_boot=(o[ix]-v[ix]).mean(1)
        ratio_stable=bool(ci(den_boot)[0]>0 and np.all(den_boot!=0))
        matched=True if scheme=='fixed' else eqmap[scheme]['pair_equivalence'] and eqmap[scheme]['each_condition_target_equivalence']
        interval=[float(d-critical),float(d+critical)]
        contrasts.append(dict(protocol=protocol,scheme=scheme,n_rep=8,R_instant=float(x.mean()),R_memory=float(y.mean()),
            difference_R=float(d),pointwise_CI95=ci(boots[:,j]),simultaneous_CI95=interval,
            simultaneous_critical95=critical,difference_recovery=float(d/(o.mean()-v.mean())) if o.mean()>v.mean() else None,
            difference_recovery_CI95=ci((y[ix]-x[ix]).mean(1)/den_boot) if ratio_stable else None,
            recovery_interval_stable=ratio_stable,operational_matching_validated=bool(matched),
            memory_advantage_resolved_simultaneously=bool(interval[0]>0 and matched),
            direction='memory' if interval[0]>0 else 'instantaneous' if interval[1]<0 else 'unresolved'))
    # Secondary tauW1 validation: fixed gain and joint calibration.
    validations=[]
    for scheme in ['fixed','joint']:
        suffix='ag0' if scheme=='fixed' else scheme
        a=replicate[('maniobra',1.,12.,f'full_t0_{suffix}')]
        b=replicate[('maniobra',1.,12.,f'full_t1_{suffix}')]
        x=np.array([a[i]['R'] for i in range(8)]);y=np.array([b[i]['R'] for i in range(8)])
        validations.append(dict(scheme=scheme,n_rep=8,R_instant=float(x.mean()),R_memory=float(y.mean()),
            difference_R=float((y-x).mean()),CI95=ci((y[ix]-x[ix]).mean(1)),secondary=True))
    statistic_controls=[]
    for protocol in ['forzado','maniobra']:
        differences={}
        for stat in ['full','binary']:
            a=replicate[(protocol,4.,12.,f'{stat}_t0_ag0')]
            b=replicate[(protocol,4.,12.,f'{stat}_t1_ag0')]
            x0=np.array([a[i]['R'] for i in range(8)]);x1=np.array([b[i]['R'] for i in range(8)])
            d=x1-x0;differences[stat]=d
            statistic_controls.append(dict(protocol=protocol,stat=stat,n_rep=8,
                R_instant=float(x0.mean()),R_memory=float(x1.mean()),
                difference_R=float(d.mean()),CI95=ci(d[ix].mean(1)),secondary=True))
        interaction=differences['binary']-differences['full']
        statistic_controls.append(dict(protocol=protocol,stat='interaction_binary_minus_full',n_rep=8,
            difference_R=float(interaction.mean()),CI95=ci(interaction[ix].mean(1)),secondary=True))
    info={'new_response_runs':len(rows),'condition_count':len(conditions),'replicate_condition_rows':len(rep_rows),
       'bootstrap':B,'bootstrap_seed':SEED,'resampling_unit':'paired seed replicate, mean over 18 angles',
       'primary_contrasts':contrasts,'secondary_tauW1':validations,'conditions':conditions,
       'secondary_statistic_controls':statistic_controls,
       'simultaneous_family_size':8,'simultaneous_method':'percentile95 maximum absolute centred bootstrap deviation over eight predeclared R contrasts',
       'null_matching':eq,'response_designs':{p:J(f'resultados/diseno_respuestas_{p}.json') for p in ['fixed','matched']}}
    jsonout('resultados/RESULTADOS_MEMORIA_GANANCIA.json',info)
    csvout('resultados/respuestas_por_angulo.csv',rows)
    csvout('resultados/respuestas_por_replica.csv',rep_rows)
    flat=[]
    for c in conditions:
        q={k:v for k,v in c.items() if not isinstance(v,(dict,list))}
        for name in ['R','absolute_gain','oracle_advantage','recovery','difference_from_oracle']:
            q[name+'_lo'],q[name+'_hi']=c[name+'_CI95'] or [None,None]
        for name,d in c['diagnostics'].items():q[name]=d['mean']
        flat.append(q)
    csvout('resultados/resumen_condiciones.csv',flat)
    flat=[]
    for c in contrasts:
        q={k:v for k,v in c.items() if not isinstance(v,(dict,list))}
        for name in ['pointwise_CI95','simultaneous_CI95','difference_recovery_CI95']:
            q[name+'_lo'],q[name+'_hi']=c[name] or [None,None]
        flat.append(q)
    csvout('resultados/contrastes_principales.csv',flat)
    print(json.dumps({'runs':len(rows),'conditions':len(conditions),'contrasts':contrasts},indent=2))

if __name__=='__main__':main()
