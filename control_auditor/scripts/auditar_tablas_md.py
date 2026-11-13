"""Compara las seis tablas numéricas del MD con resultados y calibración."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
md=(ROOT/'INFORME_PRIORIDAD_3.md').read_text()
x=json.loads((ROOT/'resultados/RESULTADOS_MEMORIA_GANANCIA.json').read_text())
c=json.loads((ROOT/'calibracion/calibracion_final.json').read_text())
checked=0
rows_checked=0

def table(section):
    block=md.split('## '+section+'\n',1)[1].split('\n## ',1)[0]
    lines=[line for line in block.splitlines() if line.startswith('|')]
    return [[v.strip() for v in line.strip('|').split('|')] for line in lines[2:]]

def verify(text, values, digits):
    global checked
    actual=[float(v) for v in re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',text)]
    expected=[float(f'{v:.{digits}f}') for v in values]
    assert actual==expected,(text,actual,expected)
    checked+=len(values)

t=table('Calibración independiente de la respuesta')
assert len(t)==6
for row,v in zip(t,c['variants']):
    s=v['validation_summary']
    verify(row[1],[v['tau']],0)
    verify(row[2],[v['gain']],5);verify(row[3],[v['intercept']],5)
    verify(row[4],[s['fpr']]+s['ci95_fpr'],5)
    verify(row[5],[s['mean_attention']]+s['ci95_mean_attention'],5)
rows_checked+=len(t)

t=table('Contrastes principales de memoria');assert len(t)==8
for row,v in zip(t,x['primary_contrasts']):
    assert row[0]==v['protocol']
    verify(row[2],[v['R_instant']],4);verify(row[3],[v['R_memory']],4)
    verify(row[4],[v['difference_R']]+v['simultaneous_CI95'],4)
    verify(row[5],[100*q for q in [v['difference_recovery']]+v['difference_recovery_CI95']],2)
rows_checked+=len(t)

central={'full_t0_ag0','full_t1_ag0','full_t0_joint','full_t1_joint'}
t=table('Recuperación y valores absolutos')
conditions=[v for v in x['conditions'] if v['burn']==12 and v['variant'] in central]
assert len(t)==len(conditions)==12
for row,v in zip(t,conditions):
    assert row[0]==v['protocol'];verify(row[1],[v['tauW']],0)
    verify(row[3],[v['R']]+v['R_CI95'],4)
    verify(row[4],[v['R_base']],4);verify(row[5],[v['R_oracle']],4)
    verify(row[6],[100*q for q in [v['recovery']]+v['recovery_CI95']],2)
rows_checked+=len(t)

t=table('Atención durante el giro: diagnóstico descriptivo')
conditions=[v for v in conditions if v['tauW']==4];assert len(t)==len(conditions)==8
for row,v in zip(t,conditions):
    assert row[0]==v['protocol']
    for index,key in [(2,'nonself_pair_weight_mean'),(3,'leader_weight_per_receiver'),(4,'leader_attention_fraction')]:
        d=v['diagnostics']['diag_'+key];verify(row[index],[d['mean']]+d['CI95'],5)
rows_checked+=len(t)

for heading,values in [('Validación secundaria con tauW de 1 s',x['secondary_tauW1']),
                      ('Mismo estadístico binario con y sin memoria',[v for v in x['secondary_statistic_controls'] if not v['stat'].startswith('interaction')])]:
    t=table(heading);assert len(t)==len(values)
    offset=1 if 'tauW' in heading else 2
    for row,v in zip(t,values):
        verify(row[offset],[v['R_instant']],4)
        verify(row[offset+1],[v['R_memory']],4)
        verify(row[offset+2],[v['difference_R']]+v['CI95'],4)
    rows_checked+=len(t)

result={'passed':True,'tables':6,'data_rows':rows_checked,'numeric_values':checked,
        'discrepancies':0,'method':'Compara las cifras impresas con las salidas numéricas a la precisión decimal declarada.'}
(ROOT/'evidencia/AUDITORIA_TABLAS_MD.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
