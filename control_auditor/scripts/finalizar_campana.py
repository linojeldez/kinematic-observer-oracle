"""Espera la campaña actual y genera tablas e informe al completar ambas fases."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1'
extra='/opt/homebrew/Cellar/numpy/2.5.2/lib/python3.14/site-packages'
env['PYTHONPATH']=extra+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
started=time.time()
while True:
    all_done=True;states={}
    for phase in ['fixed','matched']:
        done=ROOT/f'logs/resumen_respuestas_{phase}.json'
        prog=ROOT/f'logs/progreso_{phase}.json'
        if done.exists():
            states[phase]=json.loads(done.read_text())
            all_done=all_done and states[phase].get('finished',False)
        else:
            states[phase]=json.loads(prog.read_text()) if prog.exists() else {'completed':0}
            all_done=False
    lines=['# Estado de la ejecución','',
      'Campaña de memoria y ganancia con parámetros predefinidos. No se modifican el paper ni sus fuentes.','',
      '| Fase | Completadas | Planificadas | Estado |','|---|---:|---:|---|']
    for phase,v in states.items():
        lines.append(f"| {phase} | {v.get('completed',0)} | {v.get('planned',v.get('completed',''))} | {'terminada' if v.get('finished') else 'en ejecución'} |")
    lines+=['','La calibración nula está validada; las conclusiones de respuesta se emitirán al completar ambas fases.']
    (ROOT/'ESTADO_EJECUCION.md').write_text('\n'.join(lines)+'\n')
    if all_done:break
    time.sleep(20)
for script in ['validar_equivalencia_nula.py','analizar_respuestas.py','informe_resultados.py']:
    print('GENERANDO',script,flush=True)
    with (ROOT/'logs'/(Path(script).stem+'.log')).open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)
print('GENERANDO figuras',flush=True)
with (ROOT/'logs/figuras.log').open('w') as f:
    subprocess.run(['/usr/local/bin/Rscript','--vanilla',str(ROOT/'scripts/figuras_resultados.R')],
                   cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(ROOT/'logs/finalizacion.json').write_text(json.dumps({'finished':True,'elapsed_monitor':time.time()-started},indent=2))
print('LISTO: tablas, informe y figuras; falta la revisión final del expediente.',flush=True)
