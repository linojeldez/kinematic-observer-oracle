"""Compatibilidad adicional de Student con memoria frente al secuencial legado."""
from pathlib import Path
import hashlib
import json
import numpy as np
import motor_controles as m

ROOT=Path(__file__).resolve().parents[1]
before=hashlib.sha256((ROOT/'fuentes/flocking.py').read_bytes()).hexdigest()
rows=[]
for tau in [.25,1.]:
    for protocol in ['forzado','maniobra']:
        p=m.make_params(880007,tauW=4.,detector='secuencial',tau_det=tau)
        old=m.engine.run_response(p,'WiSt',float(np.deg2rad(95)),3.,t_s=3.,every_coh=100,pulse=protocol=='maniobra')
        new=m.run_trial({'stat':'full','tau':tau,'gain':tau/.083},880007,float(np.deg2rad(95)),
                        protocol=protocol,tauW=4.,t_R=3.,burn=3.,diagnostics=False)
        error=float(np.max(np.abs(old['proj']-new['proj'])))
        assert error<1e-12,(protocol,tau,error)
        rows.append({'tau':tau,'protocol':protocol,'max_projection_error':error,
                     'R_old':old['r'],'R_new':new['R'],'pass':True})
after=hashlib.sha256((ROOT/'fuentes/flocking.py').read_bytes()).hexdigest()
assert before==after
out={'pass':True,'checks':rows,'engine_sha256_before':before,'engine_sha256_after':after}
(ROOT/'piloto/anclas_secuencial.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
