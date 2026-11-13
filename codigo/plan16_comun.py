"""Arnés común del plan 16 (E70-E72): corrida con reanudación y estadística pareada.

El motor (`flocking.py`) no se toca. Los trabajos de varios informados usan `e68.job` (misma
semilla 7 + 1000 i_phi + réplica, mismos informados 5000 + réplica, mismo factor de ganancia).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from fase4 import R_rep, N_PHI

W = int(os.environ.get("FASE4_W", "5"))


def correr(jobs, f_parcial, etiqueta, jobfn, clavefn, previos=None):
    """Corre los trabajos que faltan (en paralelo), guarda parciales cada 2 min.
    `previos`: resultados ya guardados (otras etapas) que cuentan como hechos."""
    hechos = {}
    for x in previos or []:
        hechos[clavefn(x)] = x
    if os.path.exists(f_parcial):
        for x in json.load(open(f_parcial)):
            hechos[clavefn(x)] = x
    faltan = [a for a in jobs if clavefn(a) not in hechos]
    print(f"[{etiqueta}] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    nuevos = {}
    if os.path.exists(f_parcial):
        nuevos = {clavefn(x): x for x in json.load(open(f_parcial))}
    if faltan:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(jobfn, a) for a in faltan]
            for k, fu in enumerate(as_completed(futs), 1):
                x = fu.result()
                hechos[clavefn(x)] = x
                nuevos[clavefn(x)] = x
                if time.time() - ult > 120 or k == len(futs):
                    json.dump(list(nuevos.values()), open(f_parcial, "w"))
                    ult = time.time()
                    print(f"[{etiqueta}] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    return [hechos[clavefn(a)] for a in jobs if clavefn(a) in hechos], list(nuevos.values()), time.time() - t0


def series(res, celda, noms, N, pulso, tau_W, win, reps):
    """R por réplica (media sobre los 18 ángulos) de varias condiciones, con las réplicas
    comunes a todas. Devuelve (ids, {nom: array})."""
    d = {nom: R_rep(res, celda, nom, N, pulso, win, tau_W, reps) for nom in noms}
    com = sorted(set.intersection(*[set(ids) for ids, _ in d.values()]))
    return com, {nom: np.array([R[ids.index(k)] for k in com]) for nom, (ids, R) in d.items()}


def rec_boot(S, ref, obs, n_boot=4000, seed=42):
    """Recuperación de cada observador y diferencias, con un mismo remuestreo pareado de réplicas.
    S: {nom: array por réplica}; ref, «oraculo» deben estar. Devuelve dict con rec, ic95 por obs y
    una función para diferencias."""
    Rv, Ro = S[ref], S["oraculo"]
    n = len(Rv)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, n, (n_boot, n))
    den_b = Ro[b].mean(axis=1) - Rv[b].mean(axis=1)
    den = Ro.mean() - Rv.mean()
    out, boots = {}, {}
    for nom in obs:
        Rm = S[nom]
        rec = (Rm.mean() - Rv.mean()) / den
        rb = (Rm[b].mean(axis=1) - Rv[b].mean(axis=1)) / den_b
        boots[nom] = rb
        lo, hi = np.percentile(rb, [2.5, 97.5])
        out[nom] = {"rec": float(rec), "ic95": [float(lo), float(hi)], "n_rep": int(n), "R": float(Rm.mean())}
    out["_abs"] = {"R_ref": float(Rv.mean()), "R_oraculo": float(Ro.mean()), "ventaja": float(den),
                   "ventaja_ic95": [float(v) for v in np.percentile(den_b, [2.5, 97.5])]}

    def dif(a, c):
        d = boots[a] - boots[c]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"dif": float(out[a]["rec"] - out[c]["rec"]), "ic95": [float(lo), float(hi)],
                "p_menor0": float((d <= 0).mean())}
    return out, dif


def guardar(out, L, f_json, f_txt):
    json.dump(out, open(f_json, "w"), indent=1, default=float)
    open(f_txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
