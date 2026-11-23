"""Etapa 76 (revisión del 14-nov, objeción 10): calibración del «posterior» del observador que
integra, en vuelo sin informados ni falsas alarmas.

El observador puntúa el ciclo de trabajo d (media exponencial, con constante tau_det, del indicador
|eps| > 1,5 sigma_theta) con un log-cociente binomial multiplicado por n = tau_det / tau_obs, como si
la ventana tuviera n muestras independientes con tasa de excedencia q0 bajo «no informado». Aquí
se mide, con el propio motor (N = 250, k = 15, ruido de referencia, 20 s de asentamiento y 60 s de
registro, semillas 11-14):

  q        tasa de excedencia del indicador           (supuesta: q0 = P_0(|eps| > 1,5 sigma_theta))
  tau_I    tiempo de correlación integrado del indicador (un lado)
  n_ef     q (1 - q) / Var(d): muestras independientes que dan la varianza medida del ciclo de trabajo
  y la distribución del posterior bajo «nadie informado» con pi0 = 0,048 y 0,173.

Es la medición del 14-nov (tmp/auditoria_2026-11-14/muestras_efectivas.py) con salida JSON.
Diseño en docs/bitacora/resultados_E76-E85.md («E76»).

    PYTHONPATH=codigo python codigo/e76_muestras_efectivas.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from flocking import Flock, Params
from parametros import P0
from fase4 import ARM

RAIZ = os.environ.get("E76_RAIZ") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.environ.get("E76_SALIDA", "resultados")
TAUS, SEMILLAS, PIS = (0.25, 1.0, 4.0), (11, 12, 13, 14), (0.048, 0.173)
T_BURN, T_REG, L_ACF = 20.0, 60.0, 3.0


def job(a):
    tau_det, seed = a
    model, kw = ARM["adaptado"]
    p = Params(**{**asdict(P0), **kw, "tau_det": tau_det, "lambda_fp": 0.0, "tau_W": 4.0, "seed": seed})
    f = Flock(p, model)
    for _ in range(int(T_BURN / p.dt)):
        f.step()
    T = int(T_REG / p.dt)
    I = np.empty((T, p.N), bool); Dl = np.empty((T, p.N)); E = np.empty((T, p.N))
    for t in range(T):
        f.step()
        idx, _ = f._neighbors()
        eps = f.omega - f.omega[idx].mean(axis=1)
        E[t] = eps
        I[t] = np.abs(eps) > p.c_det * p.sigma_theta
        Dl[t] = f.delta
    q = float(I.mean())
    Ln = int(L_ACF / p.dt)

    def acf(z):
        z = z - z.mean(0)
        v = (z ** 2).mean()
        return np.array([1.0] + [float((z[l:] * z[:-l]).mean() / v) for l in range(1, Ln)])
    rI, rE = acf(I.astype(float)), acf(E)
    q0, q1 = (float(v) for v in f._q01)
    n = tau_det / p.tau_obs
    llr = n * (Dl * np.log(q1 / q0) + (1 - Dl) * np.log((1 - q1) / (1 - q0)))
    post = {}
    for pi0 in PIS:
        w = 1 / (1 + np.exp(-(llr + np.log(pi0 / (1 - pi0)))))
        post[f"{pi0:g}"] = {"media": float(w.mean()), "sobre_0.5": float((w > 0.5).mean()), "sobre_0.1": float((w > 0.1).mean())}
    return {"tau_det": tau_det, "semilla": seed, "q": q, "q0": q0, "q1": q1,
            "tau_I": float(p.dt * (0.5 + rI[1:].sum())), "tau_E": float(p.dt * (0.5 + rE[1:].sum())),
            "acf_I": {f"{s:g}": float(rI[int(round(s / p.dt))]) for s in (0.05, 0.1, 0.2, 0.5, 1.0)},
            "acf_E": {f"{s:g}": float(rE[int(round(s / p.dt))]) for s in (0.05, 0.1, 0.2, 0.5, 1.0)},
            "n_ef": float(q * (1 - q) / Dl.var()), "n_supuesto": float(n), "d_media": float(Dl.mean()),
            "ancho_robusto": float(np.median(np.abs(E - np.median(E))) * 1.4826), "post": post,
            "tau_obs": float(p.tau_obs), "sigma_theta": float(p.sigma_theta), "c_det": float(p.c_det)}


if __name__ == "__main__":
    os.chdir(RAIZ)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("FASE4_W", "5"))) as ex:
        res = list(ex.map(job, [(td, s) for td in TAUS for s in SEMILLAS]))
    out = {"corridas": res, "por_ventana": {}}
    L = ["=== E76 · calibración del posterior del observador que integra, sin informados ===",
         f"N = {P0.N}, {T_BURN:g} s + {T_REG:g} s, semillas {SEMILLAS}; tau_obs {res[0]['tau_obs']:g} s, "
         f"sigma_theta {res[0]['sigma_theta']:g}, umbral {res[0]['c_det']:g} sigma",
         "tau_det  semilla  q       q0      q1      tau_I    n_ef    n_supuesto  ancho"]
    for r in res:
        L.append(f"{r['tau_det']:6g}  {r['semilla']:6d}  {r['q']:.4f}  {r['q0']:.4f}  {r['q1']:.4f}  {r['tau_I']:.3f} s  "
                 f"{r['n_ef']:6.2f}  {r['n_supuesto']:6.2f}      {r['ancho_robusto']:.3f}")
    for td in TAUS:
        rs = [r for r in res if r["tau_det"] == td]
        m = lambda c: float(np.mean([r[c] for r in rs]))
        q = {"q": m("q"), "q0": rs[0]["q0"], "q1": rs[0]["q1"], "tau_I": m("tau_I"), "n_ef": m("n_ef"),
             "n_ef_sd": float(np.std([r["n_ef"] for r in rs])), "n_supuesto": rs[0]["n_supuesto"],
             "cociente": m("n_ef") / rs[0]["n_supuesto"], "ancho_robusto": m("ancho_robusto"),
             "acf_I": {c: float(np.mean([r["acf_I"][c] for r in rs])) for c in rs[0]["acf_I"]},
             "acf_E": {c: float(np.mean([r["acf_E"][c] for r in rs])) for c in rs[0]["acf_E"]},
             "post": {pi: {c: float(np.mean([r["post"][pi][c] for r in rs])) for c in rs[0]["post"][pi]} for pi in rs[0]["post"]}}
        out["por_ventana"][f"{td:g}"] = q
        L.append(f"tau_det {td:g} s: q {q['q']:.4f} (supuesta {q['q0']:.4f}); n_ef {q['n_ef']:.2f} ± {q['n_ef_sd']:.2f} frente a "
                 f"{q['n_supuesto']:.2f} supuestas (x{q['cociente']:.2f}); tau_I {q['tau_I']:.3f} s; posterior medio bajo H0: "
                 + ", ".join(f"pi0 {pi}: {v['media']:.4f} (> 0,5: {v['sobre_0.5']:.5f})" for pi, v in q["post"].items()))
    u = out["por_ventana"]["1"]
    L.append(f"autocorrelación del indicador a 0,05 / 0,1 / 0,2 s: {u['acf_I']['0.05']:.3f} / {u['acf_I']['0.1']:.3f} / {u['acf_I']['0.2']:.3f}; "
             f"del residuo: {u['acf_E']['0.05']:.3f} / {u['acf_E']['0.1']:.3f} / {u['acf_E']['0.2']:.3f}")
    L.append(f"{time.time() - t0:.0f} s")
    json.dump(out, open(os.path.join(SALIDA, "E76_muestras_efectivas.json"), "w"), indent=1)
    open(os.path.join(SALIDA, "E76_muestras_efectivas.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
