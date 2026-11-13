"""Etapa 58 (plan 13 y revisión integral, anexo B.2): memoria frente a ganancia de evidencia.

Una subclase de Flock multiplica el log-cociente cinemático por un factor fijo. Diseño y
predicciones M1-M3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e58_memoria_ganancia.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params, run_response, phi_grid
from parametros import P0
import fase4
from fase4 import (D_PHI, SEMILLA_R, WINS, _w, trabajo, clave, crudos_previos, pareado, limpio)

TAU_OBS = P0.tau_obs
N_REP, TAU_W = 4, 4.0
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))
F_CRUDO, F_JSON, F_TXT = ("resultados/E58_memoria_ganancia_crudo.json", "resultados/E58_memoria_ganancia.json",
                          "resultados/E58_memoria_ganancia.txt")
NOM_PREV = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s", 4.0: "ciclo 4 s"}


class FlockGanancia(Flock):
    factor = 1.0

    def _llr_kinematic(self, idx):
        return self.factor * super()._llr_kinematic(idx)


def n_ef(tau):
    return max(tau / TAU_OBS, 1.0)


def mecanismos():
    out = []
    for m in (3.0, 6.0, 12.0, 24.0):
        out.append((f"inst_m{m:g}", "amplitud", {}, m))
    for tau in (0.25, 0.5, 2.0, 4.0):
        out.append((f"ciclo{tau:g}_n12", "adaptado", {"tau_det": tau}, n_ef(1.0) / n_ef(tau)))
    for n in (3.0, 6.0, 24.0, 48.0):
        out.append((f"ciclo1_n{n:g}", "adaptado", {"tau_det": 1.0}, n / n_ef(1.0)))
    return out


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    flocking.Flock = type("FG", (FlockGanancia,), {"factor": float(a["factor"])})
    t0 = time.time()
    try:
        r = run_response(p, a["model"], a["phi"], a["T"], every_coh=100, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "seg": time.time() - t0}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phis = phi_grid(D_PHI)
    jobs = []
    for etq, pulso, ref in PROT:
        for nom, base, kw, fac in mecanismos():
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    a = trabajo("base", nom, 250, TAU_W, pulso, i, phi, rep, kw=kw, base=base)
                    a["factor"] = fac
                    jobs.append(a)
        # ancla: factor 1 con el ciclo de 1 s, réplica 0, dos ángulos
        for i in (4, 13):
            a = trabajo("ancla", "adaptado", 250, TAU_W, pulso, i, phis[i], 0, base="adaptado")
            a["factor"] = 1.0
            jobs.append(a)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=fase4.W) as ex:
        for fu in as_completed([ex.submit(job, a) for a in jobs]):
            res.append(fu.result())
    seg = time.time() - t0
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    prev = crudos_previos()
    for x in json.load(open("resultados/E42b_barrido_ventana_crudo.json"))["resp"]:
        prev.append({**x, "celda": "base"})
    idx = {clave(x): x for x in prev}
    dm, n = 0.0, 0
    for x in res:
        if x["celda"] == "ancla":
            y = idx.get(clave({**x, "celda": "base"}))
            for w in x["r"]:
                if y and w in y["r"]:
                    dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    # mezcla: nuevas + guardadas (refs, oráculo, m = 1, ventanas con su n_ef)
    todo = [x for x in res if x["celda"] == "base"]
    todo += [x for x in prev if x["N"] == 250 and abs(x["tau_W"] - TAU_W) < 1e-9 and x["rep"] < N_REP
             and x["nom"] in ("ViSt", "CoLd", "oraculo", "amplitud") + tuple(NOM_PREV.values())]
    out = {"ancla": {"delta_max": dm, "comparaciones": n}, "rec": {}}
    L = [f"=== E58 · memoria frente a ganancia (N = 250, tau_W = 4 s, 4 réplicas) ===",
         f"ancla (factor 1 frente a E42): |Δ| máx = {dm:.2e} en {n} comparaciones"]
    for etq, pulso, ref in PROT:
        g = lambda nom: limpio(pareado(todo, "base", nom, ref, 250, pulso, 15.8, TAU_W))
        fila = {"inst_m1": g("amplitud")}
        fila.update({f"inst_m{m:g}": g(f"inst_m{m:g}") for m in (3, 6, 12, 24)})
        fila.update({f"ciclo{t:g}_nef{n_ef(t):.0f}": g(NOM_PREV[t]) for t in (0.25, 0.5, 1.0, 2.0, 4.0)})
        fila.update({f"ciclo{t:g}_n12": g(f"ciclo{t:g}_n12") for t in (0.25, 0.5, 2.0, 4.0)})
        fila.update({f"ciclo1_n{nn:g}": g(f"ciclo1_n{nn:g}") for nn in (3, 6, 24, 48)})
        out["rec"][etq] = fila
        L.append(f"\n--- {etq} ---")
        L.append("sin memoria, ganancia m = 1/3/6/12/24: " + " / ".join(f"{fila[f'inst_m{m}']['rec']:.3f}" for m in ("1", "3", "6", "12", "24")))
        L.append("ciclo con su n_ef (0,25/0,5/1/2/4 s): " + " / ".join(f"{fila[k]['rec']:.3f}" for k in fila if k.startswith("ciclo") and "_nef" in k))
        L.append("ciclo con n_ef = 12 (0,25/0,5/1/2/4 s): " + " / ".join(
            f"{(fila[f'ciclo{t:g}_n12'] if t != 1.0 else fila['ciclo1_nef12'])['rec']:.3f}" for t in (0.25, 0.5, 1.0, 2.0, 4.0)))
        L.append("ciclo 1 s con n_ef = 3/6/12/24/48: " + " / ".join(
            f"{(fila[f'ciclo1_n{nn}'] if nn != '12' else fila['ciclo1_nef12'])['rec']:.3f}" for nn in ("3", "6", "12", "24", "48")))
    f, m = out["rec"]["forzado"], out["rec"]["maniobra"]
    inst_f = max(f[f"inst_m{x}"]["rec"] for x in ("1", "3", "6", "12", "24"))
    inst_m = max(m[f"inst_m{x}"]["rec"] for x in ("1", "3", "6", "12", "24"))
    out["pred"] = {
        "M1": {"max_inst_forzado": inst_f, "max_inst_maniobra": inst_m, "cumple": bool(inst_f < 0.30 and inst_m < 0.15)},
        "M2": {"cumple": bool(all(max((r[f"ciclo{t:g}_n12"] if t != 1.0 else r["ciclo1_nef12"])["rec"] for t in (1.0, 2.0)) ==
                                  max((r[f"ciclo{t:g}_n12"] if t != 1.0 else r["ciclo1_nef12"])["rec"] for t in (0.25, 0.5, 1.0, 2.0, 4.0))
                                  and r["ciclo0.25_n12"]["rec"] < 0.5 * r["ciclo1_nef12"]["rec"] for r in (f, m)))},
        "M3": {"cumple": bool(all(max(r[k]["rec"] for k in ("ciclo1_n6", "ciclo1_nef12", "ciclo1_n24"))
                                  - min(r[k]["rec"] for k in ("ciclo1_n6", "ciclo1_nef12", "ciclo1_n24")) < 0.15 for r in (f, m)))},
        "regla_inst_mitad": {"forzado": inst_f / f["ciclo1_nef12"]["rec"], "maniobra": inst_m / m["ciclo1_nef12"]["rec"]}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
