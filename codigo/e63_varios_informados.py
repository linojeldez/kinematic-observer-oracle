"""Etapa 63 (plan 14): varios informados y un prior igual a su fracción.

12 de 250 agentes informados (al azar, semilla 5000 + réplica), observador de 1 s con pi0 en
{12/250, 0.173, 0.004}. Diseño y predicciones S1-S3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e63_varios_informados.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from flocking import Flock, Params, phi_grid
from parametros import P0
from fase4 import D_PHI, SEMILLA_R, WINS, W, _w, ARM, T_N, clave, R_rep, rec_ic, limpio, crudos_previos

N_INF, N_REP, N = 12, 8, 250
PIS = (12 / 250, 0.173, 0.004)
CELDAS = (("forzado", False, 4.0), ("maniobra", True, 4.0), ("maniobra", True, 1.0))
F_CRUDO, F_JSON, F_TXT = ("resultados/E63_varios_informados_crudo.json", "resultados/E63_varios_informados.json",
                          "resultados/E63_varios_informados.txt")


class FlockMulti(Flock):
    def __init__(self, p, model):
        super().__init__(p, model)
        self.mask_forz = np.zeros(p.N, dtype=bool)
        self.n_obj = None

    def desired_direction(self):
        out = super().desired_direction()
        if self.mask_forz.any():
            out = out.copy()
            out[self.mask_forz] = self.n_obj
        return out


def correr_uno(p, model, phi, t_R, pulso, inf, t_s=3.0):
    f = FlockMulti(p, model)
    for _ in range(int(round(t_s / p.dt))):
        f.step()
    n_phi = np.array([np.cos(phi), np.sin(phi)])
    if inf is None:                                   # ancla: el líder del motor
        inf = np.array([f.leader])
    inf = np.asarray(inf)
    f.n_obj = n_phi
    f.mask_forz[inf] = True
    f.will[inf] = 1.0
    f.d_des[inf] = n_phi
    activo = np.ones(len(inf), dtype=bool)            # aún no soltado
    n_steps = int(round(t_R / p.dt))
    proj = np.empty(n_steps)
    for s in range(n_steps):
        f.step()
        if pulso and activo.any():
            v = f.vel[inf]
            alin = (v @ n_phi) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            suelta = activo & ((alin > np.cos(np.deg2rad(10.0))) | (s * p.dt > 2.5))
            if suelta.any():
                activo &= ~suelta
                f.mask_forz[inf[suelta]] = False
        f.will[inf[activo]] = 1.0
        proj[s] = f.V() @ n_phi / p.v0
    return proj


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    t0 = time.time()
    proj = correr_uno(p, a["model"], a["phi"], a["T"], a["pulso"], a["inf"])
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "seg": time.time() - t0}


def informados(rep):
    return sorted(np.random.default_rng(5000 + rep).choice(N, N_INF, replace=False).tolist())


def trabajos():
    phis = phi_grid(D_PHI)
    conds = [("CoLd", "CoLd", {}), ("oraculo", "oraculo", {})]
    conds += [(f"obs_pi{p:.3f}", "adaptado", {"pi0": p}) for p in PIS]
    out = []
    for etq, pulso, tw in CELDAS:
        for nom, base, kw in conds:
            model, kw0 = ARM[base]
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    out.append({"celda": "multi", "nom": nom, "model": model, "kw": {**kw0, **kw}, "N": N,
                                "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phi), "rep": rep,
                                "T": T_N[N], "inf": informados(rep)})
    # ancla: un solo informado (el líder del motor), réplica 0, dos ángulos, forzado y maniobra
    for etq, pulso, tw in CELDAS[:2]:
        for nom, base in (("CoLd", "CoLd"), ("oraculo", "oraculo"), ("adaptado", "adaptado")):
            model, kw0 = ARM[base]
            for i in (4, 13):
                out.append({"celda": "ancla", "nom": nom, "model": model, "kw": dict(kw0), "N": N, "tau_W": tw,
                            "pulso": pulso, "i_phi": i, "phi": float(phis[i]), "rep": 0, "T": T_N[N], "inf": None})
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = trabajos()
    f_par = F_CRUDO + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E63] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for k, fu in enumerate(as_completed(futs), 1):
            x = fu.result(); hechos[clave(x)] = x
            if time.time() - ult > 120 or k == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w")); ult = time.time()
                print(f"[E63] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    res = [hechos[clave(a)] for a in jobs]
    seg = time.time() - t0
    json.dump({"resp": res, "informados": {r: informados(r) for r in range(N_REP)}}, open(F_CRUDO, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    prev = {(x["nom"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]): x for x in crudos_previos() if x["N"] == 250}
    dm, n = 0.0, 0
    for x in res:
        if x["celda"] == "ancla":
            nom = "ViSt" if (x["nom"] == "CoLd" and not x["pulso"]) else x["nom"]
            y = prev.get((x["nom"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"])) or prev.get((nom, x["tau_W"], x["pulso"], x["i_phi"], x["rep"]))
            if y:
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    L = [f"=== E63 · {N_INF} informados de {N}, observador de 1 s ({N_REP} réplicas) ===",
         f"ancla (un informado frente a E42): |Δ| máx = {dm:.2e} en {n} comparaciones"]
    out = {"ancla": {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)}, "rec": {}, "abs": {}}
    e48 = json.load(open("resultados/absolutos_E48.json"))
    for etq, pulso, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        iv, Rv = R_rep(res, "multi", "CoLd", N, pulso, 15.8, tw)
        io, Ro = R_rep(res, "multi", "oraculo", N, pulso, 15.8, tw)
        fila = {}
        for p in PIS:
            im, Rm = R_rep(res, "multi", f"obs_pi{p:.3f}", N, pulso, 15.8, tw)
            com = sorted(set(im) & set(iv) & set(io))
            g = lambda ids, R: np.array([R[ids.index(k)] for k in com])
            fila[f"{p:.3f}"] = limpio(rec_ic(g(im, Rm), g(iv, Rv), g(io, Ro)))
        out["rec"][c] = fila
        out["abs"][c] = {"R_CoLd": float(np.mean(Rv)), "R_oraculo": float(np.mean(Ro)), "ventaja": float(np.mean(Ro) - np.mean(Rv)),
                         "ventaja_un_lider": e48[f"{etq}|250|{tw:g}"]["ventaja_oraculo"]}
        L.append(f"{c:12s}: R CoLd {np.mean(Rv):.3f}, oráculo {np.mean(Ro):.3f} (ventaja {np.mean(Ro) - np.mean(Rv):.3f}; con un líder "
                 f"{out['abs'][c]['ventaja_un_lider']:.3f}) | " + " | ".join(
                     f"pi0 {k}: {v['rec']:.3f} [{v['ic95'][0]:.3f}; {v['ic95'][1]:.3f}]" for k, v in fila.items()))
    R = out["rec"]; A = out["abs"]; cs = list(R)
    out["pred"] = {
        "S1": {"rec": [R[c]["0.048"]["rec"] for c in cs], "cumple": bool(all(R[c]["0.048"]["rec"] > 0.25 for c in cs))},
        "S2": {"cumple": bool(all(R[c]["0.004"]["rec"] < R[c]["0.048"]["rec"] for c in cs))},
        "S3": {"cumple": bool(all(A[c]["ventaja"] < A[c]["ventaja_un_lider"] for c in cs))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
