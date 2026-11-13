"""Etapa 68 (plan 15, punto 2): recuperación contra la fracción de informados.

n_inf de N agentes informados, elegidos al azar (semilla 5000 + réplica, como E63), con un
prior igual a su fracción. N = 250: n_inf en {1, 3, 12, 25} (0,4-10 %); N = 1000: 48 (la
fracción de la celda central, 4,8 %). Tres celdas: líder forzado tau_W = 4 s, maniobra única
tau_W = 4 y 1 s. Observadores: integrador de 1 s, sin memoria, y sin memoria con el
log-cociente multiplicado por 6 y 12 (en la celda central también por 3 y 24); referencias
CoLd y oráculo. 32 réplicas en la celda central (N = 250, n_inf = 12), 8 en las demás.

Diseño y predicciones F1-F3 en docs/bitacora/resultados_E66-E75.md.

    PYTHONPATH=codigo python codigo/e68_fraccion_informados.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from flocking import Params, phi_grid
from parametros import P0
from fase4 import D_PHI, SEMILLA_R, WINS, W, _w, ARM, T_N, R_rep, rec_ic, limpio
from e63_varios_informados import FlockMulti, correr_uno

CELDAS = (("forzado", False, 4.0), ("maniobra", True, 4.0), ("maniobra", True, 1.0))
GRUPOS = ((250, 1), (250, 3), (250, 12), (250, 25), (1000, 48))
CENTRAL = (250, 12)
F_CRUDO, F_JSON, F_TXT = ("resultados/E68_fraccion_informados_crudo.json", "resultados/E68_fraccion_informados.json",
                          "resultados/E68_fraccion_informados.txt")


class FlockMultiG(FlockMulti):
    factor = 1.0

    def _llr_kinematic(self, idx):
        return self.factor * super()._llr_kinematic(idx)


def informados(N, n, rep):
    return sorted(np.random.default_rng(5000 + rep).choice(N, n, replace=False).tolist())


def job(a):
    import e63_varios_informados as e63
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    t0 = time.time()
    clase = type("FMG", (FlockMultiG,), {"factor": float(a["factor"])})
    e63.FlockMulti, viejo = clase, e63.FlockMulti
    try:
        proj = correr_uno(p, a["model"], a["phi"], a["T"], a["pulso"], a["inf"])
    finally:
        e63.FlockMulti = viejo
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "seg": time.time() - t0}


def clave(x):
    return f'{x["celda"]}|{x["nom"]}|{x["N"]}|{x["tau_W"]:g}|{int(x["pulso"])}|{x["i_phi"]}|{x["rep"]}'


def condiciones(N, n):
    pi = n / N
    out = [("CoLd", "CoLd", {}, 1.0), ("oraculo", "oraculo", {}, 1.0),
           ("integrador", "adaptado", {"pi0": pi}, 1.0), ("sin_memoria", "amplitud", {"pi0": pi}, 1.0),
           ("sin_memoria_x6", "amplitud", {"pi0": pi}, 6.0), ("sin_memoria_x12", "amplitud", {"pi0": pi}, 12.0)]
    if (N, n) == CENTRAL:
        out += [("sin_memoria_x3", "amplitud", {"pi0": pi}, 3.0), ("sin_memoria_x24", "amplitud", {"pi0": pi}, 24.0)]
    if N == 1000:                                     # por costo: sin las ganancias
        out = [x for x in out if x[0] in ("CoLd", "oraculo", "integrador", "sin_memoria")]
    return out


def n_rep(N, n, nom):
    if (N, n) == CENTRAL and nom in ("CoLd", "oraculo", "integrador"):
        return 32
    return 8


def trabajos():
    phis = phi_grid(D_PHI)
    out = []
    for N, n in GRUPOS:
        for etq, pulso, tw in CELDAS:
            for nom, base, kw, fac in condiciones(N, n):
                model, kw0 = ARM[base]
                for i, phi in enumerate(phis):
                    for rep in range(n_rep(N, n, nom)):
                        out.append({"celda": f"n{n}", "nom": nom, "model": model, "kw": {**kw0, **kw}, "factor": fac,
                                    "N": N, "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phi), "rep": rep,
                                    "T": T_N[N], "inf": informados(N, n, rep)})
    out.sort(key=lambda a: -a["N"])
    return out


def analizar(res):
    out, L = {"rec": {}, "abs": {}}, ["=== E68 · recuperación contra la fracción de informados (prior = fracción) ==="]
    for N, n in GRUPOS:
        for etq, pulso, tw in CELDAS:
            c = f"N{N}|n{n}|{etq}|{tw:g}"
            nr = 32 if (N, n) == CENTRAL else 8
            iv, Rv = R_rep(res, f"n{n}", "CoLd", N, pulso, T_N[N], tw, list(range(nr)))
            io, Ro = R_rep(res, f"n{n}", "oraculo", N, pulso, T_N[N], tw, list(range(nr)))
            fila = {}
            for nom, *_ in condiciones(N, n):
                if nom in ("CoLd", "oraculo"):
                    continue
                im, Rm = R_rep(res, f"n{n}", nom, N, pulso, T_N[N], tw, list(range(n_rep(N, n, nom))))
                com = sorted(set(im) & set(iv) & set(io))
                g = lambda ids, R: np.array([R[ids.index(k)] for k in com])
                fila[nom] = limpio(rec_ic(g(im, Rm), g(iv, Rv), g(io, Ro)))
            out["rec"][c] = fila
            out["abs"][c] = {"R_CoLd": float(np.mean(Rv)), "R_oraculo": float(np.mean(Ro)),
                             "ventaja": float(np.mean(Ro) - np.mean(Rv)), "fraccion": n / N}
            inst = max(v["rec"] for k, v in fila.items() if k.startswith("sin_memoria"))
            L.append(f"{c:26s} ventaja {out['abs'][c]['ventaja']:.3f} | integrador {fila['integrador']['rec']:.3f} "
                     f"[{fila['integrador']['ic95'][0]:.3f}; {fila['integrador']['ic95'][1]:.3f}] | sin memoria "
                     + " ".join(f"{k.replace('sin_memoria', 'x1').replace('x1_', '')}: {v['rec']:.3f}"
                                for k, v in fila.items() if k.startswith("sin_memoria"))
                     + f" (máx {inst:.3f})")
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = trabajos()
    f_par = F_CRUDO + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E68] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for k, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or k == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w"))
                ult = time.time()
                print(f"[E68] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    seg = time.time() - t0
    res = [hechos[clave(a)] for a in jobs]
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    # ancla: la celda central con 8 réplicas reproduce E63 (mismos informados y semillas)
    e63c = {(x["nom"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]): x
            for x in json.load(open("resultados/E63_varios_informados_crudo.json"))["resp"] if x["celda"] == "multi"}
    mapa = {"CoLd": "CoLd", "oraculo": "oraculo", "integrador": "obs_pi0.048"}
    dm, nc = 0.0, 0
    for x in res:
        if x["celda"] == "n12" and x["N"] == 250 and x["nom"] in mapa and x["rep"] < 8:
            y = e63c.get((mapa[x["nom"]], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]))
            if y:
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); nc += 1
    out, L = analizar(res)
    out["ancla"] = {"delta_max": dm, "comparaciones": nc, "cumple": bool(nc > 0 and dm < 1e-12)}
    L.insert(1, f"ancla (celda central frente a E63): |Δ| máx = {dm:.2e} en {nc} comparaciones")
    R = out["rec"]
    c250 = [c for c in R if c.startswith("N250")]
    serie = {cel: [R[f"N250|n{n}|{cel}"]["integrador"]["rec"] for n in (1, 3, 12, 25)]
             for cel in ("forzado|4", "maniobra|4", "maniobra|1")}
    out["pred"] = {
        "F1": {"serie": serie, "cumple": bool(all(s[0] < s[2] for s in serie.values()))},
        "F2": {"max_sin_memoria": {c: max(v["rec"] for k, v in R[c].items() if k.startswith("sin_memoria")) for c in R},
               "cumple": bool(all(max(v["rec"] for k, v in R[c].items() if k.startswith("sin_memoria")) < 0.25 for c in R))},
        "F3": {"N1000": {c: R[c]["integrador"]["rec"] for c in R if c.startswith("N1000")},
               "cumple": bool(all(R[c]["integrador"]["rec"] > 0.25 for c in R if c.startswith("N1000")))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
