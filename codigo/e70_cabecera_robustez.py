"""Etapa 70 (plan 16, R3 del revisor): la cabecera con otro ruido, otra agilidad y k = 70.

Régimen de la cabecera del paper: 12 de 250 informados (semilla 5000 + réplica, los de E63 y E68),
prior igual a su fracción, tres celdas (líder forzado tau_W = 4 s; maniobra única tau_W = 4 y 1 s).
Seis variantes de la dinámica con 8 réplicas pareadas: ruido x0,5 y x2, a_max = 5, 20 y 40 m s^-2
(sigma_W proporcional a a_max, como en E55), y k = 70. La anchura del residuo del observador es la
calibrada en cada variante (E49, E55, E65). La referencia es la celda central de E68.

Diseño y predicciones G1-G4 en docs/bitacora/resultados_E66-E75.md («Plan 16»).

    PYTHONPATH=codigo python codigo/e70_cabecera_robustez.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from fase4 import D_PHI, T_N, ARM
from parametros import P0
from e68_fraccion_informados import job, clave, informados, CELDAS
from plan16_comun import correr, series, rec_boot, guardar

S2 = float(np.sqrt(2.0))
N, NINF, N_REP = 250, 12, 8
PI = NINF / N
OBS = ("integrador", "sin_memoria", "sin_memoria_x6", "sin_memoria_x12")
F_CRUDO, F_JSON, F_TXT = ("resultados/E70_cabecera_robustez_crudo.json", "resultados/E70_cabecera_robustez.json",
                          "resultados/E70_cabecera_robustez.txt")


def variantes():
    """{nombre: (kw de la dinámica, kw del observador, omega_max / sigma_theta)}."""
    a49 = json.load(open("resultados/E49_sensibilidad_anchuras.json"))
    a55 = json.load(open("resultados/E55_agilidad.json"))["anchuras"]
    a70 = json.load(open("resultados/E65_k70_anchura.json"))["sigma"]
    out = {"ref": ({}, {}, (P0.a_max / P0.v0) / P0.sigma_theta)}
    for f in (0.5, 2.0):
        s = a49[f"k15|f{f:g}"]["sigma"]
        din = {"sigma_xi": f * S2}
        out[f"f{f:g}"] = (din, {**din, "sigma_theta": s}, (P0.a_max / P0.v0) / s)
    for a in (5.0, 20.0, 40.0):
        s = a55[str(a)]
        din = {"a_max": a}
        out[f"a{a:g}"] = (din, {**din, "sigma_theta": s, "sigma_W": 0.811 * a / 10.0}, (a / P0.v0) / s)
    out["k70"] = ({"k": 70}, {"k": 70, "sigma_theta": a70}, (P0.a_max / P0.v0) / a70)
    return out


def condiciones():
    return [("CoLd", "CoLd", 1.0), ("oraculo", "oraculo", 1.0), ("integrador", "adaptado", 1.0),
            ("sin_memoria", "amplitud", 1.0), ("sin_memoria_x6", "amplitud", 6.0),
            ("sin_memoria_x12", "amplitud", 12.0)]


def trabajos():
    phis = phi_grid(D_PHI)
    out = []
    for var, (din, cal, _) in variantes().items():
        if var == "ref":
            continue
        for etq, pulso, tw in CELDAS:
            for nom, base, fac in condiciones():
                model, kw0 = ARM[base]
                obs = base in ("adaptado", "amplitud")
                kw = {**kw0, **(cal if obs else din), **({"pi0": PI} if obs else {})}
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        out.append({"celda": f"n12_{var}", "nom": nom, "model": model, "kw": kw, "factor": fac,
                                    "N": N, "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phi),
                                    "rep": rep, "T": T_N[N], "inf": informados(N, NINF, rep)})
    # ancla: celda central de E68, réplica 0, ángulos 4 y 13
    for etq, pulso, tw in CELDAS:
        for nom, base, fac in condiciones():
            model, kw0 = ARM[base]
            kw = {**kw0, **({"pi0": PI} if base in ("adaptado", "amplitud") else {})}
            for i in (4, 13):
                out.append({"celda": "ancla", "nom": nom, "model": model, "kw": kw, "factor": fac, "N": N,
                            "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phis[i]), "rep": 0,
                            "T": T_N[N], "inf": informados(N, NINF, 0)})
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = trabajos()
    res, nuevos, seg = correr(jobs, F_CRUDO + ".parcial", "E70", job, clave)
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    e68 = [x for x in json.load(open("resultados/E68_fraccion_informados_crudo.json"))["resp"]
           if x["celda"] == "n12" and x["N"] == N]
    idx = {(x["nom"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]): x for x in e68}
    dm, nc = 0.0, 0
    for x in res:
        if x["celda"] == "ancla":
            y = idx.get((x["nom"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]))
            for w in x["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w])); nc += 1
    todo = res + [{**x, "celda": "n12_ref"} for x in e68]
    V = variantes()
    out = {"ancla": {"delta_max": dm, "comparaciones": nc, "cumple": bool(nc > 0 and dm < 1e-12)},
           "omega_sobre_sigma": {v: q[2] for v, q in V.items()}, "rec": {}, "dif": {}}
    L = [f"=== E70 · cabecera (12 de 250, prior 12/250) con otro ruido, agilidad y k = 70; {N_REP} réplicas ===",
         f"ancla (celda central de E68): |Δ| máx = {dm:.2e} en {nc} comparaciones"]
    for var in V:
        L.append(f"\n--- {var}: omega_max/sigma_theta = {V[var][2]:.2f} ---")
        for etq, pulso, tw in CELDAS:
            c = f"{var}|{etq}|{tw:g}"
            com, S = series(todo, f"n12_{var}", ("CoLd", "oraculo") + OBS, N, pulso, tw, T_N[N], list(range(N_REP)))
            q, dif = rec_boot(S, "CoLd", OBS)
            out["rec"][c] = q
            mejor = max(OBS[1:], key=lambda k: q[k]["rec"])
            out["dif"][c] = {"mejor_sin_memoria": mejor, **dif("integrador", mejor)}
            L.append(f"{etq} τ_W {tw:g}: ventaja {q['_abs']['ventaja']:.3f} | integrador {q['integrador']['rec']:.3f} "
                     f"[{q['integrador']['ic95'][0]:.3f}; {q['integrador']['ic95'][1]:.3f}] | sin memoria "
                     + " ".join(f"{k.replace('sin_memoria', 'x1').replace('x1_', '')} {q[k]['rec']:.3f}" for k in OBS[1:])
                     + f" | integrador − {mejor}: {out['dif'][c]['dif']:+.3f} [{out['dif'][c]['ic95'][0]:+.3f}; "
                     f"{out['dif'][c]['ic95'][1]:+.3f}]  (n = {len(com)})")
    R = out["rec"]
    cel = [f"{etq}|{tw:g}" for etq, _, tw in CELDAS]
    var6 = [v for v in V if v != "ref"]
    ri = lambda v, c: R[f"{v}|{c}"]["integrador"]["rec"]
    out["pred"] = {
        "G1": {"min": min(ri(v, c) for v in var6 for c in cel),
               "bajo_25": [f"{v}|{c}" for v in var6 for c in cel if ri(v, c) <= 0.25],
               "bajo_15": [f"{v}|{c}" for v in var6 for c in cel if ri(v, c) < 0.15],
               "cumple": bool(all(ri(v, c) > 0.25 for v in var6 for c in cel))},
        "G2": {"pares": {c: [ri("f2", c) - ri("a5", c), ri("f0.5", c) - ri("a20", c)] for c in cel},
               "cumple": bool(all(abs(ri("f2", c) - ri("a5", c)) <= 0.10 and abs(ri("f0.5", c) - ri("a20", c)) <= 0.10
                                  for c in cel))},
        "G3": {"pierde": [f"{v}|{c}" for v in var6 for c in cel if out["dif"][f"{v}|{c}"]["dif"] <= 0],
               "cumple": bool(all(out["dif"][f"{v}|{c}"]["dif"] > 0 for v in var6 for c in cel))},
        "G4": {"cociente": {c: ri("k70", c) / ri("ref", c) for c in cel},
               "cumple": bool(all(ri("k70", c) >= 2 / 3 * ri("ref", c) for c in cel))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    guardar(out, L, F_JSON, F_TXT)
