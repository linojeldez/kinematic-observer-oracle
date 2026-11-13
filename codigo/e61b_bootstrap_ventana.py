"""Etapa 61b (plan 14, revisor independiente): incertidumbre de la ventana óptima.

Sin simulaciones. Desde los crudos de E42b, E55, E56, E58 y E61 (4 réplicas): remuestreo de
réplicas y, en cada remuestreo, la ventana de mayor recuperación. Peso ligado a la ventana y
peso fijo (n_ef = 12).

    PYTHONPATH=codigo python codigo/e61b_bootstrap_ventana.py
"""
import json
import os

import numpy as np

from fase4 import R_rep, crudos_previos

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L_ = lambda f: json.load(open(f"resultados/{f}"))["resp"]
todo = crudos_previos() + [{**x, "celda": "base"} for x in L_("E42b_barrido_ventana_crudo.json")]
todo += L_("E55_agilidad_crudo.json") + L_("E56_ventana_ruido_crudo.json") + L_("E61_ventana_peso_fijo_crudo.json")
todo += [x for x in L_("E58_memoria_ganancia_crudo.json") if x["celda"] == "base"]
VENT = (0.25, 0.5, 1.0, 2.0, 4.0)
PREV = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s", 4.0: "ciclo 4 s"}
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))


def nombres(celda, peso):
    if celda == "base":
        if peso == "ligado":
            return {v: ("base", PREV[v]) for v in VENT}
        return {v: ("base", f"ciclo{v:g}_n12" if v != 1.0 else "adaptado") for v in VENT}
    if peso == "ligado":
        return {v: (celda, f"ciclo{v:g}") for v in VENT}
    return {v: (celda, f"ciclo{v:g}_n12" if v != 1.0 else "ciclo1") for v in VENT}


rng = np.random.default_rng(611)
out, L = {}, ["=== E61b · incertidumbre de la ventana óptima (4 réplicas, 2000 remuestreos) ==="]
for celda in ("base", "a5", "a20", "a40", "f0.5", "f2"):
    for etq, pulso, ref in PROT:
        refc = celda
        for peso in ("ligado", "fijo"):
            nm = nombres(celda, peso)
            iv, Rv = R_rep(todo, refc, ref, 250, pulso, 15.8, 4.0)
            io, Ro = R_rep(todo, refc, "oraculo", 250, pulso, 15.8, 4.0)
            S = {v: dict(zip(*R_rep(todo, c, n, 250, pulso, 15.8, 4.0))) for v, (c, n) in nm.items()}
            com = sorted(set(iv) & set(io) & set.intersection(*[set(s) for s in S.values()]))
            com = [k for k in com if k < 4]
            if len(com) < 4:
                L.append(f"{celda} {etq} {peso}: faltan réplicas ({com})"); continue
            A = {v: np.array([S[v][k] for k in com]) for v in VENT}
            rv = np.array([dict(zip(iv, Rv))[k] for k in com]); ro = np.array([dict(zip(io, Ro))[k] for k in com])
            B = rng.integers(0, len(com), (2000, len(com)))
            den = ro[B].mean(1) - rv[B].mean(1)
            recs = np.stack([(A[v][B].mean(1) - rv[B].mean(1)) / den for v in VENT], 1)
            gana = np.bincount(recs.argmax(1), minlength=len(VENT)) / len(B)
            punto = [(A[v].mean() - rv.mean()) / (ro.mean() - rv.mean()) for v in VENT]
            mejor = VENT[int(np.argmax(punto))]
            out[f"{etq}|{celda}|{peso}"] = {"mejor": mejor, "frac": {str(v): float(g) for v, g in zip(VENT, gana)},
                                           "rec": {str(v): float(r) for v, r in zip(VENT, punto)}}
            L.append(f"{celda:5s} {etq:9s} peso {peso:6s}: mejor {mejor:g} s | gana en: "
                     + ", ".join(f"{v:g} s {100 * g:.0f} %" for v, g in zip(VENT, gana) if g > 0))
json.dump(out, open("resultados/E61b_bootstrap_ventana.json", "w"), indent=1)
open("resultados/E61b_bootstrap_ventana.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
