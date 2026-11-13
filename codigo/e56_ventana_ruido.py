"""Etapa 56 (plan 13): la ventana óptima del observador con otro nivel de ruido.

Ruido x0,5 y x2 (celdas de E49), observador calibrado; barrido de 0,25 a 4 s. Diseño, ancla y
predicciones en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e56_ventana_ruido.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from fase4 import D_PHI, trabajo, correr, pareado, limpio

S2 = float(np.sqrt(2.0))
FS = (0.5, 2.0)
VENT = (0.25, 0.5, 1.0, 2.0, 4.0)
N_REP, TAU_W = 4, 4.0
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))
F_CRUDO, F_JSON, F_TXT = ("resultados/E56_ventana_ruido_crudo.json", "resultados/E56_ventana_ruido.json",
                          "resultados/E56_ventana_ruido.txt")


def trabajos(anch):
    phis = phi_grid(D_PHI)
    out = []
    for f in FS:
        kw0 = dict(k=15, sigma_xi=f * S2)
        cal = dict(kw0, sigma_theta=anch[f"k15|f{f:g}"]["sigma"])
        for etq, pulso, ref in PROT:
            mecs = [(ref, ref, kw0), ("oraculo", "oraculo", kw0), ("amplitud", "amplitud", cal)]
            mecs += [(f"ciclo{v:g}", "adaptado", {**cal, "tau_det": v}) for v in VENT]
            for nom, base, kw in mecs:
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        out.append(trabajo(f"f{f:g}", nom, 250, TAU_W, pulso, i, phi, rep, kw=kw, base=base))
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    anch = json.load(open("resultados/E49_sensibilidad_anchuras.json"))
    res, seg = correr(trabajos(anch), F_CRUDO + ".parcial", "E56")
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    L = ["=== E56 · ventana óptima con ruido x0,5 y x2 (N = 250, k = 15, tau_W = 4 s, 4 réplicas) ==="]
    out = {"rec": {}, "mejor": {}}
    # ancla: líder forzado, réplicas 0-3, frente a la cruz de E49
    e49 = json.load(open("resultados/E49_sensibilidad_crudo.json"))["resp"]
    idx = {(x["celda"], x["nom"], x["pulso"], x["i_phi"], x["rep"]): x for x in e49}
    mapa = {"ViSt": "ViSt", "oraculo": "oraculo", "amplitud": "amplitud_cal", "ciclo1": "adaptado_cal"}
    dm, n = 0.0, 0
    for x in res:
        if x["pulso"] or x["nom"] not in mapa:
            continue
        y = idx.get((f"k15|{x['celda']}", mapa[x["nom"]], False, x["i_phi"], x["rep"]))
        if y is None:
            continue
        for w in x["r"]:
            if w in y["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    out["ancla"] = {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)}
    L.append(f"ancla frente a la cruz de E49: |Δ| máx = {dm:.2e} en {n} comparaciones")
    for f in FS:
        for etq, pulso, ref in PROT:
            fila = {nom: limpio(pareado(res, f"f{f:g}", nom, ref, 250, pulso, 15.8, TAU_W))
                    for nom in ["amplitud"] + [f"ciclo{v:g}" for v in VENT]}
            vs = {v: fila[f"ciclo{v:g}"]["rec"] for v in VENT}
            mejor = max(vs, key=vs.get)
            out["rec"][f"{etq}|{f:g}"] = fila
            out["mejor"][f"{etq}|{f:g}"] = {"ventana": mejor, "rec": vs[mejor]}
            L.append(f"ruido x{f:g} {etq:9s}: ventaja {fila['ciclo1']['ventaja']:.3f} | sin memoria "
                     f"{fila['amplitud']['rec']:.3f} | " + " | ".join(f"{v:g} s {vs[v]:.3f}" for v in VENT)
                     + f" | mejor {mejor:g} s")
    m = out["mejor"]
    out["pred"] = {
        "R1": {"ventanas": [m["forzado|2"]["ventana"], m["maniobra|2"]["ventana"]],
               "cumple": bool(all(2 <= m[f"{e}|2"]["ventana"] <= 4 for e in ("forzado", "maniobra")))},
        "R2": {"ventanas": [m["forzado|0.5"]["ventana"], m["maniobra|0.5"]["ventana"]],
               "cumple": bool(all(0.25 <= m[f"{e}|0.5"]["ventana"] <= 1 for e in ("forzado", "maniobra")))},
        "regla": {"todas_entre_1_y_2": bool(all(1 <= v["ventana"] <= 2 for v in m.values()))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
