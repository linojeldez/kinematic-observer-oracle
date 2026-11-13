"""Etapa 70b (plan 16): en el régimen de la cabecera, ¿una ventana ajustada a la agilidad recupera lo que
la de 1 s pierde con a_max = 20 y 40? Mismas semillas, informados y referencias que E70.

Diseño y predicción A1 en docs/bitacora/resultados_E66-E75.md («E70b»).

    PYTHONPATH=codigo python codigo/e70b_ventana_agilidad.py
"""
from __future__ import annotations

import json
import os

from flocking import phi_grid
from fase4 import D_PHI, T_N, ARM
from e68_fraccion_informados import job, clave, informados, CELDAS
from e70_cabecera_robustez import variantes, N, NINF, N_REP, PI, OBS
from plan16_comun import correr, series, rec_boot, guardar

VENT = (0.25, 0.5)
AGIL = ("a20", "a40")
F_CRUDO, F_JSON, F_TXT = ("resultados/E70b_ventana_agilidad_crudo.json", "resultados/E70b_ventana_agilidad.json",
                          "resultados/E70b_ventana_agilidad.txt")


def trabajos():
    phis = phi_grid(D_PHI)
    V = variantes()
    out = []
    for var in AGIL:
        _, cal, _ = V[var]
        model, kw0 = ARM["adaptado"]
        for etq, pulso, tw in CELDAS:
            for v in VENT:
                kw = {**kw0, **cal, "pi0": PI, "tau_det": v}
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        out.append({"celda": f"n12_{var}", "nom": f"integrador_{v:g}s", "model": model, "kw": kw,
                                    "factor": 1.0, "N": N, "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phi),
                                    "rep": rep, "T": T_N[N], "inf": informados(N, NINF, rep)})
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res, nuevos, seg = correr(trabajos(), F_CRUDO + ".parcial", "E70b", job, clave)
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    todo = res + json.load(open("resultados/E70_cabecera_robustez_crudo.json"))["resp"]
    e70 = json.load(open("resultados/E70_cabecera_robustez.json"))
    out, L = {"rec": {}}, ["=== E70b · ventana corta con agilidad alta (12 de 250, prior 12/250, 8 réplicas) ==="]
    obs = ["integrador_0.25s", "integrador_0.5s", "integrador"] + list(OBS[1:])
    gana = 0
    for var in AGIL:
        for etq, pulso, tw in CELDAS:
            c = f"{var}|{etq}|{tw:g}"
            com, S = series(todo, f"n12_{var}", ["CoLd", "oraculo"] + obs, N, pulso, tw, T_N[N], list(range(N_REP)))
            q, dif = rec_boot(S, "CoLd", obs)
            corta = "integrador_0.25s" if var == "a40" else "integrador_0.5s"
            mejor_sm = max(OBS[1:], key=lambda k: q[k]["rec"])
            d = dif(corta, mejor_sm)
            gana += d["dif"] > 0
            out["rec"][c] = {"q": {k: q[k] for k in obs}, "corta": corta, "dif_vs_sm": d, "mejor_sm": mejor_sm}
            L.append(f"{c:22s} 0,25 s {q['integrador_0.25s']['rec']:.3f} | 0,5 s {q['integrador_0.5s']['rec']:.3f} | 1 s "
                     f"{q['integrador']['rec']:.3f} | {mejor_sm} {q[mejor_sm]['rec']:.3f} | {corta} − sin memoria "
                     f"{d['dif']:+.3f} [{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]")
    R = out["rec"]
    out["pred"] = {"A1": {"min_corta": min(R[c]["q"][R[c]["corta"]]["rec"] for c in R), "gana_al_sin_memoria": int(gana),
                          "cumple": bool(all(R[c]["q"][R[c]["corta"]]["rec"] > 0.25 for c in R) and gana >= 5)}}
    L.append(f"\nA1: {'CUMPLE' if out['pred']['A1']['cumple'] else 'FALLA'}  {json.dumps(out['pred']['A1'])}")
    L.append(f"\n{seg:.0f} s")
    guardar(out, L, F_JSON, F_TXT)
