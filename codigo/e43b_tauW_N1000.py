"""Etapa 43b: tau_W = 1 s con N = 1000 (maniobra única).

Por qué. E43 dio, con N = 250, que el detector adaptado recupera 0,998 [0,986; 1,009] de
la ventaja del oráculo cuando tau_W = 1 s. E42 dio que con tau_W = 4 s la recuperación
cae con N (0,603 -> 0,447 con maniobra única). Si lo que el detector no recupera es el
estado que sobrevive a su huella, con tau_W = 1 s la recuperación debe seguir cerca de 1
en una bandada cuatro veces mayor.

Diseño: N = 1000, tau_W = 1 s, maniobra única, 18 ángulos, 8 réplicas, mismas semillas;
oráculo, amplitud y adaptado. ViSt no usa la voluntad: se toma del crudo de E42 y se
verifica con la réplica 0 corrida de nuevo.

    PYTHONPATH=codigo python codigo/e43b_tauW_N1000.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from e42_escalera_N import (MEC, D_PHI, N_REP, _w, correr, R_rep, rec_ic)

N, T, TAU = 1000, 31.6, 1.0
F_CRUDO = "resultados/E43b_tauW_N1000_crudo.json"
F_JSON = "resultados/E43b_tauW_N1000.json"
F_TXT = "resultados/E43b_tauW_N1000.txt"
F_PARCIAL = F_CRUDO + ".parcial"


def trabajos():
    phis = phi_grid(D_PHI)
    out = []
    for nom in ("oraculo", "amplitud", "adaptado"):
        model, kw = MEC[nom]
        for i, phi in enumerate(phis):
            for rep in range(N_REP):
                out.append((nom, model, kw, N, TAU, True, i, float(phi), rep, T))
    model, kw = MEC["ViSt"]
    for i, phi in enumerate(phis):                       # control: ViSt, réplica 0
        out.append(("ViSt", model, kw, N, TAU, True, i, float(phi), 0, T))
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nuevos, seg = correr(trabajos(), F_PARCIAL, "E43b")
    with open(F_CRUDO, "w") as fh:
        json.dump({"resp": nuevos}, fh)
    if os.path.exists(F_PARCIAL):
        os.remove(F_PARCIAL)
    with open("resultados/E42_escalera_N_crudo.json") as fh:
        e42 = json.load(fh)["resp"]

    # control: ViSt con tau_W = 1 s (réplica 0) contra ViSt de E42 (tau_W = 4 s)
    idx = {(x["i_phi"], x["rep"]): x for x in e42
           if x["nom"] == "ViSt" and x["N"] == N and x["pulso"]}
    d_v = max(abs(x["r"][w] - idx[(x["i_phi"], 0)]["r"][w])
              for x in nuevos if x["nom"] == "ViSt" for w in x["r"])
    # ViSt de E42 reetiquetado a tau_W = 1 s para las 8 réplicas
    vist = [{**x, "tau_W": TAU} for x in e42 if x["nom"] == "ViSt" and x["N"] == N and x["pulso"]]
    res = [x for x in nuevos if x["nom"] != "ViSt"] + vist

    out = {"config": {"N": N, "tau_W": TAU, "T": T, "replicas": N_REP, "protocolo": "pulso"},
           "control_vist": d_v, "rec": {}}
    L = ["=== E43b · tau_W = 1 s con N = 1000 (maniobra única) ===",
         f"control: ViSt con tau_W = 1 s frente a E42 (tau_W = 4 s), réplica 0: |Δ| máx = {d_v:.2e}",
         f'{"ventana":>8s} | {"R ViSt":>7s} {"R orác.":>7s} {"ventaja":>7s} | '
         f'{"amplitud":>22s} | {"adaptado":>22s}']
    for win in (4.0, 8.0, 15.8, 31.6):
        Rv = R_rep(res, "ViSt", N, True, win, TAU)
        Ro = R_rep(res, "oraculo", N, True, win, TAU)
        fila = {}
        for nom in ("amplitud", "adaptado"):
            q = rec_ic(R_rep(res, nom, N, True, win, TAU), Rv, Ro)
            q.pop("_boot", None)
            fila[nom] = q
        out["rec"][_w(win)] = {"R_vist": float(Rv.mean()), "R_oraculo": float(Ro.mean()),
                               "ventaja": float(Ro.mean() - Rv.mean()), **fila}
        a, d = fila["amplitud"], fila["adaptado"]
        L.append(f'{win:8.1f} | {Rv.mean():7.4f} {Ro.mean():7.4f} {Ro.mean() - Rv.mean():7.4f} | '
                 f'{a["rec"]:6.3f} [{a["ic95"][0]:5.3f}; {a["ic95"][1]:5.3f}] | '
                 f'{d["rec"]:6.3f} [{d["ic95"][0]:5.3f}; {d["ic95"][1]:5.3f}]')
    r = out["rec"]["31.6"]
    out["pred"] = {
        "U0": {"control_vist": d_v, "cumple": bool(d_v < 1e-12)},
        "U1": {"rec_adaptado": r["adaptado"]["rec"], "ic95": r["adaptado"]["ic95"],
               "cumple": bool(r["adaptado"]["rec"] >= 0.85)},
        "U2": {"rec_amplitud": r["amplitud"]["rec"], "cumple": bool(r["amplitud"]["rec"] <= 0.20)},
    }
    L.append("\n=== Predicciones declaradas ===")
    for k, v in out["pred"].items():
        L.append(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v)[:300]}')
    L.append(f"\n{seg:.0f} s")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
