"""Etapa 43: la escalera de recuperación frente a tau_W.

Por qué. tau_W = 4 s es elección propia: en los modelos purificados de Balázs la
voluntad no decae y el único valor publicado es el optimizado de WillFull (1 s, tabla
S2). El Paper A lee la parte que el detector adaptado no recupera como «el estado que
sobrevive a su huella». Si eso es así, acortar tau_W debe acercar el detector al oráculo.

Diseño (banco de `run_response`; funciones de e42_escalera_N.py; nada del motor cambia):
  - N = 250, lambda_fp = 0, k = 15; tau_W en {1, 2, 4} s.
  - Protocolo de maniobra única (pulso): es el único en que tau_W actúa, porque con el
    líder forzado su voluntad se reimpone en cada paso.
  - Mecanismos: ViSt, oráculo, amplitud, adaptado (ciclo de 1 s); 18 ángulos; 8 réplicas;
    semilla 7 + 1000 * i_phi + réplica.
  - Control: con líder forzado, tau_W = 1 s debe dar exactamente lo mismo que tau_W = 4 s
    (réplica 0, los cuatro mecanismos).

Las predicciones se declaran en docs/bitacora/resultados_E36-E45.md antes de correr.

    PYTHONPATH=codigo python codigo/e43_escalera_tauW.py
    PYTHONPATH=codigo E42_RAPIDO=1 python codigo/e43_escalera_tauW.py     # humo
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from e42_escalera_N import (MEC, D_PHI, RAPIDO, N_REP, _w, correr, R_rep, rec_ic)

N = 250
T = 15.8
TAUS = (1.0, 2.0, 4.0)
SUF = "_rapido" if RAPIDO else ""
F_CRUDO = f"resultados/E43_escalera_tauW_crudo{SUF}.json"
F_JSON = f"resultados/E43_escalera_tauW{SUF}.json"
F_TXT = f"resultados/E43_escalera_tauW{SUF}.txt"
F_PARCIAL = F_CRUDO + ".parcial"


def trabajos():
    phis = phi_grid(D_PHI)
    phis_i = [(0, phis[0]), (9, phis[9])] if RAPIDO else list(enumerate(phis))
    out = []
    for tau in TAUS:
        for nom, (model, kw) in MEC.items():
            for i, phi in phis_i:
                for rep in range(N_REP):
                    out.append((nom, model, kw, N, tau, True, i, float(phi), rep, T))
    for tau in (1.0, 4.0):                               # control con líder forzado
        for nom, (model, kw) in MEC.items():
            for i, phi in phis_i:
                out.append((nom, model, kw, N, tau, False, i, float(phi), 0, T))
    return out


def analizar(res, e20=None):
    out = {"config": {"N": N, "T": T, "tau_W": list(TAUS), "replicas": N_REP,
                      "protocolo": "pulso", "rapido": RAPIDO}, "rec": {}, "pred": {}}
    L = []
    P = L.append
    P("=== E43 · escalera de recuperación frente a tau_W (maniobra única, N = 250) ===")
    P(f'{"tau_W":>6s} {"ventana":>8s} | {"R ViSt":>7s} {"R orác.":>7s} {"ventaja":>7s} | '
      f'{"amplitud":>22s} | {"adaptado":>22s}')
    for tau in TAUS:
        for win in (4.0, 8.0, 15.8):
            Rv = R_rep(res, "ViSt", N, True, win, tau)
            Ro = R_rep(res, "oraculo", N, True, win, tau)
            fila = {}
            for nom in ("amplitud", "adaptado"):
                q = rec_ic(R_rep(res, nom, N, True, win, tau), Rv, Ro)
                q.pop("_boot", None)
                fila[nom] = q
            out["rec"][f"{tau:g}|{_w(win)}"] = {"R_vist": float(Rv.mean()),
                                               "R_oraculo": float(Ro.mean()),
                                               "ventaja": float(Ro.mean() - Rv.mean()), **fila}
            a, d = fila["amplitud"], fila["adaptado"]
            P(f'{tau:6.1f} {win:8.1f} | {Rv.mean():7.4f} {Ro.mean():7.4f} '
              f'{Ro.mean() - Rv.mean():7.4f} | '
              f'{a["rec"]:6.3f} [{a["ic95"][0]:5.3f}; {a["ic95"][1]:5.3f}] | '
              f'{d["rec"]:6.3f} [{d["ic95"][0]:5.3f}; {d["ic95"][1]:5.3f}]')

    # ---- controles y anclas
    P("\n=== controles ===")
    d_tr = 0.0
    for nom in MEC:
        a = R_rep(res, nom, N, False, 15.8, 1.0)
        b = R_rep(res, nom, N, False, 15.8, 4.0)
        if len(a) and len(b):
            d_tr = max(d_tr, float(np.abs(a - b).max()))
    P(f"líder forzado, tau_W = 1 s frente a 4 s (réplica 0, cuatro mecanismos): |Δ| máx = {d_tr:.2e}")
    d_v = 0.0
    for tau in TAUS[:-1]:
        a, b = R_rep(res, "ViSt", N, True, 15.8, tau), R_rep(res, "ViSt", N, True, 15.8, 4.0)
        d_v = max(d_v, float(np.abs(a - b).max()))
    P(f"ViSt con pulso, entre valores de tau_W: |Δ| máx = {d_v:.2e}")
    d20 = None
    if e20 is not None:
        d20 = 0.0
        for k20, nom in (("R_vist", "ViSt"), ("R_orac", "oraculo")):
            a, b = np.array(e20["PULSO"][k20]), R_rep(res, nom, N, True, 15.8, 4.0)
            n = min(len(a), len(b))
            d20 = max(d20, float(np.abs(a[:n] - b[:n]).max()))
        for k20, nom in (("amplitud", "amplitud"), ("ciclo_1s", "adaptado")):
            a, b = np.array(e20["PULSO"][k20]["R"]), R_rep(res, nom, N, True, 15.8, 4.0)
            n = min(len(a), len(b))
            d20 = max(d20, float(np.abs(a[:n] - b[:n]).max()))
        P(f"tau_W = 4 s contra E20_pulso.json (R por réplica): |Δ| máx = {d20:.2e}")
    out["controles"] = {"tractor_tau1_vs_tau4": d_tr, "vist_entre_tau": d_v, "E20": d20}

    pr = out["pred"]

    def g(tau, nom, campo="rec"):
        return out["rec"][f"{tau:g}|15.8"][nom][campo] if nom else out["rec"][f"{tau:g}|15.8"][campo]

    pr["T0"] = {"E20": d20, "cumple": bool(d20 is not None and d20 < 1e-9)}
    pr["T1"] = {"tractor": d_tr, "vist": d_v, "cumple": bool(d_tr < 1e-12 and d_v < 1e-12)}
    v1, v2, v4 = g(1.0, None, "ventaja"), g(2.0, None, "ventaja"), g(4.0, None, "ventaja")
    pr["T2"] = {"ventaja": {"1": v1, "2": v2, "4": v4}, "cociente_1_4": v1 / v4,
                "cumple": bool(v1 < v2 < v4 and 0.35 <= v1 / v4 <= 0.75)}
    r1, r2, r4 = g(1.0, "adaptado"), g(2.0, "adaptado"), g(4.0, "adaptado")
    pr["T3"] = {"rec_adaptado": {"1": r1, "2": r2, "4": r4},
                "cumple": bool(r1 >= 0.75 and r4 < r2 < r1)}
    am = [g(t, "amplitud") for t in TAUS]
    pr["T4"] = {"rec_amplitud": am, "cumple": bool(max(am) <= 0.15)}
    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        P(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v, ensure_ascii=False)[:300]}')
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res, seg = correr(trabajos(), F_PARCIAL, "E43")
    with open(F_CRUDO, "w") as fh:
        json.dump({"resp": res}, fh)
    if os.path.exists(F_PARCIAL):
        os.remove(F_PARCIAL)
    e20 = None
    if os.path.exists("resultados/E20_pulso.json"):
        with open("resultados/E20_pulso.json") as fh:
            e20 = json.load(fh)
    out, L = analizar(res, e20)
    L.append(f"\n{seg:.0f} s")
    out["segundos"] = seg
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
