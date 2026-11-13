"""Etapa 72 (plan 16, R2 del revisor): persistencia con falsas alarmas.

Protocolo de E64 (N = 250, k = 15, tau_W = 4 s, 600 s tras 20 s de asentamiento, semillas
11 + réplica, D por incrementos a L = 10 s) con falsas alarmas: cada agente se sobresalta sin causa
con tasa lambda_FP (proceso de Poisson; rumbo deseado al azar y will = 1, que decae con tau_W).
lambda_FP = 8,42e-5 s^-1 (la tasa medida en peces) con 16 réplicas y x10 con 8. La persistencia
se da como P/P_ViSt contra el ViSt sin falsas alarmas de E64 (mismas semillas) y como fracción del
costo del oráculo, c_P = (1 - r_x) / (1 - r_or).

Diseño y predicciones P4-P7 en docs/bitacora/resultados_E66-E75.md («Plan 16»).

    PYTHONPATH=codigo python codigo/e72_persistencia_fa.py
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params
from parametros import P0
from fase4 import ARM
from e64_persistencia import clase, D_incr, boot_cociente, T_P, T_BURN, DT_S, SEM_P
from plan16_comun import correr, guardar

LAM = 8.42e-5
F_CRUDO, F_JSON, F_TXT = ("resultados/E72_persistencia_fa_crudo.json", "resultados/E72_persistencia_fa.json",
                          "resultados/E72_persistencia_fa.txt")


def condiciones():
    """(nombre, modelo, kw, factor, réplicas con lambda, réplicas con 10 lambda)."""
    ad, am = ARM["adaptado"][1], ARM["amplitud"][1]
    return [("ViSt", "ViSt", {}, 1.0, 16, 0),
            ("CoLd", "CoLd", {}, 1.0, 16, 8),
            ("oraculo", "WiSt", dict(ARM["oraculo"][1]), 1.0, 16, 8),
            ("adaptado_pi0.173", "WiSt", {**ad, "pi0": 0.173}, 1.0, 16, 8),
            ("adaptado_pi0.05", "WiSt", {**ad, "pi0": 0.05}, 1.0, 16, 8),
            ("amplitud_pi0.173", "WiSt", {**am, "pi0": 0.173}, 1.0, 16, 8),
            ("inst_m6_pi0.173", "WiSt", {**am}, 6.0, 16, 0),
            ("inst_m12_pi0.173", "WiSt", {**am}, 12.0, 16, 8)]


def trabajos():
    out = []
    for nom, model, kw, fac, n1, n10 in condiciones():
        for mult, nrep in ((1, n1), (10, n10)):
            for rep in range(nrep):
                out.append({"nom": nom, "model": model, "kw": kw, "factor": fac, "lam": LAM * mult,
                            "mult": mult, "rep": rep})
    out.sort(key=lambda a: a["model"] != "WiSt")
    return out


def clave(x):
    return f'{x["nom"]}|{x["mult"]}|{x["rep"]}'


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "lambda_fp": a["lam"], "tau_W": 4.0, "seed": SEM_P + a["rep"]})
    t0 = time.time()
    flocking.Flock = clase(a["factor"], None)
    try:
        f = flocking.Flock(p, a["model"])
    finally:
        flocking.Flock = Flock
    for _ in range(int(round(T_BURN / p.dt))):
        f.step()
    n0 = f.n_fp
    every = int(round(DT_S / p.dt))
    n_s = int(round(T_P / DT_S))
    V = f.V()
    prev = unw = float(np.arctan2(V[1], V[0]))
    phi = np.empty(n_s + 1)
    phi[0] = unw
    pol, coh_min = [], 1.0
    for s in range(n_s):
        for _ in range(every):
            f.step()
        V = f.V()
        a_ = float(np.arctan2(V[1], V[0]))
        unw += float(np.angle(np.exp(1j * (a_ - prev))))
        prev = a_
        phi[s + 1] = unw
        if s % 20 == 19:
            pol.append(float(f.polarization()))
            coh_min = min(coh_min, float(f.cohesion()["frac_max"]))
    return {"nom": a["nom"], "mult": a["mult"], "rep": a["rep"], "phi": (phi - phi[0]).astype(np.float32).tolist(),
            "n_fp": int(f.n_fp - n0), "pol": float(np.mean(pol)), "coh_min": coh_min, "seg": time.time() - t0}


def arr(xs):
    return np.array([np.asarray(x["phi"], dtype=float) for x in sorted(xs, key=lambda y: y["rep"])])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = trabajos()
    res, nuevos, seg = correr(jobs, F_CRUDO + ".parcial", "E72", job, clave)
    json.dump({"P": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    e64c = json.load(open("resultados/E64_persistencia_crudo.json"))["P"]
    Av = arr([x for x in e64c if x["nom"] == "ViSt"])                 # ViSt sin falsas alarmas, 16 réplicas
    e64 = json.load(open("resultados/E64_persistencia.json"))
    out = {"P": {}, "c_P": {}, "R_E64": {}}
    L = [f"=== E72 · persistencia con falsas alarmas (N = 250, k = 15, tau_W = 4 s, {T_P:g} s) ===",
         f"lambda_FP = {LAM:.3g} s^-1 por agente (x1, 16 réplicas) y x10 (8); P/P_ViSt contra ViSt sin falsas alarmas (E64)"]
    for mult in (1, 10):
        S = {}
        for x in res:
            if x["mult"] == mult:
                S.setdefault(x["nom"], []).append(x)
        if "oraculo" not in S:
            continue
        A_or = arr(S["oraculo"])
        L.append(f"\n--- lambda x{mult} ---")
        for nom, xs in S.items():
            A = arr(xs)
            r, ic = boot_cociente(A, Av, D_incr)                      # D_ViSt / D_x
            ro, ico = boot_cociente(A, A_or, D_incr)                  # D_or / D_x  (= P_x / P_or)
            fila = {"n_rep": len(xs), "r": r, "r_ic": ic, "P_sobre_Por": ro, "P_sobre_Por_ic": ico,
                    "D": D_incr(A), "n_fp": float(np.mean([x["n_fp"] for x in xs])),
                    "pol": float(np.mean([x["pol"] for x in xs])), "coh_min": float(np.min([x["coh_min"] for x in xs]))}
            out["P"][f"x{mult}|{nom}"] = fila
            L.append(f"{nom:20s} n {len(xs):2d}  P/P_ViSt {r:.3f} [{ic[0]:.3f}; {ic[1]:.3f}]  P/P_or {ro:.3f} "
                     f"[{ico[0]:.3f}; {ico[1]:.3f}]  falsas alarmas {fila['n_fp']:.1f}  pol {fila['pol']:.3f}  "
                     f"cohesión mín {fila['coh_min']:.2f}")
        r_or = out["P"][f"x{mult}|oraculo"]["r"]
        for nom in S:
            if nom in ("ViSt", "oraculo"):
                continue
            rx = out["P"][f"x{mult}|{nom}"]["r"]
            cP = (1 - rx) / (1 - r_or) if r_or < 1 else float("nan")
            rec = {c: e64["R"][c][nom]["rec"] for c in e64["R"] if nom in e64["R"][c]}
            out["c_P"][f"x{mult}|{nom}"] = {"c_P": cP, "rec_E64": rec}
            L.append(f"  {nom:20s} c_P = {cP:.3f}  | rec (E64, un líder): "
                     + "  ".join(f"{c} {v:.3f}" for c, v in rec.items()))
    P, C = out["P"], out["c_P"]
    rf = lambda nom: C[f"x1|{nom}"]["rec_E64"]["forzado|4"]
    out["pred"] = {
        "P4": {"r_or": P["x1|oraculo"]["r"], "ic": P["x1|oraculo"]["r_ic"], "cumple": bool(P["x1|oraculo"]["r"] < 0.8)},
        "P5": {"r_int": P["x1|adaptado_pi0.173"]["r"], "P_int/P_or": P["x1|adaptado_pi0.173"]["P_sobre_Por"],
               "ic": P["x1|adaptado_pi0.173"]["P_sobre_Por_ic"],
               "cumple": bool(P["x1|adaptado_pi0.173"]["r"] >= P["x1|oraculo"]["r"])},
        "P6": {"c_P_0.173": C["x1|adaptado_pi0.173"]["c_P"], "rec_0.173": rf("adaptado_pi0.173"),
               "c_P_0.05": C["x1|adaptado_pi0.05"]["c_P"], "rec_0.05": rf("adaptado_pi0.05"),
               "cumple": bool(C["x1|adaptado_pi0.173"]["c_P"] <= rf("adaptado_pi0.173")
                              and C["x1|adaptado_pi0.05"]["c_P"] <= rf("adaptado_pi0.05"))},
        "P7": {"m12": C["x1|inst_m12_pi0.173"]["c_P"] / rf("inst_m12_pi0.173"),
               "int": C["x1|adaptado_pi0.173"]["c_P"] / rf("adaptado_pi0.173"),
               "cumple": bool(C["x1|inst_m12_pi0.173"]["c_P"] / rf("inst_m12_pi0.173")
                              > C["x1|adaptado_pi0.173"]["c_P"] / rf("adaptado_pi0.173"))},
        "control_ViSt": {"r": P["x1|ViSt"]["r"], "ic": P["x1|ViSt"]["r_ic"],
                         "cumple": bool(P["x1|ViSt"]["r_ic"][0] <= 1 <= P["x1|ViSt"]["r_ic"][1])}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v, default=float)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    guardar(out, L, F_JSON, F_TXT)
