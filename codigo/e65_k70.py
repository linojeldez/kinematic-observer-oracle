"""Etapa 65 (plan 15, punto 3): la recuperación en la celda donde el motor coincide con la
referencia publicada (k = 70).

Con sigma^2 = 1 el motor queda a +1 % (ViSt) y -1 % (WiSt) de los valores publicados con
k = 70, y a -22 % / -15 % con k = 15. Si la recuperación con k = 70 es la misma que con
k = 30 (y 15), el déficit sin causa del motor con k chico no puede mover las conclusiones.

Tres celdas de referencia (N = 250, sigma^2 = 2): líder forzado tau_W = 4 s; maniobra única
tau_W = 4 y 1 s. Observador calibrado a la anchura del residuo de la celda (medida como en
E49) con 16 réplicas, y con la calibración fija (0,361) con 8. Se agrega la maniobra con
tau_W = 4 s y k = 30, que E49 no midió (8 réplicas). Diseño y predicciones Q1-Q2 en
docs/bitacora/resultados_E56-E65.md («Plan 15 · E65»).

    PYTHONPATH=codigo python codigo/e65_k70.py
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from flocking import phi_grid
from parametros import P0
from fase4 import D_PHI, T_N, W, trabajo, correr, clave, crudos_previos, pareado, limpio, R_rep
from e49_sensibilidad import _anchura

S2 = float(np.sqrt(2.0))
N_REP, N_REP_FIJO, N_REP_K30 = 16, 8, 8
CELDAS = (("forzado", False, 4.0, "ViSt"), ("maniobra", True, 4.0, "CoLd"), ("maniobra", True, 1.0, "CoLd"))
F_CRUDO, F_JSON, F_TXT = "resultados/E65_k70_crudo.json", "resultados/E65_k70.json", "resultados/E65_k70.txt"


def anchura(k):
    with ProcessPoolExecutor(max_workers=W) as ex:
        res = list(ex.map(_anchura, [(k, 1.0, s) for s in range(8)]))
    v = np.array([w for _, _, w, _ in res])
    return {"sigma": round(float(v.mean()), 4), "sd": float(v.std()), "pol": float(np.mean([q for *_, q in res]))}


def trabajos(sig70, sig30):
    phis = phi_grid(D_PHI)
    out = []

    def add(c, nom, tau, pulso, base, kw, nrep):
        for i, phi in enumerate(phis):
            for rep in range(nrep):
                out.append(trabajo(c, nom, 250, tau, pulso, i, phi, rep, kw=kw, base=base))

    kw70 = dict(k=70, sigma_xi=S2)
    cal70 = dict(kw70, sigma_theta=sig70)
    for etq, pulso, tau, ref in CELDAS:
        add("k70", ref, tau, pulso, ref, kw70, N_REP)
        add("k70", "oraculo", tau, pulso, "oraculo", kw70, N_REP)
        add("k70", "amplitud_cal", tau, pulso, "amplitud", cal70, N_REP)
        add("k70", "adaptado_cal", tau, pulso, "adaptado", cal70, N_REP)
        add("k70", "amplitud_fijo", tau, pulso, "amplitud", kw70, N_REP_FIJO)
        add("k70", "adaptado_fijo", tau, pulso, "adaptado", kw70, N_REP_FIJO)
    kw30 = dict(k=30, sigma_xi=S2)
    cal30 = dict(kw30, sigma_theta=sig30)
    for nom, base, kw in (("CoLd", "CoLd", kw30), ("oraculo", "oraculo", kw30),
                          ("amplitud_cal", "amplitud", cal30), ("adaptado_cal", "adaptado", cal30)):
        add("k30|f1", nom, 4.0, True, base, kw, N_REP_K30)
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    e49 = json.load(open("resultados/E49_sensibilidad.json"))
    sig30 = e49["anchuras"]["k30|f1"]["sigma"]
    f_anch = "resultados/E65_k70_anchura.json"
    if os.path.exists(f_anch):
        an70 = json.load(open(f_anch))
    else:
        an70 = anchura(70)
        json.dump(an70, open(f_anch, "w"), indent=1)
    print(f"[E65] anchura k = 70: {an70['sigma']:.4f} ± {an70['sd']:.4f} (k = 30: {sig30:.4f})", flush=True)
    res, seg = correr(trabajos(an70["sigma"], sig30), F_CRUDO + ".parcial", "E65")
    json.dump({"resp": res, "anchura_k70": an70}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")

    e49c = [x for x in json.load(open("resultados/E49_sensibilidad_crudo.json"))["resp"]]
    e48 = [{**x, "celda": "base"} for x in json.load(open("resultados/E48_replicas_crudo.json"))["resp"]
           if x.get("celda", "base") == "base"]
    todo = list({clave(x): x for x in crudos_previos() + e48}.values()) + e49c + res
    out = {"anchura": {"k70": an70, "k30": sig30, "k15": P0.sigma_theta}, "rec": {}, "abs": {}}
    L = [f"=== E65 · recuperación con k = 70 (N = 250, sigma^2 = 2) ===",
         f"anchura del residuo: k = 70 {an70['sigma']:.4f}, k = 30 {sig30:.4f}, k = 15 (fija) {P0.sigma_theta:.3f} rad/s"]
    for etq, pulso, tau, ref in CELDAS:
        c = f"{etq}|{tau:g}"
        fila = {}
        for nom in ("adaptado_cal", "amplitud_cal", "adaptado_fijo", "amplitud_fijo"):
            n = N_REP if nom.endswith("cal") else N_REP_FIJO
            fila[f"k70|{nom}"] = limpio(pareado(todo, "k70", nom, ref, 250, pulso, 15.8, tau, reps=list(range(n))))
        for nom in ("adaptado_cal", "amplitud_cal"):
            q = pareado(todo, "k30|f1", nom, ref, 250, pulso, 15.8, tau, reps=list(range(8)))
            if q["n_rep"]:
                fila[f"k30|{nom}"] = limpio(q)
        for nom, nb in (("adaptado", "adaptado"), ("amplitud", "amplitud")):
            fila[f"k15|{nom}"] = limpio(pareado(todo, "base", nb, ref, 250, pulso, 15.8, tau, reps=list(range(32))))
        out["rec"][c] = fila
        iv, Rv = R_rep(todo, "k70", ref, 250, pulso, 15.8, tau, list(range(N_REP)))
        io, Ro = R_rep(todo, "k70", "oraculo", 250, pulso, 15.8, tau, list(range(N_REP)))
        ia, Ra = R_rep(todo, "k70", "adaptado_cal", 250, pulso, 15.8, tau, list(range(N_REP)))
        out["abs"][c] = {"R_ref": float(np.mean(Rv)), "R_oraculo": float(np.mean(Ro)), "R_adaptado": float(np.mean(Ra)),
                         "ventaja": float(np.mean(Ro) - np.mean(Rv))}
        L.append(f"\n--- {c}: R {ref} {np.mean(Rv):.3f}, oráculo {np.mean(Ro):.3f}, integrador {np.mean(Ra):.3f} (k = 70) ---")
        for k_, q in fila.items():
            L.append(f"  {k_:20s} {q['rec']:.3f} [{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}]  n = {q['n_rep']}")
    R = out["rec"]
    pares = [(c, R[c]["k70|adaptado_cal"]["rec"], R[c]["k30|adaptado_cal"]["rec"]) for c in R if "k30|adaptado_cal" in R[c]]
    out["pred"] = {
        "Q1": {"pares": pares, "cumple": bool(pares and all(abs(a - b) <= 0.10 for _, a, b in pares))},
        "Q2": {"sin_memoria": [R[c]["k70|amplitud_cal"]["rec"] for c in R],
               "cumple": bool(all(R[c]["k70|amplitud_cal"]["rec"] < 0.25 for c in R))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
