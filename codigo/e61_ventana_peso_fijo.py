"""Etapa 61 (plan 14, cierre de E58): ventana óptima con el peso de la evidencia fijo.

Celdas de E55 (a_max 5, 20, 40) y de E56 (ruido x0,5 y x2), mismo observador calibrado, pero el
log-cociente se multiplica por 12 / n_ef(tau) para que todas las ventanas pesen la evidencia
como la de 1 s (n_ef = 12). Diseño y predicciones G1-G3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e61_ventana_peso_fijo.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params, run_response, phi_grid
from parametros import P0
from fase4 import D_PHI, SEMILLA_R, WINS, W, _w, trabajo, clave, pareado, limpio

S2 = float(np.sqrt(2.0))
AS, FS = (5.0, 20.0, 40.0), (0.5, 2.0)
VENT = (0.25, 0.5, 1.0, 2.0, 4.0)
N_REP, TAU_W = 4, 4.0
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))
F_CRUDO, F_JSON, F_TXT = ("resultados/E61_ventana_peso_fijo_crudo.json", "resultados/E61_ventana_peso_fijo.json",
                          "resultados/E61_ventana_peso_fijo.txt")


class FlockGanancia(Flock):
    factor = 1.0

    def _llr_kinematic(self, idx):
        return self.factor * super()._llr_kinematic(idx)


def n_ef(tau):
    return max(tau / P0.tau_obs, 1.0)


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    flocking.Flock = type("FG", (FlockGanancia,), {"factor": float(a["factor"])})
    t0 = time.time()
    try:
        r = run_response(p, a["model"], a["phi"], a["T"], every_coh=100, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "seg": time.time() - t0}


def trabajos(anch55, anch49):
    phis = phi_grid(D_PHI)
    cals = {}
    for a in AS:
        cals[f"a{a:g}"] = dict(a_max=a, sigma_theta=anch55[str(a)], sigma_W=0.811 * a / 10.0)
    for f in FS:
        cals[f"f{f:g}"] = dict(k=15, sigma_xi=f * S2, sigma_theta=anch49[f"k15|f{f:g}"]["sigma"])
    out = []
    for celda, cal in cals.items():
        for etq, pulso, ref in PROT:
            for v in VENT:
                if v == 1.0:
                    continue
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        x = trabajo(celda, f"ciclo{v:g}_n12", 250, TAU_W, pulso, i, phi, rep,
                                    kw={**cal, "tau_det": v}, base="adaptado")
                        x["factor"] = n_ef(1.0) / n_ef(v)
                        out.append(x)
        # ancla: factor 1 con la ventana de 0,5 s, réplica 0, dos ángulos (debe dar E55 / E56)
        for etq, pulso, ref in PROT:
            for i in (4, 13):
                x = trabajo("ancla_" + celda, "ciclo0.5", 250, TAU_W, pulso, i, phis[i], 0,
                            kw={**cal, "tau_det": 0.5}, base="adaptado")
                x["factor"] = 1.0
                out.append(x)
    return out


def correr(jobs, f_parcial):
    hechos = {}
    if os.path.exists(f_parcial):
        for x in json.load(open(f_parcial)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E61] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for i, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or i == len(futs):
                json.dump(list(hechos.values()), open(f_parcial, "w"))
                ult = time.time()
                print(f"[E61] {i}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    return [hechos[clave(a)] for a in jobs if clave(a) in hechos], time.time() - t0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    anch55 = json.load(open("resultados/E55_agilidad.json"))["anchuras"]
    anch49 = json.load(open("resultados/E49_sensibilidad_anchuras.json"))
    res, seg = correr(trabajos(anch55, anch49), F_CRUDO + ".parcial")
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    prev = json.load(open("resultados/E55_agilidad_crudo.json"))["resp"]
    prev += json.load(open("resultados/E56_ventana_ruido_crudo.json"))["resp"]
    idx = {(x["celda"], x["nom"], x["pulso"], x["i_phi"], x["rep"]): x for x in prev}
    dm, n = 0.0, 0
    for x in res:
        if x["celda"].startswith("ancla_"):
            y = idx.get((x["celda"][6:], x["nom"], x["pulso"], x["i_phi"], x["rep"]))
            if y:
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    todo = [x for x in res if not x["celda"].startswith("ancla_")]
    todo += [x for x in prev if x["nom"] in ("ViSt", "CoLd", "oraculo", "amplitud", "ciclo1")]
    e58 = json.load(open("resultados/E58_memoria_ganancia.json"))["rec"]
    L = ["=== E61 · ventana óptima con el peso de la evidencia fijo (n_ef = 12; N = 250, tau_W = 4 s, 4 réplicas) ===",
         f"ancla (factor 1, ventana 0,5 s) frente a E55 y E56: |Δ| máx = {dm:.2e} en {n} comparaciones"]
    out = {"ancla": {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)},
           "rec": {}, "mejor": {}, "banda": {}}
    conds = [f"a{a:g}" for a in AS] + [f"f{f:g}" for f in FS]
    for celda in conds + ["a10"]:
        for etq, pulso, ref in PROT:
            if celda == "a10":
                r58 = e58[etq]
                vs = {v: (r58[f"ciclo{v:g}_n12"] if v != 1.0 else r58["ciclo1_nef12"])["rec"] for v in VENT}
                fila = {"fuente": "E58"}
            else:
                fila = {(f"ciclo{v:g}_n12" if v != 1.0 else "ciclo1"):
                        limpio(pareado(todo, celda, (f"ciclo{v:g}_n12" if v != 1.0 else "ciclo1"), ref, 250, pulso, 15.8, TAU_W))
                        for v in VENT}
                vs = {v: fila[f"ciclo{v:g}_n12" if v != 1.0 else "ciclo1"]["rec"] for v in VENT}
            mejor = max(vs, key=vs.get)
            banda = [v for v in VENT if vs[v] >= 0.9 * vs[mejor]]
            out["rec"][f"{etq}|{celda}"] = {**fila, "serie": {str(v): vs[v] for v in VENT}}
            out["mejor"][f"{etq}|{celda}"] = {"ventana": mejor, "rec": vs[mejor]}
            out["banda"][f"{etq}|{celda}"] = banda
            L.append(f"{celda:5s} {etq:9s}: " + " | ".join(f"{v:g} s {vs[v]:.3f}" for v in VENT)
                     + f" | mejor {mejor:g} s | banda ≥90 %: {banda}")
    m, b = out["mejor"], out["banda"]
    out["pred"] = {
        "G1": {"a5": [m[f"{e}|a5"]["ventana"] for e, _, _ in PROT], "a40": [m[f"{e}|a40"]["ventana"] for e, _, _ in PROT],
               "cumple": bool(all(m[f"{e}|a5"]["ventana"] >= 2 and m[f"{e}|a40"]["ventana"] <= 0.5 for e, _, _ in PROT))},
        "G2": {"f0.5": [m[f"{e}|f0.5"]["ventana"] for e, _, _ in PROT], "f2": [m[f"{e}|f2"]["ventana"] for e, _, _ in PROT],
               "cumple": bool(all(m[f"{e}|f2"]["ventana"] >= m[f"{e}|f0.5"]["ventana"] for e, _, _ in PROT))},
        "G3": {"celdas_con_banda_2_o_mas": int(sum(len(b[f"{e}|{c}"]) >= 2 for c in conds for e, _, _ in PROT)),
               "de": 2 * len(conds),
               "cumple": bool(sum(len(b[f"{e}|{c}"]) >= 2 for c in conds for e, _, _ in PROT) >= 6)}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
