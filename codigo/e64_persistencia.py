"""Etapa 64 (plan 15, punto 1): persistencia del observador sin falsas alarmas.

Sin informados, el oráculo es ViSt por construcción (todos los w valen cero). El observador
no: su posterior se mueve con el ruido del residuo y puede dar peso a fluctuaciones. Aquí se
mide P = D_ref / D_Phi con lambda_fp = 0 (sin falsas alarmas: eso es la credulidad, Paper B)
para cada observador y prior, y su responsividad en las tres celdas de referencia, para
dibujar la curva (R(pi0), P(pi0)) contra el punto del oráculo (R_oráculo, P_ViSt).

Dos estimadores de D_Phi sobre la misma trayectoria del rumbo global Phi(t):
  * «balazs»: <(Phi(t) - Phi(0))^2> sobre réplicas, ajuste por el origen en la mitad final
    (el de `flocking.angular_diffusion`, el del paper de referencia);
  * «incr»: <(Phi(t + L) - Phi(t))^2> / L promediado sobre todos los orígenes t y réplicas
    (L = 10 s por defecto; se guardan L = 2, 5, 10, 20, 50 s para ver el régimen difusivo).
El cociente P_obs / P_ViSt = D_ViSt / D_obs no depende de la referencia de campo medio.

Diseño y predicciones P1-P3 en docs/bitacora/resultados_E56-E65.md («Plan 15 · E64»).

    PYTHONPATH=codigo python codigo/e64_persistencia.py [piloto]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params, run_response, phi_grid
from parametros import P0
from fase4 import (D_PHI, SEMILLA_R, WINS, _w, ARM, T_N, FlockAuto, clave, crudos_previos, R_rep,
                   rec_ic, limpio)

W = int(os.environ.get("FASE4_W", "9"))
T_P, T_BURN, DT_S = 600.0, 20.0, 0.5
LAGS = (2.0, 5.0, 10.0, 20.0, 50.0)
L_MAIN = 10.0
SEM_P = 11                       # semilla de persistencia: 11 + réplica (E5, E22)
PIS = (0.004, 0.01, 0.05, 0.173, 0.3)
CELDAS = (("forzado", False, "ViSt", 4.0), ("maniobra", True, "CoLd", 4.0), ("maniobra", True, "CoLd", 1.0))
N_REP_R = 8
F_CRUDO, F_JSON, F_TXT = ("resultados/E64_persistencia_crudo.json", "resultados/E64_persistencia.json",
                          "resultados/E64_persistencia.txt")


class FlockG(Flock):
    """Observador con el log-cociente multiplicado por un factor (E58) y, si tau_base,
    con suelo de ruido estimado localmente (FlockAuto, E49)."""
    factor = 1.0

    def _llr_kinematic(self, idx):
        return self.factor * super()._llr_kinematic(idx)


def clase(factor=1.0, tau_base=None):
    if tau_base:
        return type("FAG", (FlockAuto,), {"tau_base": float(tau_base),
                                          "_llr_kinematic": lambda s, idx: factor * FlockAuto._llr_kinematic(s, idx)})
    return type("FG", (FlockG,), {"factor": float(factor)})


def condiciones():
    """(nombre, modelo, kw, factor, tau_base, n_rep de persistencia)."""
    out = [("ref_campo_medio", "ViSt", {"k": P0.N - 1}, 1.0, None, 16),
           ("ViSt", "ViSt", {}, 1.0, None, 16)]
    ad, am = ARM["adaptado"][1], ARM["amplitud"][1]
    for p in PIS:
        out.append((f"adaptado_pi{p:g}", "WiSt", {**ad, "pi0": p}, 1.0, None, 16))
        out.append((f"amplitud_pi{p:g}", "WiSt", {**am, "pi0": p}, 1.0, None, 16))
    for p in (0.004, 0.173):
        out.append((f"secuencial_pi{p:g}", "WiSt", {**ad, "detector": "secuencial", "pi0": p}, 1.0, None, 8))
        out.append((f"auto_pi{p:g}", "WiSt", {**ad, "pi0": p}, 1.0, 5.0, 8))
    out.append(("inst_m6_pi0.173", "WiSt", {**am}, 6.0, None, 8))
    out.append(("inst_m12_pi0.173", "WiSt", {**am}, 12.0, None, 8))
    return out


def job_P(a):
    p = Params(**{**asdict(P0), **a["kw"], "lambda_fp": 0.0, "tau_W": 4.0, "seed": SEM_P + a["rep"]})
    t0 = time.time()
    flocking.Flock = clase(a["factor"], a["tau_base"])
    try:
        f = flocking.Flock(p, a["model"])
    finally:
        flocking.Flock = Flock
    for _ in range(int(round(T_BURN / p.dt))):
        f.step()
    every = int(round(DT_S / p.dt))
    n_s = int(round(T_P / DT_S))
    V = f.V()
    prev = unw = float(np.arctan2(V[1], V[0]))
    phi = np.empty(n_s + 1)
    phi[0] = unw
    att, fpr, pol = [], [], []
    for s in range(n_s):
        for _ in range(every):
            f.step()
        V = f.V()
        a_ = float(np.arctan2(V[1], V[0]))
        unw += float(np.angle(np.exp(1j * (a_ - prev))))
        prev = a_
        phi[s + 1] = unw
        if a["model"] == "WiSt" and s % 2 == 1:          # cada 1 s: pesos sobre aristas sin autoenlace
            idx, dist = f._neighbors()
            w = f.w_hat(idx, dist)[:, 1:]
            att.append(float(w.mean()))
            fpr.append(float((w > 0.5).mean()))
        if s % 20 == 19:
            pol.append(float(f.polarization()))
    return {"nom": a["nom"], "rep": a["rep"], "phi": (phi - phi[0]).astype(np.float32).tolist(),
            "att": float(np.mean(att)) if att else 0.0, "fpr": float(np.mean(fpr)) if fpr else 0.0,
            "fpr_max": float(np.max(fpr)) if fpr else 0.0,
            "pol": float(np.mean(pol)), "coh": float(f.cohesion()["frac_max"]), "seg": time.time() - t0}


def job_R(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    t0 = time.time()
    flocking.Flock = clase(a["factor"], a["tau_base"])
    try:
        r = run_response(p, a["model"], a["phi"], a["T"], every_coh=100, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"], "pol": r["pol"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "seg": time.time() - t0}


def job(a):
    return job_P(a) if a["tipo"] == "P" else job_R(a)


def clave_j(x):
    if x.get("tipo", "R" if "i_phi" in x else "P") == "P" and "i_phi" not in x:
        return f'P|{x["nom"]}|{x["rep"]}'
    return "R|" + clave(x)


# R ya guardada con las mismas semillas: nombre E64 -> (archivo, nombre allí)
R_PREVIA = {"adaptado_pi0.173": "adaptado", "amplitud_pi0.173": "amplitud",
            "adaptado_pi0.004": "pi0.004", "adaptado_pi0.05": "pi0.05"}


def trabajos(piloto=False):
    phis = phi_grid(D_PHI)
    out = []
    for nom, model, kw, fac, tb, nrep in condiciones():
        for rep in range(2 if piloto else nrep):
            out.append({"tipo": "P", "nom": nom, "model": model, "kw": kw, "factor": fac, "tau_base": tb, "rep": rep})
    if piloto:
        return out
    for nom, model, kw, fac, tb, nrep in condiciones():
        if model != "WiSt" or nom in R_PREVIA:
            continue
        for etq, pulso, ref, tw in CELDAS:
            for i, phi in enumerate(phis):
                for rep in range(N_REP_R):
                    out.append({"tipo": "R", "celda": "base", "nom": nom, "model": model, "kw": kw, "factor": fac,
                                "tau_base": tb, "N": 250, "tau_W": tw, "pulso": pulso, "i_phi": i,
                                "phi": float(phi), "rep": rep, "T": T_N[250]})
    return out


# ---------------------------------------------------------------- análisis

def D_balazs(phis):
    t = np.arange(1, phis.shape[1]) * DT_S
    msd = (phis[:, 1:] ** 2).mean(axis=0)
    m = t > t[-1] / 2
    return float((t[m] @ msd[m]) / (t[m] @ t[m]))


def D_incr(phis, L=L_MAIN):
    j = int(round(L / DT_S))
    d = phis[:, j:] - phis[:, :-j]
    return float((d ** 2).mean() / L)


def boot_cociente(A, B, fn, n_boot=2000, seed=3):
    """fn(B)/fn(A) con IC remuestreando réplicas PAREADAS (las semillas son las mismas: 11 + réplica).
    Ambas series vienen ordenadas por réplica; se usan las n primeras de cada una."""
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    rng = np.random.default_rng(seed)
    v = fn(B) / fn(A)
    bs = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        bs.append(fn(B[b]) / fn(A[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(v), [float(lo), float(hi)]


def analizar(resP, resR):
    S = {}
    for x in resP:
        S.setdefault(x["nom"], []).append(x)
    arr = {k: np.array([np.asarray(x["phi"], dtype=float) for x in sorted(v, key=lambda y: y["rep"])])
           for k, v in S.items()}
    out = {"P": {}, "R": {}, "lags": {}}
    Dref_b, Dref_i = D_balazs(arr["ref_campo_medio"]), D_incr(arr["ref_campo_medio"])
    Dv = arr["ViSt"]
    for k, A in arr.items():
        fila = {"n_rep": int(len(A)), "D_balazs": D_balazs(A), "D_incr": D_incr(A),
                "P_balazs": Dref_b / D_balazs(A), "P_incr": Dref_i / D_incr(A),
                "att": float(np.mean([x["att"] for x in S[k]])), "fpr": float(np.mean([x["fpr"] for x in S[k]])),
                "pol": float(np.mean([x["pol"] for x in S[k]])), "coh": float(np.mean([x["coh"] for x in S[k]])),
                "seg": float(np.mean([x["seg"] for x in S[k]]))}
        # P_obs / P_ViSt = D_ViSt / D_obs
        fila["ratio_incr"], fila["ratio_incr_ic"] = boot_cociente(A, Dv, D_incr)
        fila["ratio_balazs"], fila["ratio_balazs_ic"] = boot_cociente(A, Dv, D_balazs)
        out["P"][k] = fila
        out["lags"][k] = {f"{L:g}": D_incr(A, L) for L in LAGS}
    # responsividad: E64 + corridas previas con las mismas semillas
    prev = crudos_previos() + [{**x, "celda": "base"} for x in json.load(open("resultados/E62_prior_crudo.json"))["resp"]]
    todo = list({clave(x): x for x in prev}.values()) + resR
    for etq, pulso, ref, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        iv, Rv = R_rep(todo, "base", ref, 250, pulso, 15.8, tw, list(range(N_REP_R)))
        io, Ro = R_rep(todo, "base", "oraculo", 250, pulso, 15.8, tw, list(range(N_REP_R)))
        out["R"][c] = {"_ref": {"R_ref": float(np.mean(Rv)), "R_oraculo": float(np.mean(Ro))}}
        for nom, model, kw, fac, tb, nrep in condiciones():
            if model != "WiSt":
                continue
            im, Rm = R_rep(todo, "base", R_PREVIA.get(nom, nom), 250, pulso, 15.8, tw, list(range(N_REP_R)))
            com = sorted(set(im) & set(iv) & set(io))
            if not com:
                continue
            g = lambda ids, R: np.array([R[ids.index(k)] for k in com])
            q = limpio(rec_ic(g(im, Rm), g(iv, Rv), g(io, Ro)))
            q["R"] = float(np.mean(g(im, Rm)))
            out["R"][c][nom] = q
    return out


def texto(out, seg):
    P, R = out["P"], out["R"]
    L = ["=== E64 · persistencia del observador sin falsas alarmas (N = 250, k = 15, ruido de referencia) ===",
         f"t_P = {T_P:g} s tras {T_BURN:g} s; D por incrementos a L = {L_MAIN:g} s (y por el origen, Balázs)",
         f"P_ViSt = {P['ViSt']['P_incr']:.3f} (incr) / {P['ViSt']['P_balazs']:.3f} (Balázs)", "",
         f"{'condición':22s} {'n':>3s} {'P/P_ViSt incr [IC]':>26s} {'Balázs [IC]':>24s} {'atención':>9s} {'FPR':>7s}"]
    for k, f in P.items():
        L.append(f"{k:22s} {f['n_rep']:3d} {f['ratio_incr']:7.3f} [{f['ratio_incr_ic'][0]:.3f}; {f['ratio_incr_ic'][1]:.3f}]"
                 f"     {f['ratio_balazs']:6.3f} [{f['ratio_balazs_ic'][0]:.3f}; {f['ratio_balazs_ic'][1]:.3f}]"
                 f" {f['att']:9.4f} {f['fpr']:7.4f}")
    L.append("\nD por incrementos según el retardo L (régimen difusivo si es plano):")
    for k, d in out["lags"].items():
        L.append(f"  {k:22s} " + "  ".join(f"{l}: {v:.2e}" for l, v in d.items()))
    for c, fila in R.items():
        L.append(f"\n--- responsividad, {c} (R ref {fila['_ref']['R_ref']:.3f}, oráculo {fila['_ref']['R_oraculo']:.3f}) ---")
        for k, q in fila.items():
            if k.startswith("_"):
                continue
            L.append(f"  {k:22s} R {q['R']:.3f}  rec {q['rec']:.3f} [{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}]  "
                     f"P/P_ViSt {P[k]['ratio_incr']:.3f}")
    if "pred" in out:
        L.append("\n=== Predicciones ===")
        L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    return "\n".join(L)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    piloto = len(sys.argv) > 1 and sys.argv[1] == "piloto"
    jobs = trabajos(piloto)
    f_par = (F_CRUDO + ".piloto" if piloto else F_CRUDO) + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave_j(x)] = x
    faltan = [a for a in jobs if clave_j(a) not in hechos]
    print(f"[E64] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for k, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave_j(x)] = x
            if time.time() - ult > 120 or k == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w"))
                ult = time.time()
                print(f"[E64] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    seg = time.time() - t0
    res = [hechos[clave_j(a)] for a in jobs]
    resP = [x for x in res if "i_phi" not in x]
    resR = [x for x in res if "i_phi" in x]
    if piloto:
        for x in resP:
            print(f'{x["nom"]:22s} rep {x["rep"]}  {x["seg"]:.0f} s  att {x["att"]:.4f}  fpr {x["fpr"]:.4f} '
                  f'pol {x["pol"]:.3f}  Dincr {D_incr(np.array([x["phi"]], dtype=float)):.2e}')
        sys.exit(0)
    json.dump({"P": resP, "R": resR}, open(F_CRUDO, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    out = analizar(resP, resR)
    P = out["P"]
    out["pred"] = {
        "P1": {"ratio_adaptado_0.173": P["adaptado_pi0.173"]["ratio_incr"],
               "ic": P["adaptado_pi0.173"]["ratio_incr_ic"],
               "cumple": bool(P["adaptado_pi0.173"]["ratio_incr"] >= 0.8)},
        "P2": {"inst_m6": P["inst_m6_pi0.173"]["ratio_incr"], "inst_m12": P["inst_m12_pi0.173"]["ratio_incr"],
               "adaptado": P["adaptado_pi0.173"]["ratio_incr"],
               "cumple": bool(max(P["inst_m6_pi0.173"]["ratio_incr"], P["inst_m12_pi0.173"]["ratio_incr"])
                              < P["adaptado_pi0.173"]["ratio_incr"])},
        "P3": {"serie": [P[f"adaptado_pi{p:g}"]["ratio_incr"] for p in PIS],
               "cumple": bool(all(np.diff([P[f"adaptado_pi{p:g}"]["ratio_incr"] for p in PIS]) <= 0.02))}}
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write(texto(out, seg) + "\n")
    print(texto(out, seg))
