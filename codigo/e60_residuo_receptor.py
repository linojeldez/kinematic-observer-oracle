"""Etapa 60 (plan 14, objeción mayor 2): residuo calculable desde lo que ve el receptor, e
identificación del estado informado.

En el motor, el posterior sobre j usa el residuo de j frente a SU vecindario y es el mismo para
todos los observadores. Aquí el ciclo de trabajo se lleva por par (i, j) en una matriz N x N,
actualizado sólo mientras j está entre los vecinos de i (si no, se relaja hacia q0):
  emisor      el del motor (ancla; sólo se registra la identificación)
  par_propio  residuo de j frente a su vecindario, memoria por par
  par_local   residuo de j frente al vecindario de i, eps_ij = w_j - <w>_{N_i}, memoria por par,
              anchura calibrada en vuelo sin informados
Diseño y predicciones L1-L4 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e60_residuo_receptor.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from scipy.stats import t as _t

import flocking
from flocking import Flock, Params, run_response, phi_grid, _t_scale_from_mad
from parametros import P0
from fase4 import D_PHI, SEMILLA_R, WINS, W, _w, trabajo, clave, crudos_previos, pareado, limpio

N_REP = 8
CELDAS = (("forzado", False, "ViSt", 4.0), ("maniobra", True, "CoLd", 4.0), ("maniobra", True, "CoLd", 1.0))
MODOS = ("emisor", "par_propio", "par_local")
T_ACT = 3.0                      # run_response asienta 3 s antes de informar al líder
F_CRUDO, F_JSON, F_TXT = ("resultados/E60_residuo_receptor_crudo.json", "resultados/E60_residuo_receptor.json",
                          "resultados/E60_residuo_receptor.txt")
ULTIMO = {}


class FlockPar(Flock):
    modo = "emisor"
    sig_local = None

    def __init__(self, p, model):
        super().__init__(p, model)
        self._D = None
        self._par_paso = -1
        self._ult = None
        self.reg_t, self.reg_w, self.reg_i = [], [], []
        ULTIMO["f"] = self

    def _q(self, s0):
        p = self.p
        c = _t_scale_from_mad(p.nu)
        s1 = float(np.hypot(s0, p.sigma_W))
        um = p.c_det * s0
        q0 = float(np.clip(2.0 * _t.sf(um / (s0 * c), p.nu), 1e-6, 1 - 1e-6))
        q1 = float(np.clip(2.0 * _t.sf(um / (s1 * c), p.nu), 1e-6, 1 - 1e-6))
        return q0, q1

    def w_hat(self, idx, dist=None):
        if self.modo == "emisor":
            w = super().w_hat(idx, dist)
        else:
            w = self._w_par(idx)
        self._ult = (idx, w)
        return w

    def _w_par(self, idx):
        p = self.p
        N = p.N
        if self.modo == "par_propio":
            eps = self.omega - self.omega[idx].mean(axis=1)
            E = eps[idx]
            s0 = p.sigma_theta
        else:
            E = self.omega[idx] - self.omega[idx].mean(axis=1)[:, None]
            s0 = float(self.sig_local)
        q0, q1 = self._q(s0)
        filas = np.broadcast_to(np.arange(N)[:, None], idx.shape)
        if self._D is None:
            self._D = np.full((N, N), q0)
        if self._par_paso != self._paso:
            a = np.exp(-p.dt / max(p.tau_det, 1e-6))
            viejo = self._D[filas, idx]
            self._D = a * self._D + (1.0 - a) * q0
            self._D[filas, idx] = a * viejo + (1.0 - a) * (np.abs(E) > p.c_det * s0)
            self._par_paso = self._paso
        d = self._D[filas, idx]
        n_ef = max(p.tau_det / max(p.tau_obs, 1e-6), 1.0)
        llr = n_ef * (d * np.log(q1 / q0) + (1.0 - d) * np.log((1 - q1) / (1 - q0)))
        logit = llr + np.log(p.pi0 / (1.0 - p.pi0))
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))

    def desired_direction(self):
        out = super().desired_direction()
        if self._ult is not None and self._paso % 10 == 0:
            idx, w = self._ult
            N = self.p.N
            m = idx != np.arange(N)[:, None]
            tot = np.bincount(idx[m], weights=w[m], minlength=N)
            cnt = np.bincount(idx[m], minlength=N)
            self.reg_t.append(self.t)
            self.reg_w.append((tot / np.maximum(cnt, 1)).astype(np.float32))
            self.reg_i.append(self.will > 0.5)
        return out


def auc(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    if len(x1) == 0 or len(x0) == 0:
        return float("nan")
    from scipy.stats import rankdata                 # rangos con empates promediados
    r = rankdata(np.concatenate([x1, x0]))
    return float((r[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0)))


def identificacion(f, pulso):
    t = np.array(f.reg_t)
    Wm = np.array(f.reg_w)
    I = np.array(f.reg_i)
    L = f.leader
    post = t >= T_ACT - 1e-9
    no = ~I
    fa = float(((Wm > 0.5) & no)[post].sum() / max(no[post].sum(), 1))
    inf = post & I[:, L]
    hit = float((Wm[inf, L] > 0.5).mean()) if inf.any() else float("nan")
    s4 = post & (t < T_ACT + 4.0) & I[:, L]
    x1 = Wm[s4, L]
    x0 = np.concatenate([Wm[s][no[s]] for s in np.flatnonzero(s4)]) if s4.any() else np.array([])
    au = auc(x1, x0)
    pers = float("nan")
    if pulso:
        apag = np.flatnonzero(post & ~I[:, L])
        if len(apag):
            s = apag[0]
            n = 0
            while s + n < len(t) and Wm[s + n, L] > 0.5:
                n += 1
            pers = 0.1 * n
    return {"FA": fa, "HIT": hit, "AUROC4": au, "PERS": pers}


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    flocking.Flock = type("FP", (FlockPar,), {"modo": a["modo"], "sig_local": a["sig_local"]})
    t0 = time.time()
    try:
        r = run_response(p, a["model"], a["phi"], a["T"], every_coh=100, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    return {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"], "pulso": a["pulso"],
            "i_phi": a["i_phi"], "rep": a["rep"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= a["T"] + 1e-9},
            "id": identificacion(ULTIMO["f"], a["pulso"]), "seg": time.time() - t0}


def _anchura(seed):
    p = Params(**{**asdict(P0), "N": 250, "seed": 100 + seed})
    f = Flock(p, "ViSt")
    for _ in range(300):
        f.step()
    loc, emi = [], []
    for s in range(1200):
        f.step()
        if s % 5 == 0:
            idx, _ = f._neighbors()
            E = f.omega[idx] - f.omega[idx].mean(axis=1)[:, None]
            loc.append(np.abs(E[:, 1:]).ravel())
            emi.append(np.abs(f.omega - f.omega[idx].mean(axis=1)))
    return float(1.4826 * np.median(np.concatenate(loc))), float(1.4826 * np.median(np.concatenate(emi)))


def trabajos(sig_local):
    phis = phi_grid(D_PHI)
    out = []
    for etq, pulso, ref, tw in CELDAS:
        for modo in MODOS:
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    x = trabajo("base", f"rr_{modo}", 250, tw, pulso, i, phi, rep, base="adaptado")
                    x["modo"], x["sig_local"] = modo, sig_local
                    out.append(x)
    return out


def correr(jobs, f_parcial):
    hechos = {}
    if os.path.exists(f_parcial):
        for x in json.load(open(f_parcial)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E60] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for i, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or i == len(futs):
                json.dump(list(hechos.values()), open(f_parcial, "w"))
                ult = time.time()
                print(f"[E60] {i}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    return [hechos[clave(a)] for a in jobs if clave(a) in hechos], time.time() - t0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with ProcessPoolExecutor(max_workers=W) as ex:
        an = list(ex.map(_anchura, range(8)))
    sig_local = round(float(np.mean([a for a, _ in an])), 4)
    sig_emisor = round(float(np.mean([b for _, b in an])), 4)
    print(f"anchura del residuo local {sig_local}, del residuo del motor {sig_emisor}", flush=True)
    res, seg = correr(trabajos(sig_local), F_CRUDO + ".parcial")
    json.dump({"resp": res, "sig_local": sig_local, "sig_emisor": sig_emisor}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    prev = crudos_previos()
    prev += json.load(open("resultados/E48_replicas_crudo.json"))["resp"]
    prev = [{**x, "celda": "base"} for x in prev]
    idx = {(x["nom"], x["N"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"]): x for x in prev}
    dm, n = 0.0, 0
    for x in res:
        if x["nom"] == "rr_emisor":
            y = idx.get(("adaptado", 250, x["tau_W"], x["pulso"], x["i_phi"], x["rep"]))
            if y:
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    todo = list({clave(x): x for x in prev}.values()) + res
    L = [f"=== E60 · residuo calculable por el receptor (N = 250, k = 15, ciclo de 1 s, {N_REP} réplicas) ===",
         f"anchura calibrada del residuo local: {sig_local} rad/s (residuo del motor: {sig_emisor})",
         f"ancla L1 (modo emisor frente a E42-E48): |Δ| máx = {dm:.2e} en {n} comparaciones"]
    out = {"sig_local": sig_local, "sig_emisor": sig_emisor,
           "ancla": {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)},
           "rec": {}, "id": {}}
    for etq, pulso, ref, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        out["rec"][c], out["id"][c] = {}, {}
        for modo in MODOS:
            q = limpio(pareado(todo, "base", f"rr_{modo}", ref, 250, pulso, 15.8, tw, reps=list(range(N_REP))))
            out["rec"][c][modo] = q
            ids = [x["id"] for x in res if x["nom"] == f"rr_{modo}" and x["pulso"] == pulso and abs(x["tau_W"] - tw) < 1e-9]
            out["id"][c][modo] = {k: float(np.nanmean([d[k] for d in ids])) for k in ("FA", "HIT", "AUROC4", "PERS")}
        L.append(f"\n--- {etq}, tau_W = {tw:g} s ---")
        for modo in MODOS:
            q, d = out["rec"][c][modo], out["id"][c][modo]
            L.append(f"  {modo:10s}: recuperación {q['rec']:.3f} [{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}] | "
                     f"FA {d['FA']:.4f} | HIT {d['HIT']:.3f} | AUROC(4 s) {d['AUROC4']:.3f} | PERS {d['PERS']:.2f} s")
    R = out["rec"]
    I = out["id"]
    cs = [f"{e}|{tw:g}" for e, _, _, tw in CELDAS]
    out["pred"] = {
        "L1": out["ancla"],
        "L2": {"dif": [R[c]["par_propio"]["rec"] - R[c]["emisor"]["rec"] for c in cs],
               "cumple": bool(all(abs(R[c]["par_propio"]["rec"] - R[c]["emisor"]["rec"]) < 0.05 for c in cs))},
        "L3": {"cociente": [R[c]["par_local"]["rec"] / R[c]["emisor"]["rec"] for c in cs],
               "cumple": bool(all(R[c]["par_local"]["rec"] >= (2 / 3) * R[c]["emisor"]["rec"] for c in cs))},
        "L4": {"HIT_forzado": I["forzado|4"]["emisor"]["HIT"], "AUROC4_min": min(I[c]["emisor"]["AUROC4"] for c in cs),
               "FA_emisor_max": max(I[c]["emisor"]["FA"] for c in cs), "FA_local_max": max(I[c]["par_local"]["FA"] for c in cs),
               "cumple": bool(I["forzado|4"]["emisor"]["HIT"] < 0.3 and min(I[c]["emisor"]["AUROC4"] for c in cs) > 0.9
                              and max(I[c]["emisor"]["FA"] for c in cs) < 0.01 and max(I[c]["par_local"]["FA"] for c in cs) < 0.02)}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
