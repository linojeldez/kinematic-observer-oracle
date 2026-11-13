"""Arnés común de la fase 4 del plan de fortalecimiento del Paper A (E48 a E50).

Generaliza el de E42 (mismo banco `run_response`, misma semilla 7 + 1000 * i_phi +
réplica, misma rejilla de 18 ángulos) para admitir celdas con otro k, otro ruido y otra
calibración, y un observador que estima su propio suelo de ruido (`FlockAuto`). El motor
(`flocking.py`) no se toca.

Cada trabajo es un dict:
    celda, nom, model, kw, N, tau_W, pulso, i_phi, phi, rep, T, tau_base (None o segundos)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from scipy.special import stdtr

import flocking
from flocking import Flock, Params, run_response, phi_grid, _t_scale_from_mad
from parametros import P0

W = int(os.environ.get("FASE4_W", "8"))
SEMILLA_R = 7
D_PHI = 10.0
T_N = {250: 15.8, 500: 22.4, 1000: 31.6, 2000: 44.7}
WINS = (4.0, 8.0, 15.8, 22.4, 31.6, 44.7)
N_PHI = int(round(180.0 / D_PHI))      # ángulos que debe tener una réplica completa

ARM = {
    "ViSt": ("ViSt", dict()),
    "CoLd": ("CoLd", dict()),
    "oraculo": ("WiSt", dict(will_mode="oracle")),
    "amplitud": ("WiSt", dict(will_mode="bayes")),
    "adaptado": ("WiSt", dict(will_mode="bayes", detector="ciclo", tau_det=1.0)),
}


def _w(t):
    return f"{t:g}"


class FlockAuto(Flock):
    """Observador que estima su propio suelo de ruido (C3 del plan).

    El detector del motor recibe sigma_theta como parámetro (calibración externa). Aquí
    se reemplaza por una línea base local y móvil: para cada emisor j,

        sigma_j(t) = EMA_{tau_base}[ 1,4826 * mediana_{m en vecindario de j} |eps_m(t)| ],

    la anchura robusta del residuo en el vecindario de j (j y sus k vecinos), promediada
    con olvido exponencial de constante tau_base. No entra ninguna anchura medida fuera
    de la corrida. La mediana sobre el vecindario hace que un solo individuo que maniobra
    no mueva la línea base. El resto del detector es el del motor: mismo umbral en
    unidades de la anchura (c_det), misma t de Student (nu), misma escala de la maniobra
    (sigma_W), mismo prior y mismo suavizado. La estimación no se actualiza antes de
    t_ini (la velocidad angular suavizada parte de cero) y, mientras no existe, la
    evidencia es nula (posterior = prior).
    """

    tau_base = 5.0
    t_ini = 0.5
    ultimo = None

    def __init__(self, p, model):
        super().__init__(p, model)
        self._sig_num = np.zeros(p.N)
        self._sig_den = 0.0
        self._sig_paso = -1
        self._q0 = None
        self.sig_ini = None            # media de sigma_j cuando el líder se informa
        self.sig_max = 0.0             # máximo de esa media durante la corrida
        FlockAuto.ultimo = self

    def _sigma_local(self, eps, idx):
        p = self.p
        if self._sig_paso != self._paso:
            if self.t >= self.t_ini:
                a = np.exp(-p.dt / self.tau_base)
                inst = 1.4826 * np.median(np.abs(eps)[idx], axis=1)
                self._sig_num = a * self._sig_num + (1.0 - a) * inst
                self._sig_den = a * self._sig_den + (1.0 - a)
            self._sig_paso = self._paso
        if self._sig_den <= 0.0:
            return None
        s = self._sig_num / self._sig_den
        m = float(s.mean())
        if self.sig_ini is None and self.will.max() > 0.5:
            self.sig_ini = m
        self.sig_max = max(self.sig_max, m)
        return s

    def _llr_kinematic(self, idx):
        p = self.p
        eps = self.omega - self.omega[idx].mean(axis=1)
        nu = p.nu
        c = _t_scale_from_mad(nu)
        if self._q0 is None:
            self._q0 = float(np.clip(2.0 * stdtr(nu, -p.c_det / c), 1e-6, 1 - 1e-6))
            self.delta[:] = self._q0
        s0 = self._sigma_local(eps, idx)
        if s0 is None:
            return np.zeros(p.N)
        return llr_local(eps, s0, p, self)


def llr_local(eps, s0, p, f):
    """Log-cociente del detector del motor con anchura por emisor s0 (N,)."""
    nu = p.nu
    c = _t_scale_from_mad(nu)
    s0 = np.maximum(s0, 1e-9)
    s1 = np.hypot(s0, p.sigma_W)
    if p.detector == "ciclo":
        q0 = f._q0
        q1 = np.clip(2.0 * stdtr(nu, -p.c_det * s0 / (s1 * c)), 1e-6, 1 - 1e-6)
        if f._delta_paso != f._paso:
            a = np.exp(-p.dt / max(p.tau_det, 1e-6))
            f.delta = a * f.delta + (1.0 - a) * (np.abs(eps) > p.c_det * s0)
            f._delta_paso = f._paso
        n_ef = max(p.tau_det / max(p.tau_obs, 1e-6), 1.0)
        return n_ef * (f.delta * np.log(q1 / q0)
                       + (1.0 - f.delta) * np.log((1 - q1) / (1 - q0)))
    t0, t1 = s0 * c, s1 * c
    return np.log(t0 / t1) + 0.5 * (nu + 1.0) * (
        np.log1p(eps ** 2 / (nu * t0 ** 2)) - np.log1p(eps ** 2 / (nu * t1 ** 2)))


def trabajo(celda, nom, N, tau_W, pulso, i_phi, phi, rep, kw=None, base=None, tau_base=None):
    model, kw0 = ARM[base or nom]
    return {"celda": celda, "nom": nom, "model": model, "kw": {**kw0, **(kw or {})},
            "N": N, "tau_W": tau_W, "pulso": bool(pulso), "i_phi": i_phi, "phi": float(phi),
            "rep": rep, "T": T_N[N], "tau_base": tau_base}


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": a["N"], "lambda_fp": 0.0,
                  "tau_W": a["tau_W"], "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    t0 = time.time()
    if a["tau_base"]:
        flocking.Flock = type("FlockAutoJob", (FlockAuto,), {"tau_base": float(a["tau_base"])})
    try:
        r = run_response(p, a["model"], a["phi"], a["T"], every_coh=100, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    out = {"celda": a["celda"], "nom": a["nom"], "N": a["N"], "tau_W": a["tau_W"],
           "pulso": a["pulso"], "i_phi": a["i_phi"], "rep": a["rep"], "pol": r["pol"],
           "frac_min": r["frac_min"],
           "r": {_w(w): float(proj[:int(round(w / p.dt))].mean())
                 for w in WINS if w <= a["T"] + 1e-9},
           "seg": time.time() - t0}
    if a["tau_base"]:
        f = FlockAuto.ultimo
        out["sig_ini"], out["sig_max"] = f.sig_ini, f.sig_max
    return out


def clave(x):
    return (f'{x["celda"]}|{x["nom"]}|{x["N"]}|{x["tau_W"]:g}|{int(x["pulso"])}|'
            f'{x["i_phi"]}|{x["rep"]}')


def correr(jobs, f_parcial, etiqueta):
    """Corre con reanudación: guarda parciales cada 2 min y se salta lo ya hecho."""
    hechos = {}
    if os.path.exists(f_parcial):
        with open(f_parcial) as fh:
            for x in json.load(fh):
                hechos[clave(x)] = x
        print(f"[{etiqueta}] reanudando: {len(hechos)} trabajos ya hechos", flush=True)
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[{etiqueta}] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    if faltan:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(job, a) for a in faltan]
            for i, fu in enumerate(as_completed(futs), 1):
                x = fu.result()
                hechos[clave(x)] = x
                if time.time() - ult > 120 or i == len(futs):
                    with open(f_parcial, "w") as fh:
                        json.dump(list(hechos.values()), fh)
                    ult = time.time()
                    print(f"[{etiqueta}] {i}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    return [hechos[clave(a)] for a in jobs if clave(a) in hechos], time.time() - t0


def R_rep(res, celda, nom, N, pulso, win, tau_W, reps=None):
    """R por réplica (media sobre los 18 ángulos); sólo réplicas completas. Devuelve
    (ids de réplica, array)."""
    d = {}
    for x in res:
        if (x.get("celda", "base") == celda and x["nom"] == nom and x["N"] == N
                and x["pulso"] == pulso and abs(x["tau_W"] - tau_W) < 1e-9
                and _w(win) in x["r"]):
            d.setdefault(x["rep"], {})[x["i_phi"]] = x["r"][_w(win)]
    ids = [k for k in sorted(d) if len(d[k]) == N_PHI and (reps is None or k in reps)]
    return ids, np.array([float(np.mean(list(d[k].values()))) for k in ids])


def rec_ic(Rm, Rv, Ro, n_boot=4000, seed=42):
    """Recuperación (Rm - Rv)/(Ro - Rv) con IC 95 % remuestreando réplicas pareadas."""
    n = min(len(Rm), len(Rv), len(Ro))
    if n == 0:
        return {"rec": float("nan"), "ic95": [float("nan")] * 2, "n_rep": 0}
    Rm, Rv, Ro = Rm[:n], Rv[:n], Ro[:n]
    rec = (Rm.mean() - Rv.mean()) / (Ro.mean() - Rv.mean())
    rng = np.random.default_rng(seed)
    b = rng.integers(0, n, (n_boot, n))
    den = Ro[b].mean(axis=1) - Rv[b].mean(axis=1)
    recs = (Rm[b].mean(axis=1) - Rv[b].mean(axis=1)) / den
    lo, hi = np.percentile(recs, [2.5, 97.5])
    return {"rec": float(rec), "ic95": [float(lo), float(hi)], "n_rep": int(n),
            "R": float(Rm.mean()), "ventaja": float(Ro.mean() - Rv.mean()),
            "ventaja_ic95": [float(v) for v in np.percentile(den, [2.5, 97.5])],
            "_boot": recs}


def pareado(res, celda, nom, ref, N, pulso, win, tau_W, reps=None, celda_ref=None,
            nom_or="oraculo"):
    """rec_ic con las réplicas comunes a observador, referencia y oráculo."""
    cr = celda_ref or celda
    im, Rm = R_rep(res, celda, nom, N, pulso, win, tau_W, reps)
    iv, Rv = R_rep(res, cr, ref, N, pulso, win, tau_W, reps)
    io, Ro = R_rep(res, cr, nom_or, N, pulso, win, tau_W, reps)
    com = sorted(set(im) & set(iv) & set(io))
    f = lambda ids, R: np.array([R[ids.index(k)] for k in com])
    return rec_ic(f(im, Rm), f(iv, Rv), f(io, Ro))


def fmt(q):
    return f'{q["rec"]:6.3f} [{q["ic95"][0]:6.3f}; {q["ic95"][1]:6.3f}] (n = {q["n_rep"]})'


def limpio(q):
    q = dict(q)
    q.pop("_boot", None)
    return q


def crudos_previos():
    """Réplicas 0 a 7 guardadas por E42, E43, E43b y E44b, con celda = 'base'."""
    out = []
    for f in ("E42_escalera_N", "E43_escalera_tauW", "E43b_tauW_N1000", "E44b_cold"):
        with open(f"resultados/{f}_crudo.json") as fh:
            for x in json.load(fh)["resp"]:
                out.append({**x, "celda": "base"})
    # una sola copia por clave (E43 repite celdas de E42 con tau_W = 4 s)
    return list({clave(x): x for x in out}.values())
