"""Etapa 67 (plan 15, punto 4): diagnóstico del déficit del motor frente a la referencia.

El motor queda bajo los valores publicados con k chico (ViSt -25/-22/-14/+1 %, WiSt
-18/-15/-7/-1 % con k = 5/15/30/70, N = 250, sigma^2 = 1). E10 ya probó dt, la rejilla de
ángulos, el ruido x0,7/x1,4, la condición inicial uniforme y el auto-vecino. Aquí se cambian
de a una las elecciones que el texto publicado no fija y que E10 no tocó, con el protocolo de
la validación (líder forzado, will sin decaimiento, t_R = 15,8 s, rejilla de 18 ángulos,
semillas 1000 i_phi + réplica):

  base        el motor tal cual
  asent10     asentamiento de 10 s en vez de 3 s (bandada en equilibrio)
  ruido_paso  ruido literal por paso (sigma, no sigma/sqrt(dt)): 10 veces más débil
  secuencial  actualización secuencial aleatoria de las velocidades (no síncrona)
  lider_azar  líder al azar en vez del más cercano al centro
  ruido_rumbo sólo la componente transversal del ruido (sin fluctuación de rapidez)
  unitarios   alineamiento con rumbos unitarios, no con velocidades
  rapidez_fija rapidez reimpuesta a v0 tras cada paso
  sin_si      vecindario sin el propio agente (contra la definición publicada; referencia)

Diseño, sentido esperado de cada cambio y criterio en docs/bitacora/resultados_E56-E65.md
(«Plan 15 · E67»).

    PYTHONPATH=codigo python codigo/e67_deficit.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import flocking
from flocking import Flock, Params, run_response, phi_grid, _normalize

W = int(os.environ.get("FASE4_W", "9"))
KS = (5, 15, 70)
MODELOS = ("ViSt", "WiSt")
N_REP, N_REP_SEC = 8, 4          # la actualización secuencial cuesta ~15 veces más
REF = {5: (0.094, 0.177), 15: (0.104, 0.443), 30: (0.100, 0.720), 70: (0.099, 0.899)}
VARIANTES = ("base", "asent10", "ruido_paso", "secuencial", "lider_azar", "ruido_rumbo", "unitarios",
             "rapidez_fija", "sin_si")
F_CRUDO, F_JSON, F_TXT = "resultados/E67_deficit_crudo.json", "resultados/E67_deficit.json", "resultados/E67_deficit.txt"


class FlockVar(Flock):
    var = "base"

    def __init__(self, p, model):
        super().__init__(p, model)
        if self.var == "lider_azar":
            self.leader = int(np.random.default_rng(10_000 + p.seed).integers(p.N))

    def _v_vist(self, idx):
        if self.var == "unitarios":
            return _normalize(_normalize(self.vel)[idx].sum(axis=1))
        if self.var == "sin_si":
            return _normalize(self.vel[idx].sum(axis=1) - self.vel)
        return super()._v_vist(idx)

    def _perceived_will(self, idx, dist):
        w = super()._perceived_will(idx, dist)
        if self.var == "sin_si":
            w = np.where(idx == np.arange(self.p.N)[:, None], 0.0, w)
        return w

    def step(self):
        if self.var in ("base", "asent10", "ruido_paso", "lider_azar", "unitarios", "sin_si"):
            return super().step()
        p = self.p
        if self.var == "secuencial":
            return self._step_secuencial()
        v_des = p.v0 * self.desired_direction()
        dv = v_des - self.vel
        mag = np.linalg.norm(dv, axis=1)
        acc = _normalize(dv) * np.minimum(mag / p.tau_c, p.a_max)[:, None]
        noise = self.rng.normal(0.0, p.sigma_xi / np.sqrt(p.dt), size=(p.N, 2))
        if self.var == "ruido_rumbo":
            u = _normalize(self.vel)
            perp = np.stack([-u[:, 1], u[:, 0]], axis=1)
            noise = (noise * perp).sum(axis=1)[:, None] * perp
        self.vel += (acc + noise) * p.dt
        if self.var == "rapidez_fija":
            self.vel = p.v0 * _normalize(self.vel)
        self.pos += self.vel * p.dt
        self._post()
        return acc

    def _step_secuencial(self):
        """Cada agente, en orden aleatorio, actualiza su velocidad con las velocidades ya
        actualizadas de los que lo precedieron en este paso (vecindarios del inicio del paso)."""
        p = self.p
        idx, dist = self._neighbors()
        w_nb = self.salience(idx, dist) if self.model == "WiSt" else None
        noise = self.rng.normal(0.0, p.sigma_xi / np.sqrt(p.dt), size=(p.N, 2))
        orden = self.rng.permutation(p.N)
        acc_all = np.zeros((p.N, 2))
        vel = self.vel
        for i in orden:
            if self.forced is not None and i == self.leader:
                d = self.forced
            else:
                s = vel[idx[i]].sum(axis=0)
                m = s / max(np.linalg.norm(s), 1e-12)
                if self.model == "WiSt":
                    wi = self.will[i]
                    u = (1.0 - wi) * m + wi * self.d_des[i]
                    u = u / max(np.linalg.norm(u), 1e-12)
                    d = u + (w_nb[i][:, None] * vel[idx[i]] / p.v0).sum(axis=0)
                    d = d / max(np.linalg.norm(d), 1e-12)
                else:
                    d = m
            dv = p.v0 * d - vel[i]
            mg = np.linalg.norm(dv)
            a = (dv / max(mg, 1e-12)) * min(mg / p.tau_c, p.a_max)
            vel[i] = vel[i] + (a + noise[i]) * p.dt
            acc_all[i] = a
        self.pos += self.vel * p.dt
        self._post()
        return acc_all

    def _post(self):
        p = self.p
        th = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        dth = np.angle(np.exp(1j * (th - self.theta))) / p.dt
        a_obs = np.exp(-p.dt / p.tau_obs)
        self.omega = a_obs * self.omega + (1.0 - a_obs) * dth
        self.theta = th
        if np.isfinite(p.tau_W):
            self.will *= np.exp(-p.dt / p.tau_W)
        self._head = (self._head + 1) % self._buf_n
        self._buf_will[self._head] = self.will
        self._buf_om[self._head] = self.omega
        self.t += p.dt
        self._paso += 1


def job(a):
    kw = {"noise_mode": "paper"} if a["var"] == "ruido_paso" else {}
    p = Params(N=250, k=a["k"], sigma_xi=1.0, seed=1000 * a["i_phi"] + a["rep"], **kw)
    flocking.Flock = type("FV", (FlockVar,), {"var": a["var"]})
    t0 = time.time()
    try:
        r = run_response(p, a["model"], a["phi"], 15.8, t_s=10.0 if a["var"] == "asent10" else 3.0, every_coh=200)
    finally:
        flocking.Flock = Flock
    return {**{k: a[k] for k in ("var", "model", "k", "i_phi", "rep")}, "r": r["r"], "pol": r["pol"],
            "seg": time.time() - t0}


def clave(x):
    return f'{x["var"]}|{x["model"]}|{x["k"]}|{x["i_phi"]}|{x["rep"]}'


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phis = phi_grid(10.0)
    jobs = [{"var": v, "model": m, "k": k, "i_phi": i, "phi": float(phi), "rep": rep}
            for v in VARIANTES for m in MODELOS for k in KS for i, phi in enumerate(phis)
            for rep in range(N_REP_SEC if v == "secuencial" else N_REP)]
    jobs.sort(key=lambda a: (a["var"] != "secuencial", -a["k"]))
    f_par = F_CRUDO + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E67] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for n, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or n == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w"))
                ult = time.time()
                print(f"[E67] {n}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    seg = time.time() - t0
    res = [hechos[clave(a)] for a in jobs]
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    rng = np.random.default_rng(67)
    out, L = {"R": {}, "desvio": {}}, ["=== E67 · déficit del motor frente a la referencia (N = 250, sigma^2 = 1, 8 réplicas) ==="]
    L.append(f"{'variante':13s} " + " ".join(f"{m} k={k:<3d}" for m in MODELOS for k in KS))
    for v in VARIANTES:
        fila, des = {}, {}
        for m in MODELOS:
            for k in KS:
                nr = N_REP_SEC if v == "secuencial" else N_REP
                rr = np.array([[x["r"] for x in res if x["var"] == v and x["model"] == m and x["k"] == k and x["rep"] == r]
                               for r in range(nr)]).mean(axis=1)
                ref = REF[k][0 if m == "ViSt" else 1]
                bs = [rr[rng.integers(0, nr, nr)].mean() for _ in range(2000)]
                fila[f"{m}|{k}"] = {"R": float(rr.mean()), "ic95": [float(q) for q in np.percentile(bs, [2.5, 97.5])]}
                des[f"{m}|{k}"] = float(rr.mean() / ref - 1.0)
        out["R"][v], out["desvio"][v] = fila, des
        L.append(f"{v:13s} " + " ".join(f"{100 * des[f'{m}|{k}']:+8.1f} %" for m in MODELOS for k in KS))
    D = out["desvio"]
    cierra = [v for v in VARIANTES if v != "base" and all(abs(D[v][f"{m}|{k}"]) < 0.05 for m in MODELOS for k in (5, 15))
              and all(abs(D[v][f"{m}|70"]) < 0.05 for m in MODELOS)]
    out["pred"] = {"C1": {"cierran": cierra, "cumple": bool(cierra)}}
    L.append("\n=== Criterio ===")
    L.append(f"C1 (un cambio deja ViSt y WiSt a menos de 5 % con k = 5 y 15 sin romper k = 70): "
             f"{'CUMPLE: ' + ', '.join(cierra) if cierra else 'NINGUNO'}")
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
