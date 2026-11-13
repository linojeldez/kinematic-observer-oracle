"""Etapa 35: la regla de la ventana en espinosos — el segundo cardumen del paper.

Datos: Jolles, Boogert, Sridhar, Couzin & Manica 2017 (Curr. Biol.,
10.1016/j.cub.2017.08.004; Cambridge Apollo 10.17863/CAM.12136),
B-cm1-id-full.csv: contexto de cardumen libre, 25 grupo-trials de N=5
espinosos (Gasterosteus aculeatus), 30 min a 24 fps (fps inferido del propio
dato: v_provista / desplazamiento suavizado = 23.9), x,y en mm.

CENSO (pre-compuerta, ya medido): polarización mediana 0.92; 6,784 giros
colectivos (|Omega|>0.4 rad/s, amplitud >=45°) — compuerta >=20 superada x340.

PREDICCIÓN PRE-REGISTRADA (congelada antes de calcular ninguna AUC ni huella):
  - Ley conservada (E27/E28/E33): tau*95 en 0.7-1.5 s, huella (cruda,
    censurada a 3 s) ~0.7-2 s, AUC > 0.65 — sea cual sea la omega_max medida.
  - Biomecánica: tau* ~ Dphi/omega_max con omega_max medida de estos datos
    (esperada entre grajilla 1.35 y tetra 4.9 — el espinoso llenaría el hueco
    del eje de agilidad).
  - Delta21: si la latencia es un retardo por enlace específico de especie
    (E32/E33), el espinoso debería dar ~0.3-0.6 s (como el tetra, pez), y
    PLANO si comparamos con subconjuntos por densidad de vecinos.

Diseño (armonizado con E27/E33/E34; desviaciones declaradas):
  - rumbo de la velocidad (posiciones con mediana móvil de 5 fotogramas);
    omega por pez con unwrap por segmentos; Omega de grupo = media (>=4 de 5);
  - evento = |Omega| suavizada 0.5 s > 0.4 rad/s con amplitud >= 45°;
    SEP_MIN = 3 s (los espinosos giran cada ~6.6 s — declarado; el SEP de
    10 s del tetra vaciaría el censo sin motivo físico);
  - sigma base: fotogramas a >1.5 s de todo evento y con |Omega|<0.15
    (exclusión menor que en tetras por la densidad de giros — declarado);
  - arranques por cruce interpolado de media amplitud; iniciador/rezagado por
    cuartiles q=0.25 (a N=5: 1-2 peces por lado, como E33-N5);
  - eps = omega propia - media de los k=3 vecinos más cercanos;
  - AUC(tau) pre-evento [s0-tau, s0], tau en {0.1..3}; huella del primer
    iniciador (1.5 sigma, sostén 0.2 s, tope 3 s); Delta21 = t(2)-t(1).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "datos", "jolles_espinosos", "B-cm1-id-full.csv")
FPS = 24
DT = 1.0 / FPS
OM_MIN = 0.4
AMP_MIN = np.radians(45)
SEP_MIN = 3.0
K_VEC = 3
Q = 0.25
TAUS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)


def suav(x, w):
    k = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(x, k, "same")
    return np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, x)


def cinematica(X, Y):
    Xs = pd.DataFrame(X).rolling(5, center=True, min_periods=3).median().values
    Ys = pd.DataFrame(Y).rolling(5, center=True, min_periods=3).median().values
    Vx = np.gradient(Xs, DT, axis=0)
    Vy = np.gradient(Ys, DT, axis=0)
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    for a in range(th.shape[1]):
        idx = np.flatnonzero(np.isfinite(th[:, a]))
        if len(idx) < FPS:
            continue
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) < FPS // 2:
                continue
            om[seg, a] = np.gradient(np.unwrap(th[seg, a]), DT)
    return Xs, Ys, th, suav(om, 3)


def eventos(om):
    Om = np.nanmean(om, axis=1)
    Om[np.sum(np.isfinite(om), axis=1) < 4] = np.nan
    Oms = pd.Series(Om).rolling(int(0.5 * FPS), center=True,
                                min_periods=4).mean().values
    enc = np.abs(np.nan_to_num(Oms)) > OM_MIN
    evs = []
    idx = np.flatnonzero(enc)
    if len(idx):
        for seg in np.split(idx, np.flatnonzero(np.diff(idx)
                                                > int(0.2 * FPS)) + 1):
            if len(seg) < 0.3 * FPS:
                continue
            r0, r1 = int(seg[0]), int(seg[-1])
            amp = float(np.nansum(Oms[r0:r1 + 1]) * DT)
            if abs(amp) < AMP_MIN:
                continue
            s0 = r0 + int(np.nanargmax(np.abs(Oms[r0:r1 + 1])))
            evs.append({"s0": s0, "r0": r0, "r1": r1, "amp": amp})
    evs.sort(key=lambda e: -abs(e["amp"]))
    filt = []
    for e in evs:
        if all(abs(e["s0"] - f["s0"]) * DT >= SEP_MIN for f in filt):
            filt.append(e)
    return sorted(filt, key=lambda e: e["s0"]), Oms


def cruces(th, ev):
    T, N = th.shape
    a0 = max(0, ev["r0"] - int(1.0 / DT))
    a1 = max(1, ev["r0"] - int(0.2 / DT))
    v0 = max(0, ev["r0"] - int(0.5 / DT))
    v1 = min(T, ev["r1"] + int(1.0 / DT))
    obj = ev["amp"] / 2.0
    t_rev = np.full(N, np.nan)
    for a in range(N):
        rv = th[a0:a1, a]
        rv = rv[np.isfinite(rv)]
        if len(rv) < 5:
            continue
        ref = np.arctan2(np.mean(np.sin(rv)), np.mean(np.cos(rv)))
        tr = th[v0:v1, a]
        ok = np.isfinite(tr)
        if ok.sum() < 10:
            continue
        iok = np.flatnonzero(ok)
        d = np.unwrap(tr[ok])
        d = d - d[0] + np.angle(np.exp(1j * (tr[ok][0] - ref)))
        if obj >= 0:
            cz = np.flatnonzero((d[:-1] < obj) & (d[1:] >= obj))
        else:
            cz = np.flatnonzero((d[:-1] > obj) & (d[1:] <= obj))
        if len(cz):
            j = cz[0]
            f = (obj - d[j]) / max(abs(d[j + 1] - d[j]), 1e-9) * \
                np.sign(d[j + 1] - d[j])
            t_rev[a] = (v0 + iok[j] + np.clip(f, 0, 1)
                        * (iok[j + 1] - iok[j])) * DT
    return t_rev


def residuo(Xs, Ys, om):
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    P = np.stack([Xs, Ys], -1)
    for s in range(T):
        okk = np.isfinite(P[s]).all(1) & np.isfinite(om[s])
        if okk.sum() < K_VEC + 1:
            continue
        ii = np.flatnonzero(okk)
        D = np.linalg.norm(P[s][ii][:, None] - P[s][ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, 1)[:, :K_VEC]
        eps[s, ii] = om[s][ii] - om[s][ii][vec].mean(1)
    return eps


if __name__ == "__main__":
    todo = pd.read_csv(CSV, usecols=["gr.id", "trial", "frame", "fish.id",
                                     "x", "y"])
    res = {"omega_max": [], "huellas": [], "d21": [],
           "por_tau": {t: ([], []) for t in TAUS}, "n_ev": 0, "n_gt": 0}
    for (gid, tr), g in todo.groupby(["gr.id", "trial"]):
        X = g.pivot_table(index="frame", columns="fish.id", values="x").values
        Y = g.pivot_table(index="frame", columns="fish.id", values="y").values
        Xs, Ys, th, om = cinematica(X, Y)
        evs, Oms = eventos(om)
        eps = residuo(Xs, Ys, om)
        T, N = om.shape
        lejos = np.ones(T, bool)
        for e in evs:
            lejos[max(0, e["r0"] - int(1.5 / DT)):e["r1"] + int(1.5 / DT)] = 0
        fb = np.isfinite(eps) & lejos[:, None] & \
            (np.abs(np.nan_to_num(Oms, nan=1)) < 0.15)[:, None]
        sig = float(np.nanmedian(np.abs(eps[fb])) * 1.4826) if fb.any() \
            else np.nan
        res["n_gt"] += 1
        for ev in evs:
            s0 = ev["s0"]
            t_rev = cruces(th, ev)
            okr = np.isfinite(t_rev)
            if okr.sum() < 4:
                continue
            res["n_ev"] += 1
            d = np.sort(t_rev[okr])
            res["d21"].append(float(d[1] - d[0]))
            d_rel = t_rev - np.nanmin(t_rev)
            li = np.nanquantile(d_rel[okr], Q)
            ls = np.nanquantile(d_rel[okr], 1 - Q)
            ini = [a for a in range(N) if okr[a] and d_rel[a] <= li]
            seg = [a for a in range(N) if okr[a] and d_rel[a] >= ls]
            vturn = slice(max(0, s0 - int(1 / DT)), min(T, s0 + int(1 / DT)))
            omv = np.abs(om[vturn])
            if np.isfinite(omv).any():
                res["omega_max"].append(float(np.nanpercentile(omv, 90)))
            a0 = int(np.nanargmin(np.where(okr, t_rev, np.inf)))
            s_ini = int(round(t_rev[a0] / DT))
            tramo = np.abs(eps[s_ini:s_ini + int(3 / DT), a0]) / max(sig, 1e-9)
            if len(tramo) and np.isfinite(tramo).mean() > 0.6:
                sm = suav(np.nan_to_num(tramo, nan=0.0), 5)
                ip = int(np.argmax(sm))
                dur = 3.0
                ns = int(0.2 * FPS)
                for j in range(ip, len(sm) - ns):
                    if np.all(sm[j:j + ns] < 1.5):
                        dur = j * DT
                        break
                res["huellas"].append(dur)
            for tau in TAUS:
                w0 = s0 - int(tau / DT)
                if w0 < 0:
                    continue
                mm = np.nanmean(np.abs(eps[w0:s0]) / max(sig, 1e-9), axis=0)
                res["por_tau"][tau][0].extend([mm[a] for a in ini])
                res["por_tau"][tau][1].extend([mm[a] for a in seg])

    def auc(x1, x0):
        x1 = np.asarray([v for v in x1 if np.isfinite(v)], float)
        x0 = np.asarray([v for v in x0 if np.isfinite(v)], float)
        if len(x1) == 0 or len(x0) == 0:
            return np.nan, 0, 0
        a = ((x1[:, None] > x0[None, :]).sum()
             + 0.5 * (x1[:, None] == x0[None, :]).sum()) / (len(x1) * len(x0))
        return float(a), len(x1), len(x0)

    om_max = float(np.median(res["omega_max"]))
    hue = float(np.median(res["huellas"]))
    d21 = float(np.median(res["d21"]))
    print(f"{res['n_ev']} giros con >=4 cruces en {res['n_gt']} grupo-trials")
    print(f"omega_max p90: {om_max:.2f} rad/s (grajilla 1.35, tetra 4.5-5.2)")
    print(f"huella cruda: {hue:.2f} s | Delta21: {d21:.2f} s (tetra 0.40-0.56)")
    print(f"\n{'tau':>6} {'AUC':>7} {'n':>14}")
    curva = {}
    for tau in TAUS:
        a, n1, n0 = auc(*res["por_tau"][tau])
        curva[tau] = a
        print(f"{tau:6.1f} {a:7.3f} {n1:6d}/{n0:6d}")
    fin = {t: a for t, a in curva.items() if np.isfinite(a)}
    ts = max(fin, key=fin.get)
    t95 = min(t for t, a in fin.items() if a >= 0.95 * fin[ts])
    print(f"\ntau* = {ts:.1f} s | tau*95 = {t95:.1f} s | AUC max = {fin[ts]:.3f}")
    print(f"biomecánica Dphi/om: {abs(np.radians(90))/om_max:.2f} s (a 90°) | "
          f"conservada: 0.7-1.5 s")
    with open(os.path.join(RAIZ, "resultados", "E35_espinosos.json"), "w") as f:
        json.dump({"n_eventos": res["n_ev"], "n_grupo_trials": res["n_gt"],
                   "omega_max": om_max, "huella": hue, "delta21": d21,
                   "curva_auc": curva, "tau_star": ts, "tau_star95": t95,
                   "auc_max": fin[ts],
                   "n_huellas": len(res["huellas"])}, f, indent=1)
