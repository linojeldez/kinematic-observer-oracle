"""Etapa 30 (L3-bis): la regla de la ventana, tercer taxón — palomas bajo RobotFalcon.

Datos: Papadopoulou et al. 2022 (RSOS 9:211898; Dryad 905qfttkc, espejo Zenodo
6027115), `all_emp_data.csv`: GPS 5 Hz de bandadas de palomas mensajeras (8-10 y
28-34 aves) en 43 vuelos-bandada, 27 bajo ataque del RobotFalcon y 16 control.
Origen empírico: Sankey et al. 2021 (Curr. Biol.).

PREDICCIÓN PRE-REGISTRADA (congelada antes de calcular ninguna AUC; solo se miró
el censo: vuelos, tamaños, duraciones, percentiles globales de |ang_vel| para
fijar el umbral de detección de eventos):

  - Ley conservada (E27/E28): tau*95 en 0.7-1.5 s y huella del iniciador
    1.0-1.8 s, con AUC > 0.65 — la misma escala ~1 s que grajilla y tetra.
  - Biomecánica: tau* = tau*_grajilla x (omega_grajilla/omega_paloma)
    = 1.36/omega_max s, con omega_max medido de estos mismos datos con los
    estimadores armonizados de E28 (p90 de |omega| en ventana de giro, y
    a_lat/|v| como contraste). Con el p90 global de 0.645 rad/s del censo,
    la biomecánica apunta a tau* ~ 2 s.
  - Zona degenerada (lección E29): si omega_max cae en 0.9-1.4 rad/s ambas
    hipótesis coinciden y el dataset NO discrimina; se declara y no se fuerza.
  - Censo mínimo (lección E29): >= 20 eventos de giro colectivo; con menos,
    no se concluye.

Diseño (espejo de e27_tetras, adaptado a formato largo y 5 Hz):
  - rumbo derivado de la VELOCIDAD (lección E29: nunca del eje del cuerpo ni de
    columnas precalculadas); `ang_vel` del CSV solo como verificación cruzada;
  - lon/lat -> metros con proyección equirrectangular local por vuelo;
  - evento = tramo con |Omega| > 0.2 rad/s (Omega = media entre aves de
    dtheta/dt, suavizada 1 s) y amplitud integrada |DPhi| >= 30 grados;
    s0 = pico de |Omega| (mitad del giro, análogo del cruce de E27);
    separación >= 8 s (se queda el de mayor amplitud);
  - ranking de iniciadores: instante en que cada ave cruza DPhi/2 de SU rumbo
    de referencia pre-evento (análogo del cruce propio de E27); cuartiles 25 %;
  - residuo eps = omega propia - media de los k=3 vecinos más cercanos;
    sigma global (MAD) sobre fotogramas a > 3 s de todo evento;
  - huella: |eps|/sigma del primer iniciador alineada a SU cruce, umbral 1.5,
    sostén 0.6 s, tope 5 s;
  - AUC(tau) inic./seguidores con ventana pre-colectiva [s0-tau, s0],
    tau en {0.2 ... 6} s; tau*95 = menor tau con AUC >= 95 % del máximo.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DT = 0.2              # s (5 Hz, verificado en el censo)
OM_MIN = 0.2          # rad/s: umbral de giro colectivo (>3x ruido de la media)
AMP_MIN = np.radians(30.0)
SEP_MIN = 8.0         # s entre eventos
K_VEC = 3
TAUS = (0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
Q = 0.25
CAP_HUELLA = 5.0      # s
SOSTEN = 3            # fotogramas (0.6 s) para el fin de la huella

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "datos", "palomas", "all_emp_data.csv")


def suav(x, w):
    k = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(x, k, "same")
    return np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, x)


def vuelos(df):
    """Genera (clave, X, Y) en metros por vuelo-bandada, fila = fotograma."""
    for clave, d in df.groupby(["condition", "small.big", "group.num",
                                "study.flight"]):
        idx = np.round(d["time"].values / DT).astype(int)
        aves = np.sort(d["pigeon"].unique())
        T = idx.max() + 1
        X = np.full((T, len(aves)), np.nan)
        Y = np.full((T, len(aves)), np.nan)
        lat0 = np.radians(np.nanmedian(d["lat"].values))
        mx = 111320.0 * np.cos(lat0)
        my = 110540.0
        lon0, lat0d = np.nanmedian(d["lon"].values), np.nanmedian(d["lat"].values)
        for j, p in enumerate(aves):
            dp = d[d["pigeon"] == p]
            ii = np.round(dp["time"].values / DT).astype(int)
            X[ii, j] = (dp["lon"].values - lon0) * mx
            Y[ii, j] = (dp["lat"].values - lat0d) * my
        yield clave, X, Y


def cinematica(X, Y):
    """Velocidad, rumbo y omega por ave (unwrap por segmentos contiguos)."""
    w = 3
    Xs, Ys = suav(X, w), suav(Y, w)
    Vx, Vy = np.gradient(Xs, DT, axis=0), np.gradient(Ys, DT, axis=0)
    th = np.arctan2(Vy, Vx)
    th[~(np.isfinite(X) & np.isfinite(Y))] = np.nan
    om = np.full_like(th, np.nan)
    for a in range(th.shape[1]):
        ok = np.isfinite(th[:, a])
        idx = np.flatnonzero(ok)
        if len(idx) < 15:
            continue
        cortes = np.flatnonzero(np.diff(idx) > 1)
        for seg in np.split(idx, cortes + 1):
            if len(seg) < 15:
                continue
            om[seg, a] = np.gradient(np.unwrap(th[seg, a]), DT)
    return Vx, Vy, th, suav(om, w)


def eventos_giro(om):
    """Tramos de giro colectivo: (s0, r0, r1, amplitud firmada)."""
    Om = suav(np.nanmean(om, axis=1), int(1.0 / DT))
    enc = np.abs(Om) > OM_MIN
    enc &= np.isfinite(Om)
    evs = []
    idx = np.flatnonzero(enc)
    if not len(idx):
        return evs, Om
    cortes = np.flatnonzero(np.diff(idx) > 1)
    for seg in np.split(idx, cortes + 1):
        r0, r1 = int(seg[0]), int(seg[-1])
        amp = float(np.nansum(Om[r0:r1 + 1]) * DT)
        if abs(amp) < AMP_MIN:
            continue
        s0 = r0 + int(np.nanargmax(np.abs(Om[r0:r1 + 1])))
        evs.append({"s0": s0, "r0": r0, "r1": r1, "amp": amp})
    # separación mínima: se queda el de mayor amplitud
    evs.sort(key=lambda e: -abs(e["amp"]))
    filt = []
    for e in evs:
        if all(abs(e["s0"] - f["s0"]) * DT >= SEP_MIN for f in filt):
            filt.append(e)
    return sorted(filt, key=lambda e: e["s0"]), Om


def cruces_por_ave(th, ev):
    """Instante en que cada ave cruza la mitad de la amplitud del giro."""
    T, N = th.shape
    a0 = max(0, ev["r0"] - int(3 / DT))
    a1 = max(0, ev["r0"] - int(1 / DT))
    v0 = max(0, ev["r0"] - int(2 / DT))
    v1 = min(T, ev["r1"] + int(2 / DT))
    obj = ev["amp"] / 2.0
    t_rev = np.full(N, np.nan)
    for a in range(N):
        ref_v = th[a0:a1, a]
        ref_v = ref_v[np.isfinite(ref_v)]
        if len(ref_v) < 3:
            continue
        ref = np.arctan2(np.mean(np.sin(ref_v)), np.mean(np.cos(ref_v)))
        tr = th[v0:v1, a]
        ok = np.isfinite(tr)
        if ok.sum() < 10:
            continue
        d = np.unwrap(tr[ok]) - np.unwrap(tr[ok])[0]
        d += np.angle(np.exp(1j * (tr[ok][0] - ref)))
        cz = np.flatnonzero((d[:-1] - obj) * (d[1:] - obj) <= 0) if obj >= 0 else \
             np.flatnonzero((d[:-1] - obj) * (d[1:] - obj) <= 0)
        if obj < 0:
            cz = np.flatnonzero((d[:-1] > obj) & (d[1:] <= obj))
        else:
            cz = np.flatnonzero((d[:-1] < obj) & (d[1:] >= obj))
        if len(cz):
            t_rev[a] = (v0 + np.flatnonzero(ok)[cz[0]]) * DT
    return t_rev


def residuo(X, Y, om, k=K_VEC):
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    P = np.stack([X, Y], -1)
    for s in range(T):
        p, o = P[s], om[s]
        okk = np.isfinite(p).all(1) & np.isfinite(o)
        if okk.sum() < k + 2:
            continue
        ii = np.flatnonzero(okk)
        D = np.linalg.norm(p[ii][:, None] - p[ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, 1)[:, :k]
        eps[s, ii] = o[ii] - o[ii][vec].mean(1)
    return eps


if __name__ == "__main__":
    df = pd.read_csv(CSV, index_col=0)
    todo = {"omega_max": [], "omega_max_alat": [], "huellas": [],
            "por_tau": {t: ([], []) for t in TAUS},
            "n_ev": 0, "n_ev_p": 0, "n_ev_c": 0, "n_vuelos": 0}
    base_eps = []
    paquetes = []

    for clave, X, Y in vuelos(df):
        cond = clave[0]
        Vx, Vy, th, om = cinematica(X, Y)
        evs, Om = eventos_giro(om)
        eps = residuo(X, Y, om)
        T = om.shape[0]
        lejos = np.ones(T, bool)
        for e in evs:
            lejos[max(0, e["s0"] - int(3 / DT)):e["s0"] + int(3 / DT)] = False
        fb = np.isfinite(eps) & lejos[:, None]
        base_eps.append(eps[fb])
        todo["n_vuelos"] += 1
        paquetes.append((clave, X, Y, Vx, Vy, th, om, eps, evs))
        if evs:
            print(f'{cond} g{clave[2]} vuelo {clave[3]:3d}: {len(evs)} giros '
                  f'(amp {", ".join(f"{np.degrees(e['amp']):+.0f}°" for e in evs)})')

    sig = float(np.nanmedian(np.abs(np.concatenate(base_eps))) * 1.4826)
    print(f'\nsigma base (global): {sig:.3f} rad/s')

    for clave, X, Y, Vx, Vy, th, om, eps, evs in paquetes:
        cond = clave[0]
        T, N = om.shape
        for ev in evs:
            s0 = ev["s0"]
            t_rev = cruces_por_ave(th, ev)
            okr = np.isfinite(t_rev)
            if okr.sum() < max(4, int(N * 0.6)):
                continue
            d_rel = t_rev - np.nanmin(t_rev)
            lim_i = np.nanquantile(d_rel[okr], Q)
            lim_s = np.nanquantile(d_rel[okr], 1 - Q)
            ini = [a for a in range(N) if okr[a] and d_rel[a] <= lim_i]
            seg = [a for a in range(N) if okr[a] and d_rel[a] >= lim_s]

            todo["n_ev"] += 1
            todo["n_ev_p" if cond == "p" else "n_ev_c"] += 1

            vturn = slice(max(0, s0 - int(1 / DT)), min(T, s0 + int(2 / DT)))
            omv = np.abs(om[vturn])
            if np.isfinite(omv).any():
                todo["omega_max"].append(float(np.nanpercentile(omv, 90)))
            vx, vy = Vx[vturn], Vy[vturn]
            ax = np.gradient(vx, DT, axis=0)
            ay = np.gradient(vy, DT, axis=0)
            rap = np.sqrt(vx**2 + vy**2)
            a_lat = np.abs(ax * (-vy) + ay * vx) / np.maximum(rap, 1e-9)
            oma = a_lat / np.maximum(rap, 1e-9)
            if np.isfinite(oma).any():
                todo["omega_max_alat"].append(float(np.nanpercentile(oma, 90)))

            orden = np.argsort(np.where(okr, t_rev, np.inf))
            a0 = int(orden[0])
            s_ini = int(round(t_rev[a0] / DT))
            tramo = np.abs(eps[s_ini:s_ini + int(CAP_HUELLA / DT), a0]) / max(sig, 1e-9)
            if len(tramo) and np.isfinite(tramo).mean() > 0.7:
                sm = suav(np.nan_to_num(tramo, nan=0.0), 3)
                ip = int(np.argmax(sm))
                dur = CAP_HUELLA
                for j in range(ip, len(sm) - SOSTEN):
                    if np.all(sm[j:j + SOSTEN] < 1.5):
                        dur = j * DT
                        break
                todo["huellas"].append(dur)

            for tau in TAUS:
                w0 = s0 - int(tau / DT)
                if w0 < 0:
                    continue
                m = np.nanmean(np.abs(eps[w0:s0]) / max(sig, 1e-9), axis=0)
                todo["por_tau"][tau][0].extend([m[a] for a in ini])
                todo["por_tau"][tau][1].extend([m[a] for a in seg])

    def auc(x1, x0):
        x1 = np.asarray([v for v in x1 if np.isfinite(v)], float)
        x0 = np.asarray([v for v in x0 if np.isfinite(v)], float)
        if len(x1) == 0 or len(x0) == 0:
            return np.nan, 0, 0
        a = ((x1[:, None] > x0[None, :]).sum()
             + 0.5 * (x1[:, None] == x0[None, :]).sum()) / (len(x1) * len(x0))
        return float(a), len(x1), len(x0)

    om_max = float(np.median(todo["omega_max"]))
    om_alat = float(np.median(todo["omega_max_alat"]))
    hue = float(np.median(todo["huellas"])) if todo["huellas"] else np.nan
    print(f'\n{todo["n_vuelos"]} vuelos-bandada, {todo["n_ev"]} eventos '
          f'({todo["n_ev_p"]} depredación, {todo["n_ev_c"]} control)')
    print(f'omega_max mediana (p90 en giro): {om_max:.2f} rad/s | '
          f'a_lat/|v|: {om_alat:.2f}  (grajilla 1.36/1.34, tetra 4.7-4.9)')
    print(f'huella del iniciador: {hue:.2f} s  (grajilla 1.03-1.42, tetra 1.78)')
    print(f'\n{"tau (s)":>8s} {"AUC":>7s} {"n":>12s}')
    curva = {}
    for tau in TAUS:
        a, n1, n0 = auc(*todo["por_tau"][tau])
        curva[tau] = a
        print(f'{tau:8.1f} {a:7.3f} {n1:5d}/{n0:5d}')
    fin = {t: a for t, a in curva.items() if np.isfinite(a)}
    tau_star = max(fin, key=fin.get)
    mx = fin[tau_star]
    tau95 = min(t for t, a in fin.items() if a >= 0.95 * mx)
    print(f'\ntau* = {tau_star:.1f} s | tau*95 = {tau95:.1f} s | AUC max = {mx:.3f}')
    print(f'biomecánica: 1.36/{om_max:.2f} = {1.36 / om_max:.2f} s | '
          f'conservada: 0.7-1.5 s')
    print('PRE-REGISTRO: >=20 eventos; conservada tau*95 0.7-1.5 s, huella '
          '1.0-1.8 s, AUC>0.65; degenerada si omega_max en 0.9-1.4')

    with open(os.path.join(RAIZ, "resultados", "E30_palomas.json"), "w") as f:
        json.dump({"n_vuelos": todo["n_vuelos"], "n_eventos": todo["n_ev"],
                   "n_ev_predacion": todo["n_ev_p"], "n_ev_control": todo["n_ev_c"],
                   "sigma": sig, "omega_max": om_max, "omega_max_alat": om_alat,
                   "huella": hue, "curva_auc": curva, "tau_star": tau_star,
                   "tau_star95": tau95, "auc_max": mx,
                   "prediccion_biomecanica": 1.36 / om_max,
                   "n_huellas": len(todo["huellas"])}, f, indent=1)
