"""Etapa 27 (L3): la regla de la ventana, segundo taxón — tetras de Lecheval.

La predicción, sin parámetros libres: si la ventana óptima del observador es la
duración de la maniobra y la maniobra la fija el cuerpo, entonces

    tau*_tetra / tau*_grajilla  ~  omega_max_grajilla / omega_max_tetra

con todo medido de los propios datos de cada especie. Referencias grajilla (E16/E23):
omega_max = 1.36 rad/s, huella 1.03 s, tau* = 1 s (AUC 0.69-0.71 pre-colectiva).

Datos: Lecheval et al. 2018 (Dryad 9m6d2, CC0). Trayectorias 2D a 50 fps en tanque
anular (r 250-350 mm), grupos de 8 y 10 (los del análisis de giros en U de Zheng
2024). ~50 min por experimento; los giros en U son espontáneos y hay que detectarlos.

Notas de formato que costaron el sondeo:
  - columnas X_i, Y_i, H_i por pez; SIN columna de tiempo (fila = fotograma, dt=1/50);
  - la columna H NO es el rumbo de movimiento (cambia ~0.6°/s mientras el pez recorre
    el anillo a ~50°/s): se ignora y el rumbo se deriva de la velocidad, que además es
    el mismo observable usado en grajillas y en el motor;
  - ~4 % de NaN por pérdidas de tracking: rumbos desenrollados por segmentos contiguos
    (la lección de E9: np.unwrap propaga NaN).

Detección de giro en U colectivo (con histéresis): Omega(t) = media entre peces de
dphi/dt (phi = posición angular en el anillo), suavizada 1 s; evento = cruce por cero
con signo sostenido y |Omega| > 0.3 rad/s en las ventanas [-3,-1] y [+1,+3] s, y
separación mínima de 10 s entre eventos.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

FPS = 50.0
DT = 1.0 / FPS
OM_MIN = 0.3          # rad/s: umbral de giro colectivo sostenido
SEP_MIN = 10.0        # s entre eventos
K_VEC = 3             # vecinos topológicos para el residuo (N=8-10)
TAUS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
Q = 0.25              # cuartiles iniciador/seguidor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "datos", "lecheval_tetras")


def suav(x, w):
    """Media móvil por columnas, tolerante a NaN en los bordes."""
    k = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(x, k, "same")
    return np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, x)


def cargar(fn):
    d = np.genfromtxt(fn, delimiter=",", skip_header=1)
    X, Y = d[:, 0::3], d[:, 1::3]
    return X, Y


def cinematica(X, Y, sf=1.0):
    """Velocidad, rumbo theta y omega por pez, con unwrap por segmentos contiguos."""
    w = max(3, int(round(5 * sf)))
    Xs, Ys = suav(X, w), suav(Y, w)
    Vx, Vy = np.gradient(Xs, DT, axis=0), np.gradient(Ys, DT, axis=0)
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    for a in range(th.shape[1]):
        ok = np.isfinite(th[:, a])
        idx = np.flatnonzero(ok)
        if len(idx) < 100:
            continue
        cortes = np.flatnonzero(np.diff(idx) > 1)
        for seg in np.split(idx, cortes + 1):
            if len(seg) < 50:
                continue
            om[seg, a] = np.gradient(np.unwrap(th[seg, a]), DT)
    return Vx, Vy, suav(om, max(3, int(round(5 * sf))))


def uturns(X, Y, om_min=OM_MIN):
    """Instantes (índices) de giro en U colectivo, con histéresis."""
    cx, cy = np.nanmean(X), np.nanmean(Y)
    phi = np.arctan2(Y - cy, X - cx)
    dphi = np.full_like(phi, np.nan)
    for a in range(phi.shape[1]):
        ok = np.isfinite(phi[:, a])
        idx = np.flatnonzero(ok)
        if len(idx) < 100:
            continue
        cortes = np.flatnonzero(np.diff(idx) > 1)
        for seg in np.split(idx, cortes + 1):
            if len(seg) < 50:
                continue
            dphi[seg, a] = np.gradient(np.unwrap(phi[seg, a]), DT)
    Om = suav(np.nanmean(dphi, axis=1), int(1.0 / DT))

    w = int(2.0 / DT)
    eventos = []
    ultimo = -10**9
    cruces = np.flatnonzero(np.diff(np.sign(Om)) != 0)
    for s in cruces:
        if s - ultimo < SEP_MIN / DT or s < 3 * w or s > len(Om) - 3 * w:
            continue
        antes = np.nanmean(Om[s - int(3 / DT):s - int(1 / DT)])
        despues = np.nanmean(Om[s + int(1 / DT):s + int(3 / DT)])
        if np.sign(antes) != np.sign(despues) and \
           abs(antes) > om_min and abs(despues) > om_min:
            eventos.append(s)
            ultimo = s
    return eventos, dphi


def analizar(fn, k=K_VEC, om_min=OM_MIN, q=Q, sf=1.0, cap=2.0, min_aves=None, min_finitos=None):
    X, Y = cargar(fn)
    Vx, Vy, om = cinematica(X, Y, sf)
    eventos, dphi = uturns(X, Y, om_min)
    if not eventos:
        return None
    T, N = om.shape

    # residuo contra los K vecinos más cercanos, por fotograma
    eps = np.full_like(om, np.nan)
    P = np.stack([X, Y], -1)
    for s in range(0, T):
        p, o = P[s], om[s]
        okk = np.isfinite(p).all(1) & np.isfinite(o)
        if okk.sum() < (min_finitos if min_finitos else k + 2):
            continue
        ii = np.flatnonzero(okk)
        D = np.linalg.norm(p[ii][:, None] - p[ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, 1)[:, :k]
        eps[s, ii] = o[ii] - o[ii][vec].mean(1)

    # línea base: lejos (>5 s) de cualquier giro
    lejos = np.ones(T, bool)
    for s in eventos:
        lejos[max(0, s - int(5 / DT)):s + int(5 / DT)] = False
    fb = np.isfinite(eps) & lejos[:, None]
    sig = float(np.nanmedian(np.abs(eps[fb] - np.nanmedian(eps[fb]))) * 1.4826)

    dphis = suav(dphi, int(0.5 / DT))
    res = {"eventos": [], "omega_max": [], "huellas": []}
    for s0 in eventos:
        # inversión por pez: cruce por cero de su dphi propio cerca del evento
        vent = slice(max(0, s0 - int(3 / DT)), min(T, s0 + int(3 / DT)))
        t_rev = np.full(N, np.nan)
        for a in range(N):
            tr = dphis[vent, a]
            okk = np.isfinite(tr)
            if okk.sum() < 20:
                continue
            cz = np.flatnonzero(np.diff(np.sign(tr[okk])) != 0)
            if len(cz):
                t_rev[a] = (vent.start + np.flatnonzero(okk)[cz[0]]) * DT
        okr = np.isfinite(t_rev)
        if okr.sum() < (min_aves if min_aves else max(4, int(N * 0.6))):
            continue
        orden = np.argsort(t_rev)
        d_rel = t_rev - np.nanmin(t_rev)
        lim_i = np.nanquantile(d_rel[okr], q)
        lim_s = np.nanquantile(d_rel[okr], 1 - q)
        ini = [a for a in range(N) if okr[a] and d_rel[a] <= lim_i]
        seg = [a for a in range(N) if okr[a] and d_rel[a] >= lim_s]

        # omega_max: p90 de |omega| en la ventana del giro
        vturn = slice(max(0, s0 - int(1 / DT)), min(T, s0 + int(2 / DT)))
        omv = np.abs(om[vturn])
        if np.isfinite(omv).any():
            res["omega_max"].append(float(np.nanpercentile(omv, 90)))
        # estimador alternativo, el de las grajillas de E16: a_lat/|v| (B5)
        vx, vy = Vx[vturn], Vy[vturn]
        ax = np.gradient(vx, DT, axis=0)
        ay = np.gradient(vy, DT, axis=0)
        rap = np.sqrt(vx**2 + vy**2)
        a_lat = np.abs(ax * (-vy) + ay * vx) / np.maximum(rap, 1e-9)
        oma = a_lat / np.maximum(rap, 1e-9)
        if np.isfinite(oma).any():
            res.setdefault("omega_max_alat", []).append(
                float(np.nanpercentile(oma, 90)))

        # huella del primer iniciador: |eps|/sig alineada a SU inversión
        a0 = int(orden[0])
        s_ini = int(round(t_rev[a0] / DT))
        tramo = np.abs(eps[s_ini:s_ini + int(cap / DT), a0]) / max(sig, 1e-9)
        if np.isfinite(tramo).mean() > 0.7:
            sm = suav(np.nan_to_num(tramo, nan=0.0), 9)
            ip = int(np.argmax(sm))
            dur = cap
            for j in range(ip, len(sm) - 9):
                if sm[j] < 1.5 and np.all(sm[j:j + 9] < 1.5):
                    dur = j * DT
                    break
            res["huellas"].append(dur)

        # estadísticos de ventana pre-colectiva para la AUC
        fila = {"ini": ini, "seg": seg, "stats": {}}
        for tau in TAUS:
            w0 = s0 - int(tau / DT)
            if w0 < 0:
                continue
            m = np.nanmean(np.abs(eps[w0:s0]) / max(sig, 1e-9), axis=0)
            fila["stats"][tau] = m.tolist()
        res["eventos"].append(fila)
    res["sigma"] = sig
    return res


def auc(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan, 0, 0
    a = ((x1[:, None] > x0[None, :]).sum()
         + 0.5 * (x1[:, None] == x0[None, :]).sum()) / (len(x1) * len(x0))
    return float(a), len(x1), len(x0)


if __name__ == "__main__":
    todo = {"omega_max": [], "huellas": [], "por_tau": {t: ([], []) for t in TAUS},
            "n_eventos": 0, "n_exp": 0}
    for grupo in ("8", "10"):
        for fn in sorted(glob.glob(os.path.join(DATA, grupo, "*.csv"))):
            r = analizar(fn)
            if r is None:
                continue
            todo["n_exp"] += 1
            todo["n_eventos"] += len(r["eventos"])
            todo["omega_max"] += r["omega_max"]
            todo["huellas"] += r["huellas"]
            for ev in r["eventos"]:
                for tau, vals in ev["stats"].items():
                    v = np.array(vals)
                    todo["por_tau"][tau][0].extend(v[ev["ini"]].tolist())
                    todo["por_tau"][tau][1].extend(v[ev["seg"]].tolist())
            print(f'{os.path.basename(fn):32s} {len(r["eventos"]):3d} giros U '
                  f'(sigma={r["sigma"]:.2f})')

    om_max = float(np.median(todo["omega_max"]))
    hue = float(np.median(todo["huellas"])) if todo["huellas"] else np.nan
    print(f'\n{todo["n_exp"]} experimentos, {todo["n_eventos"]} giros en U')
    print(f'omega_max mediana: {om_max:.2f} rad/s  (grajilla: 1.36)')
    print(f'huella del iniciador: {hue:.2f} s  (grajilla: 1.03; tope de ventana 2 s)')
    print(f'\n{"tau (s)":>8s} {"AUC ini/seg":>11s} {"n":>12s}')
    curva = {}
    for tau in TAUS:
        a, n1, n0 = auc(*todo["por_tau"][tau])
        curva[tau] = a
        print(f'{tau:8.1f} {a:11.3f} {n1:5d}/{n0:5d}')
    tau_star = max((t for t in curva if np.isfinite(curva[t])), key=lambda t: curva[t])
    print(f'\ntau* tetra = {tau_star:.1f} s  |  predicción biomecánica: '
          f'tau*_grajilla × (1.36/{om_max:.2f}) = {1.0 * 1.36 / om_max:.2f} s')

    with open(os.path.join(RAIZ, "resultados", "E27_tetras.json"), "w") as f:
        json.dump({"n_exp": todo["n_exp"], "n_eventos": todo["n_eventos"],
                   "omega_max": om_max, "huella": hue,
                   "curva_auc": curva, "tau_star": tau_star,
                   "prediccion_tau": 1.36 / om_max,
                   "grajilla": {"omega_max": 1.36, "tau_star": 1.0,
                                "huella": 1.03}}, f, indent=1)
