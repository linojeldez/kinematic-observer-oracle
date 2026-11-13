"""Etapa 51b: control de verdad conocida para T95 en el ruido real de las grajillas.

A un ave no iniciadora se le suma en su velocidad angular el residuo con signo del agente
informado de una corrida del modelo (WiSt o CoLd) y se mide su T95 frente a las demás no
iniciadoras, con el mismo procedimiento de E51. Control añadido después de ver E51; diseño y
predicción en docs/bitacora/resultados_E46-E55.md.

    PYTHONPATH=codigo python codigo/e51b_inyeccion.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import turns
from e16_helpers import campos, omega, residuos, inicio_colectivo, inicio_individual
from e51_huella_aves import suavizar, ic_mediana, T_SUAV, T_POST, T_PRE, Q_INI

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_REP_INY, K = 5, 7


def traza(arg):
    """Residuo con signo del agente informado, cada 0,01 s durante 2 s (rad/s)."""
    from flocking import Flock, Params
    m, rep = arg
    CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
    q = Params(N=200, k=12, sigma_xi=float(np.sqrt(2.0)), seed=11 + rep, **CFG)
    f = Flock(q, m)
    for _ in range(int(round(8.0 / q.dt))):
        f.step()
    ag, ang = 7, 2.0
    f.will[ag] = 1.0
    f.d_des[ag] = np.array([np.cos(ang), np.sin(ang)])
    out = []
    for s in range(int(round(2.0 / q.dt))):
        f.step()
        f.will[ag] = 1.0
        i, _ = f._neighbors()
        out.append(float(f.omega[ag] - f.omega[i[ag]].mean()))
    # el motor promedia sobre el agente y sus k vecinos; las aves, sólo sobre los vecinos
    return m, rep, (np.array(out) * (q.k + 1) / q.k).tolist()


def evento(arg):
    ctx, ev, trazas, semilla = arg
    rng = np.random.default_rng(semilla)
    e = turns.cargar(ctx, ev)
    if len(e["ids"]) < 12:
        return None
    ts, P, V = campos(e)
    dt = float(np.median(np.diff(ts)))
    om = omega(V, dt)
    eps0 = residuos(P, om, K)
    s_col = inicio_colectivo(V)
    base = eps0[:s_col] if s_col and s_col > 20 else eps0
    fb = np.isfinite(base)
    sig = float(np.median(np.abs(base[fb] - np.median(base[fb]))) * 1.4826)
    n_suav = max(3, int(round(T_SUAV / dt)))
    d = e["delay"]
    ok = np.isfinite(d)
    lim = np.quantile(d[ok], Q_INI)
    ini = [a for a in np.flatnonzero(ok) if d[a] <= lim]
    no_ini = np.array([a for a in np.flatnonzero(ok) if d[a] > lim])
    n_pre, n_post = int(round(T_PRE / dt)), int(round(T_POST / dt))
    vent = []
    for a in ini:
        s0 = inicio_individual(V, a)
        if s0 is None or s0 - n_pre < 0 or s0 + n_post >= len(ts):
            continue
        vent.append(s0)
    if len(vent) < 3:
        return None
    tt = np.arange(n_post) * dt
    out = {"ev": f"{ctx} {ev}", "sigma": sig, "N": int(len(e["ids"]))}
    for m in ("WiSt", "CoLd"):
        vals, picos = [], []
        for s0 in vent:
            for _ in range(N_REP_INY):
                foco = int(rng.choice(no_ini))
                tr = np.array(trazas[m][int(rng.integers(len(trazas[m])))])
                delta = np.interp(tt, np.arange(len(tr)) * 0.01, tr)
                om2 = om.copy()
                om2[s0:s0 + n_post, foco] += delta
                # sólo cambian los cuadros de la ventana: se recalculan ahí
                eps = eps0.copy()
                eps[s0:s0 + n_post] = residuos(P[s0:s0 + n_post], om2[s0:s0 + n_post], K)
                lo, hi = max(0, s0 - n_suav), min(len(ts), s0 + n_post + n_suav)
                A = suavizar(np.abs(eps[lo:hi]) / sig, n_suav)[s0 - lo:s0 - lo + n_post]
                resto = no_ini[no_ini != foco]
                thr = np.nanpercentile(A[:, resto], 95)
                vals.append(float(np.nansum(A[:, foco] > thr) * dt))
                picos.append(float(np.nanmax(A[:, foco])))
        out[m] = {"T95": float(np.median(vals)), "T95_media": float(np.mean(vals)),
                  "pico_sigma": float(np.median(picos)), "n": len(vals)}
    return out


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(511)
    with ProcessPoolExecutor(max_workers=8) as ex:
        tr = list(ex.map(traza, [(m, r) for m in ("WiSt", "CoLd") for r in range(16)]))
        trazas = {m: [t for mm, _, t in tr if mm == m] for m in ("WiSt", "CoLd")}
        evs = list(ex.map(evento, [(c, e, trazas, 1000 + i)
                                   for i, (c, e) in enumerate(turns.eventos())]))
    evs = [x for x in evs if x]
    L = ["=== E51b · T95 de una huella del modelo inyectada en aves no iniciadoras ===",
         f'{"evento":<12s} {"σ (rad/s)":>9s} {"WiSt T95":>9s} {"pico":>5s} {"CoLd T95":>9s} {"pico":>5s}']
    for x in evs:
        L.append(f'{x["ev"]:<12s} {x["sigma"]:9.3f} {x["WiSt"]["T95"]:9.2f} {x["WiSt"]["pico_sigma"]:5.1f} '
                 f'{x["CoLd"]["T95"]:9.2f} {x["CoLd"]["pico_sigma"]:5.1f}')
    out = {"eventos": evs, "agregado": {}}
    for m in ("WiSt", "CoLd"):
        v = [x[m]["T95"] for x in evs]
        out["agregado"][m] = {"mediana": float(np.median(v)), "ic95": ic_mediana(v, rng),
                              "min": float(min(v)), "max": float(max(v))}
        L.append(f'{m}: mediana sobre eventos {np.median(v):.2f} s, IC 95 % '
                 f'{out["agregado"][m]["ic95"]}, intervalo entre eventos {min(v):.2f} a {max(v):.2f} s')
    with open("resultados/E51b_inyeccion.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open("resultados/E51b_inyeccion.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
