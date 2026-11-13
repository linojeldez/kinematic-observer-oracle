"""
Los 21 eventos de giro colectivo de grajillas, con los retardos ya calculados.

Datos: `datos/giros/` — Ling H., McIvor G.E., Westley J., van der Vaart K., Yin J.,
Vaughan R.T., Thornton A. & Ouellette N.T. (2019), *Collective turns in jackdaw
flocks: kinematics and information transfer*, J. R. Soc. Interface 16:20190450.
figshare 10.6084/m9.figshare.10011374.v1

Estructura por evento:
    tracks            (frames x aves, 12): ID x y z t u v w ax ay az f_wb
    group_delay_NN    Delay (retardo de giro POR AVE), InfoDis, N, Order
    Info_turnStart_*  por evento: Ang (theta_s), Vsub2all (V_top), N_all, Ug, p_all

El índice i de `Delay` corresponde al i-ésimo ID en orden ascendente. Esto NO está
documentado por los autores; se verificó con un test de permutación contra un
cálculo independiente del retardo (correlación real muy por encima del nulo,
p < 0.001 en los nueve eventos probados). Ver `verificar_mapeo()`.

Ojo: `InfoDis` NO es una distancia medida — es (rango/densidad)^(1/3), la
"distancia de información" de Attanasi, función determinista del rango.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import scipy.io as sio

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "datos", "giros", "code")
CONTEXTOS = {"transito": "turns_transit", "mobbing": "turns_mobbing"}


def eventos(contexto=None):
    """Lista de (contexto, id_evento) disponibles."""
    out = []
    for ctx, sub in CONTEXTOS.items():
        if contexto and ctx != contexto:
            continue
        for g in sorted(glob.glob(os.path.join(DIR, sub, "group_[0-9]*.mat"))):
            out.append((ctx, re.search(r"group_(\d+)", g).group(1)))
    return out


def cargar(ctx, ev):
    """Un evento: trayectorias, retardos y metadatos, ya emparejados por ave."""
    sub = CONTEXTOS[ctx]
    t = sio.loadmat(os.path.join(DIR, sub, f"group_{ev}.mat"))["tracks"]
    d = sio.loadmat(os.path.join(DIR, sub, f"group_delay_{ev}.mat"))
    ids = np.unique(t[:, 0])
    delay = np.ravel(d["Delay"])
    assert len(ids) == len(delay), f"{ctx} {ev}: {len(ids)} aves vs {len(delay)} retardos"
    return {"ctx": ctx, "ev": ev, "tracks": t, "ids": ids,
            "delay": delay, "infodis": np.ravel(d["InfoDis"]),
            "orden": float(np.ravel(d["Order"])[0])}


def resumen_eventos():
    """Ang (theta_s) y Vsub2all (V_top) por evento, de Info_turnStart_{T,M}.mat."""
    out = {}
    for ctx, suf in [("transito", "T"), ("mobbing", "M")]:
        m = sio.loadmat(os.path.join(DIR, f"Info_turnStart_{suf}.mat"))
        out[ctx] = {"theta_s": np.ravel(m["Ang"]), "V_top": np.ravel(m["Vsub2all"]),
                    "N": np.ravel(m["N_all"]), "Ug": np.ravel(m["Ug"]),
                    "p": np.ravel(m["p_all"])}
    return out


# ----------------------------------------------------------------------------
# Predictores geométricos, en el instante inicial del evento
# ----------------------------------------------------------------------------


def predictores(e, k_vecinos=7):
    """Geometría de cada ave al inicio del giro.

    Devuelve un dict de arrays alineados con `e['delay']`:
        r_rel    distancia al centroide / radio de la bandada  (0 centro, 1 borde)
        n_loc    densidad local: nº de vecinos dentro de la distancia media global
        d_knn    distancia media a los k vecinos más cercanos
        frontal  proyección sobre la dirección de vuelo, normalizada (+1 delante)
        lateral  proyección sobre el eje de mayor elongación, normalizada
    """
    t, ids = e["tracks"], e["ids"]
    t0 = t[:, 4].min()
    fila = {int(i): None for i in ids}
    for i in ids:
        m = (t[:, 0] == i) & np.isclose(t[:, 4], t0)
        fila[int(i)] = t[m][0]
    P = np.array([fila[int(i)][1:4] for i in ids])          # posiciones
    V = np.array([fila[int(i)][5:8] for i in ids])          # velocidades

    c = P.mean(axis=0)
    dc = np.linalg.norm(P - c, axis=1)
    r_rel = dc / max(dc.max(), 1e-9)

    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    d_knn = np.sort(D, axis=1)[:, :min(k_vecinos, len(ids) - 1)].mean(axis=1)
    n_loc = (D < np.median(d_knn) * 2).sum(axis=1).astype(float)

    u = V.mean(axis=0)
    u = u / max(np.linalg.norm(u), 1e-9)
    frontal = (P - c) @ u
    frontal = frontal / max(np.abs(frontal).max(), 1e-9)

    # eje de mayor elongación (Attanasi 2015: los iniciadores están en sus extremos)
    w, Q = np.linalg.eigh(np.cov((P - c).T))
    ejel = Q[:, np.argmax(w)]
    lateral = (P - c) @ ejel
    lateral = np.abs(lateral) / max(np.abs(lateral).max(), 1e-9)

    return {"r_rel": r_rel, "n_loc": n_loc, "d_knn": d_knn,
            "frontal": frontal, "lateral": lateral, "pos": P, "vel": V}


def tabla(k_vecinos=7, min_aves=10):
    """Una fila por ave, de todos los eventos con al menos `min_aves`.

    Columnas: ctx, ev, delay_z (retardo tipificado DENTRO del evento), y los
    predictores geométricos también tipificados dentro del evento. Tipificar por
    evento elimina las diferencias de escala entre bandadas y deja la pregunta
    en su forma limpia: ¿pesa igual la geometría en los dos contextos?
    """
    filas = []
    for ctx, ev in eventos():
        e = cargar(ctx, ev)
        if len(e["ids"]) < min_aves:
            continue
        g = predictores(e, k_vecinos)
        z = lambda a: (a - a.mean()) / max(a.std(), 1e-9)
        n = len(e["ids"])
        for i in range(n):
            filas.append({"ctx": ctx, "ev": ev, "N": n,
                          "delay": e["delay"][i], "delay_z": z(e["delay"])[i],
                          "r_rel": z(g["r_rel"])[i], "n_loc": z(g["n_loc"])[i],
                          "d_knn": z(g["d_knn"])[i], "frontal": z(g["frontal"])[i],
                          "lateral": z(g["lateral"])[i]})
    return filas


# ----------------------------------------------------------------------------
# Verificación del mapeo ave <-> retardo
# ----------------------------------------------------------------------------


def _retardo_propio(e):
    """Instante en que cada ave completa el 50 % de su giro horizontal."""
    t, ids = e["tracks"], e["ids"]
    out = []
    for i in ids:
        m = t[:, 0] == i
        tt, u, v = t[m, 4], t[m, 5], t[m, 6]
        th = np.unwrap(np.arctan2(v, u))
        tot = th[-1] - th[0]
        if abs(tot) < np.deg2rad(5):
            out.append(np.nan)
            continue
        out.append(tt[np.argmax(np.abs(th - th[0]) >= 0.5 * abs(tot))])
    return np.array(out)


def verificar_mapeo(n_perm=2000, seed=0, min_aves=15):
    """Test de permutación: ¿Delay[i] es el ave del i-ésimo ID ordenado?

    Si el emparejamiento fuera arbitrario, la correlación entre `Delay` y un
    cálculo independiente del retardo sería la del azar.
    """
    rng = np.random.default_rng(seed)
    out = []
    for ctx, ev in eventos():
        e = cargar(ctx, ev)
        if len(e["ids"]) < min_aves:
            continue
        mio = _retardo_propio(e)
        ok = np.isfinite(mio)
        a, b = e["delay"][ok], mio[ok]
        if len(a) < 10:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        nulo = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
        out.append({"ctx": ctx, "ev": ev, "n": int(ok.sum()), "r": r,
                    "p": float((nulo >= r).mean()), "nulo_p95": float(np.percentile(nulo, 95))})
    return out
