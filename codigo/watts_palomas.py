"""Cargador de Watts et al. 2016 (Biol. Lett. 12:20160544; Dryad 10.5061/dryad.508j2).

8 bandadas de 5 palomas con GPS a 5 Hz. Archivos `01_FF_BIRD_RR.csv`: bandada FF, ave BIRD, suelta
RR. Protocolo (manuscrito aceptado, fig. 1a y métodos): sueltas 1-8 entrenamiento (etapa 0);
9 líder con el reloj adelantado (etapa 1); 10 líder con el reloj atrasado (etapa 2); 11 y 12
control sin desplazamiento (etapa 3); 13 bandada entera adelantada (etapa 4); 14 control (etapa
5); 15 bandada entera atrasada (etapa 6). Líder de la etapa 0: el ave con mayor rango medio de
posición (adelante-atrás) en las sueltas 5 a 8.

Procesamiento como en el artículo: coordenadas métricas, puntos de vuelo (rapidez sostenida sobre
5 m/s durante 10 s antes o después) y recorte al llegar a 100 m del palomar.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from datetime import datetime, timezone

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(RAIZ, "datos", "watts_palomas", "Data")
R_T = 6371000.0
SUELTA_LAT, SUELTA_LON = 51 + 49 / 60 + 23.48 / 3600, -(1 + 21 / 60 + 26.29 / 3600)
DT = 0.2
ETAPA = {**{r: 0 for r in range(1, 9)}, 9: 1, 10: 2, 11: 3, 12: 3, 13: 4, 14: 5, 15: 6}
CONDICION = {0: "entrenamiento", 1: "lider_adelantado", 2: "lider_atrasado", 3: "control",
             4: "bandada_adelantada", 5: "control", 6: "bandada_atrasada"}


def _xy(lat, lon):
    x = np.deg2rad(lon - SUELTA_LON) * R_T * np.cos(np.deg2rad(SUELTA_LAT))
    y = np.deg2rad(lat - SUELTA_LAT) * R_T
    return x, y


# palomar: a 5,27 km de la suelta, con la suelta a 149,5° desde el palomar
_b = np.deg2rad(149.5 + 180.0)
PALOMAR = (5270.0 * np.sin(_b), 5270.0 * np.cos(_b))


def archivos():
    out = {}
    for fn in sorted(glob.glob(os.path.join(DIR, "01_*_*_*.csv"))):
        m = re.match(r"01_(\d+)_(\w+?)_(\d+)\.csv", os.path.basename(fn))
        if m:
            out[(int(m.group(1)), m.group(2), int(m.group(3)))] = fn
    return out


def leer(fn):
    """Tiempo (s UTC), x, y (m, relativos a la suelta) y altitud; sólo puntos FIXED."""
    t, lat, lon, alt = [], [], [], []
    with open(fn, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("VALID", "").strip() not in ("FIXED", "DGPS", "SPS", "ESTIMATED") \
                    and not row.get("VALID", "").strip().startswith("FIX"):
                continue
            try:
                d = datetime.strptime(f'{row["UTC DATE"]} {row["UTC TIME"]}', "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue
            t.append(d.replace(tzinfo=timezone.utc).timestamp() + float(row["MS"]) / 1000.0)
            la, lo = float(row["LATITUDE"]), float(row["LONGITUDE"])
            lat.append(-la if row["N/S"].strip() == "S" else la)
            lon.append(-lo if row["E/W"].strip() == "W" else lo)
            alt.append(float(row["ALTITUDE"]))
    t = np.array(t)
    o = np.argsort(t)
    x, y = _xy(np.array(lat)[o], np.array(lon)[o])
    return t[o], x, y, np.array(alt)[o]


def vuelo(t, x, y):
    """Máscara de vuelo como en el artículo y recorte a 100 m del palomar."""
    v = np.hypot(np.gradient(x, t), np.gradient(y, t))
    rapido = v > 5.0
    n = int(round(10.0 / DT))
    k = np.ones(n)
    atras = np.convolve(rapido.astype(float), k, "full")[:len(v)] >= n - 0.5      # los n anteriores
    adelante = np.convolve(rapido.astype(float)[::-1], k, "full")[:len(v)][::-1] >= n - 0.5
    m = rapido & (atras | adelante)
    # vuelo de regreso: desde la última vez en vuelo a menos de 500 m de la suelta hasta la
    # primera llegada a menos de 100 m del palomar (los registros traen otros vuelos del día)
    ds = np.hypot(x, y)
    d = np.hypot(x - PALOMAR[0], y - PALOMAR[1])
    llega = np.flatnonzero(m & (d < 100.0))
    fin = llega[0] if llega.size else len(m)
    cerca = np.flatnonzero(m[:fin] & (ds[:fin] < 500.0))
    ini = cerca[-1] if cerca.size else 0
    m[:ini] = False
    m[fin:] = False
    return m


def suelta(flock, rel, arch=None):
    """Matrices (tiempo, ave) en una rejilla común de 0,2 s, sólo en vuelo."""
    arch = arch or archivos()
    aves = sorted(b for (f, b, r) in arch if f == flock and r == rel)
    series = {}
    for b in aves:
        t, x, y, _ = leer(arch[(flock, b, rel)])
        if len(t) < 100:
            continue
        m = vuelo(t, x, y)
        if m.sum() < 100:
            continue
        series[b] = (np.round(t[m] / DT).astype(np.int64), x[m], y[m])
    if not series:
        return None
    t0 = min(s[0][0] for s in series.values())
    t1 = max(s[0][-1] for s in series.values())
    T = int(t1 - t0 + 1)
    nombres = sorted(series)
    X = np.full((T, len(nombres)), np.nan)
    Y = np.full_like(X, np.nan)
    for j, b in enumerate(nombres):
        it, x, y = series[b]
        X[it - t0, j], Y[it - t0, j] = x, y
    return {"aves": nombres, "X": X, "Y": Y, "t0": t0 * DT}


def rango_posicion(s, d_max=100.0):
    """Rango por posición adelante-atrás (método del artículo): proyección de la distancia al
    centro de la bandada sobre su dirección de movimiento, promediada en los cuadros con al menos
    3 aves dentro de d_max del centro. Devuelve el promedio por ave (mayor = más adelante)."""
    X, Y = s["X"], s["Y"]
    Vx, Vy = np.gradient(X, DT, axis=0), np.gradient(Y, DT, axis=0)
    ok = np.isfinite(X) & np.isfinite(Vx)
    cx, cy = np.nanmean(np.where(ok, X, np.nan), 1), np.nanmean(np.where(ok, Y, np.nan), 1)
    cerca = ok & (np.hypot(X - cx[:, None], Y - cy[:, None]) < d_max)
    vx = np.nanmean(np.where(cerca, Vx, np.nan), 1); vy = np.nanmean(np.where(cerca, Vy, np.nan), 1)
    nrm = np.hypot(vx, vy)
    proy = ((X - cx[:, None]) * (vx / nrm)[:, None] + (Y - cy[:, None]) * (vy / nrm)[:, None])
    filas = cerca.sum(1) >= 3
    proy = np.where(cerca & filas[:, None], proy, np.nan)
    return {b: float(np.nanmean(proy[:, j])) for j, b in enumerate(s["aves"])}, int(filas.sum())
