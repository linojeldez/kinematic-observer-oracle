"""Cargador del conjunto INDEPENDIENTE de grajillas: Supplementary Data 2 a 4 de

Ling H., McIvor G.E., Westley J., van der Vaart K., Vaughan R.T., Thornton A. &
Ouellette N.T. (2019), *Behavioural plasticity and the transition to order in jackdaw
flocks*, Nat. Commun. 10:5174 (doi 10.1038/s41467-019-13281-4).

No es el Dryad `10.5061/dryad.kb8js06` (Proc. R. Soc. B 286:20190865), como decía este
encabezado hasta el 11-nov-2026: de ese archivo el proyecto sólo usa
`code_upload/Group/Group_1..6.mat`, en E34.

Son **18 bandadas** —6 de tránsito y 12 de mobbing— con trayectorias 3D completas
(ID, x, y, z, t, u, v, w) y unas 990 000 filas en total. Es un conjunto **distinto**
de los 21 eventos curados de figshare que se usaron en E1-E4: aquéllos son ventanas
recortadas alrededor de un giro colectivo, éstas son grabaciones continuas.

Por eso vale como **conjunto de validación**: los cinco parámetros del detector se
calibraron en el otro y aquí no se ajusta nada.

Formato: bloques encabezados por una línea `transit flock 01` / `mobbing flock 04
part 1`, seguidos de filas de datos. La cabecera de columnas sólo aparece al
principio del fichero.
"""

from __future__ import annotations

import os
import re

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "datos", "ling2019_natcomm")
FICHEROS = ("DataS2.txt", "DataS3.txt", "DataS4.txt")


def bandadas(dir_=DIR, unicas=True):
    """Todas las bandadas del conjunto: dict con contexto, nombre y matriz (n, 8).

    Columnas de la matriz: ID, x, y, z, t, u, v, w  (SI).

    `unicas=True` descarta bloques repetidos. **Hace falta**: en `DataS2.txt`,
    `transit flock 03` y `transit flock 04` son el mismo vuelo duplicado --- los
    bloques crudos tienen el mismo MD5. Es un defecto del fichero publicado, no del
    lector, y contarlo dos veces sesga cualquier estadística entre bandadas.
    """
    out = []
    for fn in FICHEROS:
        ruta = os.path.join(dir_, fn)
        if not os.path.exists(ruta):
            continue
        nombre, filas = None, []
        with open(ruta) as fh:
            for linea in fh:
                s = linea.strip()
                if not s or s.startswith("ID"):
                    continue
                if re.match(r"^(transit|mobbing)\s+flock", s):
                    if nombre and filas:
                        out.append({"ctx": nombre.split()[0], "nombre": nombre,
                                    "d": np.array(filas, dtype=float)})
                    nombre, filas = s, []
                    continue
                p = s.split()
                if len(p) >= 8:
                    filas.append([float(v) for v in p[:8]])
        if nombre and filas:
            out.append({"ctx": nombre.split()[0], "nombre": nombre,
                        "d": np.array(filas, dtype=float)})
    if unicas:
        vistos, unic = set(), []
        for b in out:
            h = hash(b["d"].tobytes())
            if h in vistos:
                b["duplicado_de"] = next(x["nombre"] for x in unic
                                         if hash(x["d"].tobytes()) == h)
                continue
            vistos.add(h)
            unic.append(b)
        return unic
    return out


def simultaneidad(b):
    """Cuántas aves se siguen a la vez. Decide si una bandada sirve para nada.

    Estas grabaciones son continuas pero **fragmentadas**: una bandada puede tener
    167 trayectorias y sólo 6 aves visibles en un instante dado. Sin vecindario no
    hay residuo, ni contagio, ni nada. Se comprobó que sólo 6 de las 17 bandadas
    únicas superan las 10 aves simultáneas de mediana.
    """
    _, P, _ = campos(b)
    pres = np.isfinite(P).all(axis=2).sum(axis=1)
    return {"mediana": float(np.median(pres)), "p90": float(np.percentile(pres, 90)),
            "max": int(pres.max()), "frames_ge10": int((pres >= 10).sum()),
            "frames": int(len(pres))}


def campos(b, dt_tol=1e-4):
    """Reorganiza una bandada en matrices (tiempo, ave): posición y velocidad 3D."""
    d = b["d"]
    ids = np.unique(d[:, 0])
    ts = np.unique(np.round(d[:, 4], 4))
    i_id = {v: n for n, v in enumerate(ids)}
    i_t = {v: n for n, v in enumerate(ts)}
    P = np.full((len(ts), len(ids), 3), np.nan)
    V = np.full((len(ts), len(ids), 3), np.nan)
    for f in d:
        a, s = i_id[f[0]], i_t.get(round(f[4], 4))
        if s is None:
            continue
        P[s, a], V[s, a] = f[1:4], f[5:8]
    return ts, P, V


def residuos(b, k=7, modo="horizontal", max_pasos=4000):
    """El mismo residuo de intención de E2/E4, aquí sobre datos independientes.

        eps_j = omega_j - <omega>_{k vecinos más cercanos de j}

    `modo='horizontal'` usa la velocidad angular del rumbo proyectado en el plano
    —comparable punto por punto con lo medido en E2, que era 2D—. `modo='3d'` usa
    la tasa de giro del vector velocidad completo,
    |v x dv/dt| / |v|^2, que es la magnitud que un modelo 3D necesitaría.
    """
    ts, P, V = campos(b)
    n_t = min(len(ts), max_pasos)
    dt = float(np.median(np.diff(ts[:n_t])))
    if not np.isfinite(dt) or dt <= 0:
        return np.array([])

    # Aquí las aves entran y salen del campo de visión, así que cada columna tiene
    # huecos. Hay que derivar ave por ave sobre sus tramos contiguos: un solo NaN
    # en medio de un `unwrap` o de una convolución contamina la serie entera, y en
    # estas grabaciones eso vacía la matriz.
    Vn = V[:n_t]
    om = np.full(Vn.shape[:2], np.nan)
    ker = np.ones(5) / 5
    for a in range(Vn.shape[1]):
        ok = np.isfinite(Vn[:, a]).all(axis=1)
        if ok.sum() < 15:
            continue
        # tramos contiguos de presencia
        bordes = np.flatnonzero(np.diff(ok.astype(int)))
        ini = np.r_[0 if ok[0] else bordes[0] + 1, bordes[np.diff(ok.astype(int))[bordes] == 1] + 1]
        for i0 in np.unique(ini):
            i1 = i0
            while i1 < len(ok) and ok[i1]:
                i1 += 1
            if i1 - i0 < 15:
                continue
            seg = Vn[i0:i1, a]
            if modo == "horizontal":
                th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
                th = np.convolve(th, ker, mode="same")
                w = np.gradient(th, dt)
            else:
                dV = np.gradient(seg, dt, axis=0)
                w = np.linalg.norm(np.cross(seg, dV), axis=1) / \
                    np.maximum((seg ** 2).sum(axis=1), 1e-9)
            w[:3] = np.nan; w[-3:] = np.nan        # bordes del tramo, sin derivada fiable
            om[i0:i1, a] = w

    eps = []
    for s in range(2, n_t - 2, 3):                     # 1 de cada 3 pasos: basta
        p, o = P[s], om[s]
        ok = np.isfinite(p).all(axis=1) & np.isfinite(o)
        if ok.sum() < k + 2:
            continue
        idx = np.flatnonzero(ok)
        D = np.linalg.norm(p[idx][:, None, :] - p[idx][None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, axis=1)[:, :k]
        eps.append(o[idx] - o[idx][vec].mean(axis=1))
    return np.concatenate(eps) if eps else np.array([])


def resumen():
    """Inventario: cuántas bandadas, aves, duración y frecuencia de muestreo."""
    out = []
    for b in bandadas():
        d = b["d"]
        ts = np.unique(np.round(d[:, 4], 4))
        out.append({"nombre": b["nombre"], "ctx": b["ctx"],
                    "aves": int(len(np.unique(d[:, 0]))),
                    "frames": len(ts),
                    "dur_s": float(ts[-1] - ts[0]),
                    "fps": float(1.0 / np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
                    "filas": len(d)})
    return out
