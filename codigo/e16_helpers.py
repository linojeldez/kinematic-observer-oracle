"""Instrumentación de la Etapa 16: medir la huella en grajillas reales.

La tesis del proyecto ---que la firma cinemática de una decisión se apaga mucho antes
que el estado que la produjo--- se ha medido sólo en simulación. Los datos reales se
usaron para *calibrar* el detector, no para poner a prueba la afirmación central.

Aquí se mide en los 21 eventos curados, que traen algo que hace esto posible: el
vector `Delay`, el retardo de giro **por ave**, calculado por los propios autores.
Con él se sabe quién inició cada giro sin tener que inferirlo.

El diseño es el mismo que en simulación pero al revés: allí se fijaba la intención y
se seguía el residuo; aquí se identifica a quien inició y se mira su residuo
**alineado por el instante en que empezó a girar**.
"""

from __future__ import annotations

import numpy as np

import turns


def campos(e):
    """Un evento en matrices (tiempo, ave): posición, velocidad, tiempos."""
    t, ids = e["tracks"], e["ids"]
    ts = np.unique(t[:, 4])
    idx = {int(v): n for n, v in enumerate(ids)}
    tp = {round(float(v), 6): n for n, v in enumerate(ts)}
    P = np.full((len(ts), len(ids), 3), np.nan)
    V = np.full_like(P, np.nan)
    for f in t:
        s, b = tp[round(float(f[4]), 6)], idx[int(f[0])]
        P[s, b], V[s, b] = f[1:4], f[5:8]
    return ts, P, V


def omega(V, dt, suav=5):
    """Velocidad angular horizontal, suavizada."""
    th = np.unwrap(np.arctan2(V[:, :, 1], V[:, :, 0]), axis=0)
    k = np.ones(suav) / suav
    th = np.apply_along_axis(lambda a: np.convolve(a, k, mode="same"), 0, th)
    return np.gradient(th, dt, axis=0)


def residuos(P, om, k=7):
    """eps_j(t) = omega_j - <omega>_{k vecinos más cercanos}, en cada instante."""
    eps = np.full_like(om, np.nan)
    for s in range(om.shape[0]):
        p, o = P[s], om[s]
        ok = np.isfinite(p).all(axis=1) & np.isfinite(o)
        if ok.sum() < k + 2:
            continue
        idx = np.flatnonzero(ok)
        D = np.linalg.norm(p[idx][:, None, :] - p[idx][None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        vec = np.argsort(D, axis=1)[:, :k]
        eps[s, idx] = o[idx] - o[idx][vec].mean(axis=1)
    return eps


def inicio_individual(V, a, frac=0.10):
    """Instante (índice) en que el ave `a` empieza su propio giro.

    Primer cruce del `frac` de su cambio total de rumbo --- el mismo criterio que se
    usó en E2 para el giro colectivo, aplicado ahora a cada individuo.
    """
    v = V[:, a]
    ok = np.isfinite(v).all(axis=1)
    if ok.sum() < 20:
        return None
    th = np.unwrap(np.arctan2(v[ok, 1], v[ok, 0]))
    tot = th[-1] - th[0]
    if abs(tot) < np.deg2rad(10):
        return None
    h = np.flatnonzero(np.abs(th - th[0]) >= frac * abs(tot))
    if h.size == 0:
        return None
    return int(np.flatnonzero(ok)[h[0]])


def inicio_colectivo(V, frac=0.10):
    """Instante en que la BANDADA empieza a girar. Todo lo anterior es línea base."""
    u = np.nanmean(V[:, :, :2], axis=1)
    ok = np.isfinite(u).all(axis=1)
    if ok.sum() < 30:
        return None
    th = np.unwrap(np.arctan2(u[ok, 1], u[ok, 0]))
    tot = th[-1] - th[0]
    if abs(tot) < np.deg2rad(5):
        return None
    hit = np.flatnonzero(np.abs(th - th[0]) >= frac * abs(tot))
    return int(np.flatnonzero(ok)[hit[0]]) if hit.size else None


def huella_evento(e, k=7, q_ini=0.25, t_pre=0.3, t_post=2.0):
    """Residuo de iniciadores y de seguidores, alineado por su propio inicio de giro.

    `q_ini` define quién cuenta como iniciador: el cuartil de menor `Delay`, que es
    la misma etiqueta que E2 usó para la AUC. Los seguidores ---el cuartil de mayor
    retardo--- son el control: si el exceso de residuo fuese un artefacto del
    alineamiento, aparecería igual en ellos.

    Devuelve las curvas |eps|(t) promediadas, en unidades de la anchura robusta de
    la línea base del propio evento, para que los eventos sean comparables.

    Las ventanas son cortas (0.3 s antes, 2 s después) porque **los eventos lo son**:
    duran 3.6-5.0 s en total, al ser recortes alrededor de un giro. Con 2 s basta para
    la pregunta que importa ---si la firma se apaga en ~1 s o dura todo el giro--- pero
    impide seguirla más allá. Es una limitación del dato, no del análisis.
    """
    ts, P, V = campos(e)
    dt = float(np.median(np.diff(ts)))
    om = omega(V, dt)
    eps = residuos(P, om, k)
    fin = np.isfinite(eps)
    if fin.sum() < 200:
        return None
    # sigma de la LÍNEA BASE, no del evento entero: durante el giro todas las aves
    # tienen residuos grandes y la anchura sale inflada, con lo que cualquier firma
    # parecería pequeña en unidades de sigma. Es la misma corrección que hizo falta
    # en E2 para que la AUC dejara de estar contaminada.
    s_col = inicio_colectivo(V)
    base = eps[:s_col] if s_col and s_col > 20 else eps
    fb = np.isfinite(base)
    sig = float(np.median(np.abs(base[fb] - np.median(base[fb]))) * 1.4826) \
        if fb.sum() > 100 else \
        float(np.median(np.abs(eps[fin] - np.median(eps[fin]))) * 1.4826)

    d = e["delay"]
    ok = np.isfinite(d)
    if ok.sum() < 8:
        return None
    lim_ini = np.quantile(d[ok], q_ini)
    lim_seg = np.quantile(d[ok], 1 - q_ini)

    n_pre, n_post = int(round(t_pre / dt)), int(round(t_post / dt))
    curvas = {"ini": [], "seg": []}
    for a in np.flatnonzero(ok):
        grupo = "ini" if d[a] <= lim_ini else ("seg" if d[a] >= lim_seg else None)
        if grupo is None:
            continue
        s0 = inicio_individual(V, a)
        if s0 is None or s0 - n_pre < 0 or s0 + n_post >= len(ts):
            continue
        tramo = np.abs(eps[s0 - n_pre:s0 + n_post, a]) / max(sig, 1e-9)
        if np.isfinite(tramo).mean() > 0.7:
            curvas[grupo].append(tramo)

    if len(curvas["ini"]) < 3:
        return None
    tt = (np.arange(-n_pre, n_post)) * dt
    out = {"t": tt, "sigma": sig, "dt": dt, "n_ini": len(curvas["ini"]),
           "n_seg": len(curvas["seg"])}
    for g in ("ini", "seg"):
        if curvas[g]:
            M = np.vstack(curvas[g])
            out[g] = np.nanmedian(M, axis=0)
            out[g + "_n"] = M.shape[0]
    return out


def duracion(t, curva, umbral=1.5, suav=9):
    """Cuánto se mantiene la curva por encima del umbral tras su pico.

    Mismo criterio que en simulación (E15): primer descenso sostenido, no el último
    cruce ---que con fluctuaciones daría siempre la ventana entera.
    """
    sm = np.convolve(curva, np.ones(suav) / suav, mode="same")
    pos = t >= 0
    if not pos.any():
        return np.nan
    i0 = int(np.flatnonzero(pos)[0])
    ip = i0 + int(np.nanargmax(sm[pos]))
    for j in range(ip, len(sm) - suav):
        if sm[j] < umbral and np.all(sm[j:j + suav] < umbral):
            return float(t[j])
    return float(t[-1])


# ----------------------------------------------------------------------------
# Parte 2 — la fórmula cinemática, contrastada con las aceleraciones reales
# ----------------------------------------------------------------------------


def cinematica(e, q_ini=0.25):
    """Velocidad, aceleración lateral y amplitud de giro de cada iniciador.

    E15 derivó en simulación que la huella dura $\\Delta\\varphi\\,v_0/a_{max}$. En un
    ave la magnitud que limita el giro no es la aceleración total sino su componente
    **lateral** ---la perpendicular a la velocidad, que es la que curva la
    trayectoria; la longitudinal sólo cambia la rapidez.

    Los 21 eventos traen `ax, ay, az` (columnas 8-10), así que las tres cantidades de
    la fórmula son medibles y la predicción se puede contrastar sin ajustar nada.
    """
    t, ids = e["tracks"], e["ids"]
    ts = np.unique(t[:, 4])
    idx = {int(v): n for n, v in enumerate(ids)}
    tp = {round(float(v), 6): n for n, v in enumerate(ts)}
    V = np.full((len(ts), len(ids), 3), np.nan)
    A = np.full_like(V, np.nan)
    for f in t:
        s, b = tp[round(float(f[4]), 6)], idx[int(f[0])]
        V[s, b], A[s, b] = f[5:8], f[8:11]

    d = e["delay"]
    ok = np.isfinite(d)
    if ok.sum() < 8:
        return []
    lim = np.quantile(d[ok], q_ini)

    out = []
    for a in np.flatnonzero(ok):
        if d[a] > lim:
            continue
        v, ac = V[:, a], A[:, a]
        m = np.isfinite(v).all(axis=1) & np.isfinite(ac).all(axis=1)
        if m.sum() < 20:
            continue
        vv, aa = v[m], ac[m]
        rap = np.linalg.norm(vv, axis=1)
        vhat = vv / np.maximum(rap[:, None], 1e-9)
        a_lat = np.linalg.norm(aa - (aa * vhat).sum(1)[:, None] * vhat, axis=1)
        # Amplitud del GIRO INICIAL, no del rumbo acumulado en todo el evento.
        # Medir th[-1]-th[0] da 184 grados de mediana porque en mobbing las aves
        # circulan alrededor de la amenaza y el unwrap suma vuelta tras vuelta: eso
        # no es la maniobra cuya firma se está midiendo.
        th = np.unwrap(np.arctan2(vv[:, 1], vv[:, 0]))
        i0 = inicio_individual(V, a)
        if i0 is None:
            continue
        j0 = int(np.searchsorted(np.flatnonzero(m), i0))
        j1 = min(j0 + int(round(2.0 / max(np.median(np.diff(ts)), 1e-6))), len(th) - 1)
        if j1 - j0 < 5:
            continue
        dphi = float(abs(th[j1] - th[j0]))
        if dphi < np.deg2rad(10):
            continue
        v_med = float(np.median(rap))
        a_p90 = float(np.percentile(a_lat, 90))     # la que sostiene el giro
        out.append({"v": v_med, "a_lat": a_p90, "dphi": dphi,
                    "t_pred": dphi * v_med / max(a_p90, 1e-9),
                    "omega_max": a_p90 / max(v_med, 1e-9)})
    return out
