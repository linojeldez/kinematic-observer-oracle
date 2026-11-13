"""Etapa 57 (plan 13): palomas con líder desinformado (Watts et al. 2016).

Diseño, regla de identificación del líder y predicciones Q1-Q5 en
docs/bitacora/resultados_E56-E65.md, declarados antes de correr.

    python -I codigo/e57_palomas.py      (lee datos descargados: modo aislado)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np                                                    # noqa: E402

import watts_palomas as w                                             # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = w.DT
VENT = (0.2, 0.6, 1.0, 2.0, 4.0, 10.0)
D_VEC = 50.0
DESPLAZ = {9: +1, 10: -1}            # +1 antihorario esperado (adelantado), -1 horario
CONTROL = (11, 12, 14)
PREVIAS = (7, 8)
N_BOOT = 10000


def suav(x, n=3):
    k = np.ones(n)
    f = np.isfinite(x)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, np.where(f, x, 0.0))
    den = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, f.astype(float))
    return np.where(den >= n - 0.5, num / np.maximum(den, 1e-12), np.nan)


def cinematica(s):
    X, Y = suav(s["X"]), suav(s["Y"])
    Vx, Vy = np.gradient(X, DT, axis=0), np.gradient(Y, DT, axis=0)
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    for a in range(th.shape[1]):
        idx = np.flatnonzero(np.isfinite(th[:, a]))
        if len(idx) < 10:
            continue
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) < 5:
                continue
            om[seg, a] = np.gradient(np.unwrap(th[seg, a]), DT)
    return X, Y, th, suav(om)


def residuos(X, Y, th, om):
    T, N = X.shape
    eps = np.full((T, N), np.nan)
    off = np.full((T, N), np.nan)
    for t in range(T):
        ok = np.isfinite(X[t]) & np.isfinite(om[t]) & np.isfinite(th[t])
        if ok.sum() < 3:
            continue
        ii = np.flatnonzero(ok)
        D = np.hypot(X[t, ii][:, None] - X[t, ii][None, :], Y[t, ii][:, None] - Y[t, ii][None, :])
        np.fill_diagonal(D, np.inf)
        for a_i, a in enumerate(ii):
            vec = ii[D[a_i] < D_VEC]
            if len(vec) < 2:
                continue
            eps[t, a] = om[t, a] - om[t, vec].mean()
            m = np.angle(np.exp(1j * th[t, vec]).mean())
            off[t, a] = np.angle(np.exp(1j * (th[t, a] - m)))
    return eps, off


def ventana(x, n):
    """Media de los n valores anteriores (incluido el actual); NaN si falta más del 30 %."""
    if n <= 1:
        return x.copy()
    f = np.isfinite(x)
    k = np.ones(n)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "full")[:len(a)], 0, np.where(f, x, 0.0))
    den = np.apply_along_axis(lambda a: np.convolve(a, k, "full")[:len(a)], 0, f.astype(float))
    return np.where(den >= 0.7 * n, num / np.maximum(den, 1e-12), np.nan)


def auc_dentro(S, j):
    """P(S del ave j > S de un compañero), en los mismos instantes."""
    sj = S[:, j][:, None]
    otros = np.delete(S, j, axis=1)
    ok = np.isfinite(sj) & np.isfinite(otros)
    if ok.sum() < 50:
        return np.nan, 0
    g = ((sj > otros) & ok).sum() + 0.5 * ((sj == otros) & ok).sum()
    return float(g / ok.sum()), int(ok.sum())


def dcd(s):
    """Retardo de correlación direccional medio por ave (positivo = lidera), Nagy 2010."""
    X, Y = suav(s["X"]), suav(s["Y"])
    Vx, Vy = np.gradient(X, DT, axis=0), np.gradient(Y, DT, axis=0)
    n = np.hypot(Vx, Vy)
    ux, uy = Vx / n, Vy / n
    N = X.shape[1]
    lags = range(-10, 11)                                   # ±2 s
    sc = {a: [] for a in range(N)}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            best, tau = -2, 0
            for L in lags:
                if L >= 0:
                    a, b = slice(0, len(X) - L), slice(L, len(X))
                else:
                    a, b = slice(-L, len(X)), slice(0, len(X) + L)
                c = ux[a, i] * ux[b, j] + uy[a, i] * uy[b, j]
                d = np.hypot(X[a, i] - X[b, j], Y[a, i] - Y[b, j])
                c = c[np.isfinite(c) & (d < D_VEC)]
                if len(c) < 100:
                    continue
                if c.mean() > best:
                    best, tau = c.mean(), L * DT
            sc[i].append(tau)                               # tau > 0: j repite a i con retraso
    return {s["aves"][a]: float(np.mean(v)) if v else np.nan for a, v in sc.items()}


def lider(f, arch):
    pos, pr, tau = {}, {}, {}
    for r in (5, 6, 7, 8):
        s = w.suelta(f, r, arch)
        if s is None:
            continue
        rk, _ = w.rango_posicion(s)
        vals = {b: v for b, v in rk.items() if np.isfinite(v)}
        for i, b in enumerate(sorted(vals, key=lambda b: -vals[b])):
            pos.setdefault(b, []).append(i + 1); pr.setdefault(b, []).append(vals[b])
        for b, t in dcd(s).items():
            if np.isfinite(t):
                tau.setdefault(b, []).append(t)
    cand = sorted(pos, key=lambda b: np.mean(pos[b]))
    a, b = cand[0], cand[1]
    claro = bool(np.mean(pos[b]) - np.mean(pos[a]) >= 0.5)
    if not claro:
        ta, tb = np.mean(tau.get(a, [np.nan])), np.mean(tau.get(b, [np.nan]))
        if np.isfinite(tb) and (not np.isfinite(ta) or tb > ta):
            a, b = b, a
    resto = [c for c in cand if c not in (a, b)]
    return {"lider": a, "segundo": b, "claro": claro,
            "rango": {c: float(np.mean(pos[c])) for c in cand},
            "dcd": {c: float(np.mean(tau[c])) for c in tau}, "otros": resto}


def por_suelta(args):
    f, r, ids, dist_max = args
    arch = w.archivos()
    s = w.suelta(f, r, arch)
    if s is None:
        return None
    X, Y, th, om = cinematica(s)
    eps, off = residuos(X, Y, th, om)
    if dist_max is not None:
        lejos = np.hypot(np.nanmean(X, 1), np.nanmean(Y, 1)) > dist_max
        eps[lejos] = np.nan; off[lejos] = np.nan
    out = {"f": f, "r": r, "P": {}, "n": {}, "off": {}, "s_juntas": float(np.isfinite(eps).any(1).sum() * DT)}
    for b in ids:
        if b not in s["aves"]:
            continue
        j = s["aves"].index(b)
        for v in VENT:
            S = ventana(np.abs(eps), int(round(v / DT)))
            p, n = auc_dentro(S, j)
            out["P"][f"{b}|{v:g}"], out["n"][f"{b}|{v:g}"] = p, n
        o = off[:, j]
        out["off"][b] = float(np.nanmean(o)) if np.isfinite(o).sum() >= 50 else np.nan
    return out


def resumen(D, rng):
    D = np.array([d for d in D if np.isfinite(d)])
    if len(D) < 3:
        return {"media": float("nan"), "n": int(len(D))}
    b = rng.integers(0, len(D), (N_BOOT, len(D)))
    m = D[b].mean(1)
    signos = np.array(np.meshgrid(*[[-1, 1]] * len(D))).reshape(len(D), -1).T
    nulo = (signos * np.abs(D)).mean(1)
    return {"media": float(D.mean()), "ic95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "p_signos": float((np.abs(nulo) >= abs(D.mean()) - 1e-12).mean()), "n": int(len(D)),
            "por_bandada": [float(x) for x in D]}


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(57)
    arch = w.archivos()
    fl = sorted({f for f, _, _ in arch})
    with ProcessPoolExecutor(max_workers=8) as ex:
        ids = dict(zip(fl, ex.map(lider, fl, [arch] * len(fl))))
    L = ["=== E57 · palomas con líder desinformado (Watts et al. 2016) ==="]
    for f in fl:
        L.append(f"bandada {f}: líder {ids[f]['lider']} ({'claro' if ids[f]['claro'] else 'decidido por DCD'}), "
                 f"segundo {ids[f]['segundo']}; rangos {ids[f]['rango']}; DCD {ids[f]['dcd']}")
    rels = list(DESPLAZ) + list(CONTROL) + list(PREVIAS)
    out = {"lideres": ids}
    for tramo, dmax in (("vuelo", None), ("2km", 2000.0)):
        jobs = [(f, r, [ids[f]["lider"], ids[f]["segundo"]], dmax) for f in fl for r in rels]
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = [x for x in ex.map(por_suelta, jobs) if x]
        idx = {(x["f"], x["r"]): x for x in res}

        def D(f, b, v, ctl):
            a = [idx[(f, r)]["P"].get(f"{b}|{v:g}", np.nan) for r in DESPLAZ if (f, r) in idx]
            c = [idx[(f, r)]["P"].get(f"{b}|{v:g}", np.nan) for r in ctl if (f, r) in idx]
            a, c = np.nanmean(a) if a else np.nan, np.nanmean(c) if c else np.nan
            return a - c, a, c

        def Doff(f, b, ctl):
            a = [DESPLAZ[r] * idx[(f, r)]["off"].get(b, np.nan) for r in DESPLAZ if (f, r) in idx]
            # en los controles el signo esperado no existe: se usa el mismo signo de cada suelta
            # desplazada sobre la media de los controles (diferencia de sesgo en esa dirección)
            c = [idx[(f, r)]["off"].get(b, np.nan) for r in ctl if (f, r) in idx]
            cm = np.nanmean(c) if c else np.nan
            a2 = [DESPLAZ[r] * (idx[(f, r)]["off"].get(b, np.nan) - cm) for r in DESPLAZ if (f, r) in idx]
            return float(np.nanmean(a2)) if a2 else np.nan

        T = {}
        for quien in ("lider", "segundo"):
            for ctl_nom, ctl in (("posteriores", CONTROL), ("previas", PREVIAS)):
                for v in VENT:
                    Ds = [D(f, ids[f][quien], v, ctl) for f in fl]
                    T[f"{quien}|{ctl_nom}|{v:g}"] = {**resumen([d[0] for d in Ds], rng),
                                                     "P_desplazada": float(np.nanmean([d[1] for d in Ds])),
                                                     "P_control": float(np.nanmean([d[2] for d in Ds]))}
                G = [D(f, ids[f][quien], 1.0, ctl)[0] - D(f, ids[f][quien], 0.2, ctl)[0] for f in fl]
                T[f"{quien}|{ctl_nom}|G"] = resumen(G, rng)
                T[f"{quien}|{ctl_nom}|rumbo"] = resumen([np.rad2deg(Doff(f, ids[f][quien], ctl)) for f in fl], rng)
        claros = [f for f in fl if ids[f]["claro"]]
        for v in (0.2, 1.0, 4.0):
            T[f"lider|posteriores|{v:g}|claros"] = resumen([D(f, ids[f]["lider"], v, CONTROL)[0] for f in claros], rng)
        out[tramo] = {"tabla": T, "sueltas": {f"{x['f']}|{x['r']}": {"s_juntas": x["s_juntas"]} for x in res}}
        L.append(f"\n=== tramo: {tramo} ===")
        for quien in ("lider", "segundo"):
            for ctl_nom in ("posteriores", "previas"):
                fila = " | ".join(f"{v:g} s {T[f'{quien}|{ctl_nom}|{v:g}']['media']:+.3f} "
                                  f"[{T[f'{quien}|{ctl_nom}|{v:g}'].get('ic95', [np.nan]*2)[0]:+.3f}; "
                                  f"{T[f'{quien}|{ctl_nom}|{v:g}'].get('ic95', [np.nan]*2)[1]:+.3f}]" for v in VENT)
                g = T[f"{quien}|{ctl_nom}|G"]; ru = T[f"{quien}|{ctl_nom}|rumbo"]
                L.append(f"{quien:7s} control {ctl_nom:11s} D: {fila}")
                L.append(f"{'':7s} {'':19s} P desplazada / control a 1 s: {T[f'{quien}|{ctl_nom}|1']['P_desplazada']:.3f} / "
                         f"{T[f'{quien}|{ctl_nom}|1']['P_control']:.3f}; G = {g['media']:+.3f} {g.get('ic95')} p {g.get('p_signos')}; "
                         f"rumbo {ru['media']:+.1f}° {ru.get('ic95')} p {ru.get('p_signos')}")
        for v in (0.2, 1.0, 4.0):
            q = T[f"lider|posteriores|{v:g}|claros"]
            L.append(f"sólo bandadas claras {claros}, {v:g} s: {q['media']:+.3f} {q.get('ic95')}")
    t = out["vuelo"]["tabla"]
    pr = {"Q1": {"D_1s": t["lider|posteriores|1"], "cumple": bool(t["lider|posteriores|1"].get("ic95", [0])[0] > 0)},
          "Q2": {"G": t["lider|posteriores|G"], "cumple": bool(t["lider|posteriores|G"]["media"] > 0)},
          "Q3": {"segundo_1s": t["segundo|posteriores|1"],
                 "cumple": bool(t["segundo|posteriores|1"].get("ic95", [0, 0])[0] <= 0 <= t["segundo|posteriores|1"].get("ic95", [0, 0])[1])},
          "Q4": {"rumbo": t["lider|posteriores|rumbo"], "cumple": bool(t["lider|posteriores|rumbo"].get("ic95", [0])[0] > 0)},
          "Q5": {"D_1s_2km": out["2km"]["tabla"]["lider|posteriores|1"]["media"], "D_1s_vuelo": t["lider|posteriores|1"]["media"],
                 "cumple": bool(out["2km"]["tabla"]["lider|posteriores|1"]["media"] > t["lider|posteriores|1"]["media"])}}
    out["pred"] = pr
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}" for k, v in pr.items()]
    json.dump(out, open("resultados/E57_palomas.json", "w"), indent=1, ensure_ascii=False, default=float)
    open("resultados/E57_palomas.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
