"""Etapa 54b (B4 del plan 13): control prospectivo en tetras.

Ventanas que terminan antes de la primera inversión individual (t1 - delta) en lugar de en el
giro colectivo. Diseño y predicción en docs/bitacora/resultados_E46-E55.md (antes de correr).

    PYTHONPATH=codigo python codigo/e54b_prospectivo.py
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e27_tetras as e27
from e54_peces import _dphi, detectar, suav_nan, auc

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT, K, Q = e27.DT, 3, 0.25
TAUS = ("cuadro", 0.25, 0.5, 1.0, 2.0)
DELTAS = (0.0, 0.25, 0.5)
N_BOOT = 2000


def experimento(fn):
    X, Y = e27.cargar(fn)
    Vx, Vy, om = e27.cinematica(X, Y, 1.0)
    dphi = _dphi(X, Y)
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    P = np.stack([X, Y], -1)
    for s in range(T):
        p, o = P[s], om[s]
        ok = np.isfinite(p).all(1) & np.isfinite(o)
        if ok.sum() < K + 2:
            continue
        ii = np.flatnonzero(ok)
        D = np.linalg.norm(p[ii][:, None] - p[ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        eps[s, ii] = o[ii] - o[ii][np.argsort(D, 1)[:, :K]].mean(1)
    aeps = np.abs(eps)
    eventos = detectar(dphi, 0.3, 10.0)
    if not eventos:
        return None
    lejos = np.ones(T, bool)
    for s in eventos:
        lejos[max(0, s - int(5 / DT)):s + int(5 / DT)] = False
    fb = np.isfinite(eps) & lejos[:, None]
    sig = float(np.nanmedian(np.abs(eps[fb] - np.nanmedian(eps[fb]))) * 1.4826)
    dphis = e27.suav(dphi, int(0.5 / DT))
    out = []
    for s0 in eventos:
        vent = slice(max(0, s0 - int(3 / DT)), min(T, s0 + int(3 / DT)))
        t_rev = np.full(N, np.nan)
        for a in range(N):
            tr = dphis[vent, a]
            ok = np.isfinite(tr)
            if ok.sum() < 20:
                continue
            cz = np.flatnonzero(np.diff(np.sign(tr[ok])) != 0)
            if len(cz):
                t_rev[a] = (vent.start + np.flatnonzero(ok)[cz[0]]) * DT
        okr = np.isfinite(t_rev)
        if okr.sum() < max(4, int(N * 0.6)):
            continue
        a0 = int(np.argsort(t_rev)[0])
        d_rel = t_rev - np.nanmin(t_rev)
        li, ls = np.nanquantile(d_rel[okr], Q), np.nanquantile(d_rel[okr], 1 - Q)
        ini = [a for a in range(N) if okr[a] and d_rel[a] <= li]
        seg = [a for a in range(N) if okr[a] and d_rel[a] >= ls]
        s1 = int(round(t_rev[a0] / DT))
        fila = {"ini": ini, "seg": seg, "a0": a0, "lead_s": float((s0 - s1) * DT), "st": {}}
        fines = {"colectivo": s0, **{f"d{d:g}": s1 - int(round(d / DT)) for d in DELTAS}}
        for nom, fin in fines.items():
            for tau in TAUS:
                n = 1 if tau == "cuadro" else int(tau / DT)
                if fin - n < 0:
                    continue
                with np.errstate(all="ignore"):
                    fila["st"][f"{nom}|{tau}"] = (np.nanmean(aeps[fin - n:fin], axis=0) / sig).tolist()
        out.append(fila)
    return out


def curva(exps, idx=None, sin_a0=False):
    idx = range(len(exps)) if idx is None else idx
    res = {}
    for clave in exps[0][0]["st"] if exps and exps[0] else []:
        x1, x0 = [], []
        for i in idx:
            for ev in exps[i]:
                if clave not in ev["st"]:
                    continue
                v = np.array(ev["st"][clave])
                ini = [a for a in ev["ini"] if not (sin_a0 and a == ev["a0"])]
                x1 += v[ini].tolist(); x0 += v[ev["seg"]].tolist()
        res[clave] = auc(x1, x0)
    return res


if __name__ == "__main__":
    os.chdir(RAIZ)
    fns = [fn for g in ("8", "10") for fn in sorted(glob.glob(os.path.join(e27.DATA, g, "*.csv")))]
    with ProcessPoolExecutor(max_workers=8) as ex:
        exps = [x for x in ex.map(experimento, fns) if x]
    rng = np.random.default_rng(541)
    out = {"n_exp": len(exps), "n_eventos": sum(len(x) for x in exps),
           "lead_mediana_s": float(np.median([e["lead_s"] for x in exps for e in x]))}
    B = rng.integers(0, len(exps), (N_BOOT, len(exps)))
    L = [f"=== E54b · tetras, ventanas que terminan antes de la primera inversión ===",
         f"{out['n_exp']} experimentos, {out['n_eventos']} giros; la primera inversión precede al giro colectivo en "
         f"{out['lead_mediana_s']:.2f} s (mediana)"]
    e54 = json.load(open("resultados/E54_peces.json"))["N8_10"]["curva"]
    for sin_a0 in (False, True):
        c = curva(exps, sin_a0=sin_a0)
        cb = [curva(exps, b, sin_a0) for b in B]
        clave_sal = "sin_primer_pez" if sin_a0 else "con_primer_pez"
        out[clave_sal] = {}
        L.append(f"\n--- {'sin' if sin_a0 else 'con'} el primer pez en el grupo de iniciadores ---")
        for fin in ("colectivo", "d0", "d0.25", "d0.5"):
            fila = {}
            for tau in TAUS:
                k = f"{fin}|{tau}"
                if k not in c:
                    continue
                v = [x[k] for x in cb]
                fila[str(tau)] = {"auc": c[k], "ic95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
            d = [x[f"{fin}|1.0"] - x[f"{fin}|cuadro"] for x in cb]
            fila["ganancia_1s"] = {"valor": c[f"{fin}|1.0"] - c[f"{fin}|cuadro"],
                                   "ic95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
            out[clave_sal][fin] = fila
            L.append(f"termina en {fin:9s}: " + " | ".join(
                f"{t}: {fila[str(t)]['auc']:.3f} [{fila[str(t)]['ic95'][0]:.3f}; {fila[str(t)]['ic95'][1]:.3f}]"
                for t in TAUS if str(t) in fila)
                + f" | ganancia 1 s − cuadro {fila['ganancia_1s']['valor']:+.3f} "
                  f"[{fila['ganancia_1s']['ic95'][0]:+.3f}; {fila['ganancia_1s']['ic95'][1]:+.3f}]")
    dm = max(abs(out["con_primer_pez"]["colectivo"][str(t)]["auc"] - e54[str(t)]) for t in ("cuadro", 0.5, 1.0, 2.0))
    out["ancla_E54"] = {"delta_max": dm, "cumple": bool(dm < 1e-9)}
    L.append(f"\nancla: ventanas que terminan en el giro colectivo frente a E54: |Δ| máx = {dm:.2e}")
    json.dump(out, open("resultados/E54b_prospectivo.json", "w"), indent=1, ensure_ascii=False)
    open("resultados/E54b_prospectivo.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
