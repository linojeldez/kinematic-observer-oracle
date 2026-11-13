"""Etapa 73b: variantes de E73 (ciclo de trabajo del observador del modelo, k = 5, rapidez relativa).

Diseño y predicción V1 en docs/bitacora/resultados_E66-E75.md («E73b»).

    PYTHONPATH=codigo python codigo/e73b_variantes.py
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e73_peces_entrenados as e73
from carga_segura import cargar

C_DET = 1.5
F_JSON, F_TXT = "resultados/E73b_variantes.json", "resultados/E73b_variantes.txt"


def residuo_k(X, Y, om, k):
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    P = np.stack([X, Y], -1)
    for s in range(T):
        ii = np.flatnonzero(np.isfinite(P[s]).all(1) & np.isfinite(om[s]))
        if len(ii) < k + 2:
            continue
        D = np.linalg.norm(P[s][ii][:, None] - P[s][ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        eps[s, ii] = om[s, ii] - om[s, ii][np.argsort(D, 1)[:, :k]].mean(1)
    return eps


def ventanas(x, kf, FPS, tD):
    st = {}
    fines = [(f"{tf:g}", kf + int(round(tf * FPS))) for tf in e73.TFIN]
    if np.isfinite(tD):
        fines.append(("tD", kf + int(round(tD * FPS))))
    for w in e73.VENT:
        n = 1 if w == "cuadro" else int(round(w * FPS))
        for nom, e in fines:
            with np.errstate(all="ignore"):
                st[f"{w}|{nom}"] = np.nanmean(x[e - n:e], axis=0).tolist()
    return st


def ensayo(fn):
    d = cargar(fn)
    tr = np.asarray(d["trajectories"], float)
    FPS = int(d["frames_per_second"]); DT = 1.0 / FPS
    kf = int(d["key_frame"])
    Xs, Ys, Vx, Vy, om = e73.cinematica(tr[:, :, 0], tr[:, :, 1], DT)
    N = tr.shape[1]
    ent = np.zeros(N, bool); ent[[int(i) for i in d["identities"]["trained"]]] = True
    com = np.asarray(d["setup_points"][f"{d['light_side']}_feeder"], float).ravel()
    tD = e73.t_respuesta(Xs, Ys, Vx, Vy, ~ent, com, kf, FPS)
    nb = int(e73.T_BASE * FPS)
    out = {"grupo": d["experiment"], "entrenados": ent.tolist(), "tD": tD, "var": {}}
    for k in (3, 5):
        eps = residuo_k(Xs, Ys, om, k)
        b = eps[kf - nb:kf]; b = b[np.isfinite(b)]
        sig = float(np.median(np.abs(b - np.median(b))) * 1.4826)
        if k == 3:
            out["var"]["ciclo"] = ventanas((np.abs(eps) > C_DET * sig).astype(float) * np.where(np.isfinite(eps), 1, np.nan), kf, FPS, tD)
        else:
            out["var"]["k5"] = ventanas(np.abs(eps) / sig, kf, FPS, tD)
    v = np.hypot(Vx, Vy)
    P = np.stack([Xs, Ys], -1)
    rel = np.full_like(v, np.nan)
    for s in range(len(v)):
        D = np.linalg.norm(P[s][:, None] - P[s][None, :], axis=2); np.fill_diagonal(D, np.inf)
        rel[s] = v[s] - v[s][np.argsort(D, 1)[:, :3]].mean(1)
    b = rel[kf - nb:kf]
    out["var"]["rapidez"] = ventanas(rel / (np.median(np.abs(b - np.median(b))) * 1.4826), kf, FPS, tD)
    return out


if __name__ == "__main__":
    os.chdir(e73.RAIZ)
    fns = sorted(glob.glob(os.path.join(e73.DATOS, "stimstart", "*.npy")))
    with ProcessPoolExecutor(max_workers=3) as ex:
        S = list(ex.map(ensayo, fns))
    out, L = {}, ["=== E73b · variantes (159 ensayos, 6 grupos; IC por grupos y ensayos) ==="]
    for var in ("ciclo", "k5", "rapidez"):
        ens = [{"grupo": e["grupo"], "entrenados": e["entrenados"], "st": e["var"][var]} for e in S]
        claves = [f"{w}|{t}" for w in ("cuadro", 1.0) for t in ("0", "0.5", "1", "2")]
        v, B = e73.con_ic(ens, claves, np.random.default_rng(731))
        con_tD = [e for e in ens if "1.0|tD" in e["st"]]
        vt, Bt = e73.con_ic(con_tD, ["cuadro|tD", "1.0|tD"], np.random.default_rng(732))
        q = {"auc": v, "ic": {c: e73.ic(B[c]) for c in claves}, "auc_tD": vt,
             "dif_1": {"dif": v["1.0|1"] - v["cuadro|1"], "ic95": e73.ic(B["1.0|1"] - B["cuadro|1"])},
             "dif_tD": {"dif": vt["1.0|tD"] - vt["cuadro|tD"], "ic95": e73.ic(Bt["1.0|tD"] - Bt["cuadro|tD"])}}
        out[var] = q
        L.append(f"\n--- {var} ---")
        L.append("AUC 1 cuadro / 1 s: " + "  ".join(f"{t}: {v[f'cuadro|{t}']:.3f} / {v[f'1.0|{t}']:.3f}" for t in ("0", "0.5", "1", "2"))
                 + f"  | en t_D: {vt['cuadro|tD']:.3f} / {vt['1.0|tD']:.3f}")
        L.append(f"1 s − 1 cuadro a +1 s: {q['dif_1']['dif']:+.3f} [{q['dif_1']['ic95'][0]:+.3f}; {q['dif_1']['ic95'][1]:+.3f}]; "
                 f"en t_D: {q['dif_tD']['dif']:+.3f} [{q['dif_tD']['ic95'][0]:+.3f}; {q['dif_tD']['ic95'][1]:+.3f}]")
    out["V1"] = {"cumple": bool(all(out[v]["dif_1"]["ic95"][0] > 0 and out[v]["dif_tD"]["ic95"][0] > 0 for v in ("ciclo", "k5")))}
    L.append(f"\nV1: {'CUMPLE' if out['V1']['cumple'] else 'FALLA'}")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
