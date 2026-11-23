"""Etapa 74: peces cebra con filtros de sólo pasado (objeciones 1, 2, 6 y 9 de la auditoría del 14-nov).

Rehace E73 con una cadena en que ningún valor usa cuadros posteriores a su instante (principal:
los filtros de E73 hacia atrás, 11 cuadros por valor; sensibilidad: sin suavizar, 3 cuadros), mete
la elección de ventana y de cierre dentro de la inferencia (se elige en un remuestreo de grupos y se
evalúa en los grupos que quedan fuera) y mide el AUC antes de que responda cualquier no entrenado.

Diseño y predicciones S1-S8 en docs/bitacora/resultados_E66-E75.md («E74»).

    PYTHONPATH=codigo python codigo/e74_peces_solo_pasado.py
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata, spearmanr

import e73_peces_entrenados as e73
import e73b_variantes as e73b
from carga_segura import cargar

SAL = os.environ.get("E74_SALIDA", "resultados")
F_JSON, F_TXT = os.path.join(SAL, "E74_peces_solo_pasado.json"), os.path.join(SAL, "E74_peces_solo_pasado.txt")
W = ("valor", 0.25, 0.5, 1.0, 2.0)
TG = np.round(np.arange(-1.0, 3.0001, 1 / 16), 4)          # cierres, s desde el encendido
MARG = (0.0, 0.1, 0.2)                                    # margen antes de la respuesta, s
LAGS = np.round(np.arange(-0.5, 1.0001, 1 / 16), 4)        # cierres respecto de t_E
C_DET = 1.5
N_BOOT = 2000
CAD = ("sp", "crudo", "cen", "vel", "ciclo", "k5")
T_SEL = (1 / 16, 1.5)                                     # cierres admitidos al elegir
T_SIN = 0.6                                               # «sin respuesta individual antes de»
P = ("0.25", "0.5")                                       # pareja principal: ventana 0,25 s, cierre +0,5 s


def atras(x, w):
    """Media de los w cuadros que terminan en t (NaN en los w-1 primeros)."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def cinematica_sp(X, Y, DT):
    """La cadena de E73 con cada filtro hacia atrás: un valor de omega usa los cuadros t-10...t."""
    n = e73.N_SUAV
    Xs, Ys = atras(X, n), atras(Y, n)
    Vx, Vy = np.full_like(Xs, np.nan), np.full_like(Ys, np.nan)
    Vx[1:], Vy[1:] = np.diff(Xs, axis=0) / DT, np.diff(Ys, axis=0) / DT
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    om[n + 1:] = np.diff(np.unwrap(th[n:], axis=0), axis=0) / DT
    oms = np.full_like(om, np.nan)
    oms[n + 1:] = atras(om[n + 1:], n)
    return Xs, Ys, Vx, Vy, oms


def cinematica_cruda(X, Y, DT):
    """Sin suavizar: dos diferencias hacia atrás; un valor de omega usa los cuadros t-2...t."""
    Vx, Vy = np.full_like(X, np.nan), np.full_like(Y, np.nan)
    Vx[1:], Vy[1:] = np.diff(X, axis=0) / DT, np.diff(Y, axis=0) / DT
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    om[2:] = np.diff(np.unwrap(th[1:], axis=0), axis=0) / DT
    return X, Y, Vx, Vy, om


def ancho(x, kf, nb):
    b = x[kf - nb:kf]
    b = b[np.isfinite(b)]
    return float(np.median(np.abs(b - np.median(b))) * 1.4826)


def cierres(x, kf, FPS, tiempos):
    """Media de x sobre cada ventana de W que termina en el cuadro anterior a cada cierre: (nW, nT, N)."""
    out = np.full((len(W), len(tiempos), x.shape[1]), np.nan)
    for i, w in enumerate(W):
        n = 1 if w == "valor" else int(round(w * FPS))
        for j, t in enumerate(tiempos):
            if np.isfinite(t):
                e = kf + int(round(t * FPS))
                out[i, j] = x[e - n:e].mean(axis=0)
    return out


def ensayo(fn):
    d = cargar(fn)
    tr = np.asarray(d["trajectories"], float)
    assert np.isfinite(tr).all()
    FPS = int(d["frames_per_second"]); DT = 1.0 / FPS
    kf = int(d["key_frame"]); nb = int(e73.T_BASE * FPS)
    X, Y = tr[:, :, 0], tr[:, :, 1]
    N = tr.shape[1]
    ent = np.zeros(N, bool); ent[[int(i) for i in d["identities"]["trained"]]] = True
    Xc, Yc, Vxc, Vyc, omc = e73.cinematica(X, Y, DT)
    Xs, Ys, Vxs, Vys, oms = cinematica_sp(X, Y, DT)
    _, _, Vxr, Vyr, omr = cinematica_cruda(X, Y, DT)
    x, sig = {}, {}
    for c, (px, py, om) in (("cen", (Xc, Yc, omc)), ("sp", (Xs, Ys, oms)), ("crudo", (X, Y, omr))):
        eps = e73.residuo(px, py, om)
        sig[c] = ancho(eps, kf, nb)
        x[c] = np.abs(eps) / sig[c]
        if c == "sp":
            x["ciclo"] = (np.abs(eps) > C_DET * sig[c]).astype(float)
    eps5 = e73b.residuo_k(Xs, Ys, oms, 5)
    x["k5"] = np.abs(eps5) / ancho(eps5, kf, nb)
    v = np.hypot(Vxs, Vys)
    rel = np.full_like(v, np.nan)
    Pp = np.stack([Xs, Ys], -1)
    for s in range(e73.N_SUAV, len(v)):
        D = np.linalg.norm(Pp[s][:, None] - Pp[s][None, :], axis=2); np.fill_diagonal(D, np.inf)
        rel[s] = v[s] - v[s][np.argsort(D, 1)[:, :e73.K]].mean(1)
    x["vel"] = rel / ancho(rel, kf, nb)
    out = {"grupo": d["video_code"], "lote": d["experiment"], "ent": ent, "sigma": sig,
           "quieto_crudo": float(((Vxr[1:] == 0) & (Vyr[1:] == 0)).mean()),
           "st": {c: cierres(x[c], kf, FPS, TG) for c in CAD}}
    if os.path.basename(fn).endswith("_stimstart.npy"):
        com = np.asarray(d["setup_points"][f"{d['light_side']}_feeder"], float).ravel()
        tE = e73.t_respuesta(Xc, Yc, Vxc, Vyc, ent, com, kf, FPS)
        tD = e73.t_respuesta(Xc, Yc, Vxc, Vyc, ~ent, com, kf, FPS)
        ind = []
        for j in np.flatnonzero(~ent):
            q = np.zeros(N, bool); q[j] = True
            ind.append(e73.t_respuesta(Xc, Yc, Vxc, Vyc, q, com, kf, FPS))
        ind = np.array(ind)
        t1 = float(np.nanmin(ind)) if np.isfinite(ind).any() else np.nan
        out.update({"tE": tE, "tD": tD, "t1": t1, "n_ind": int(np.isfinite(ind).sum()),
                    "n_antes": int(np.sum(ind < tD)) if np.isfinite(tD) else -1,
                    "en_tD": {c: cierres(x[c], kf, FPS, [tD - m for m in MARG]) for c in ("sp", "crudo", "cen", "vel")},
                    "en_t1": {c: cierres(x[c], kf, FPS, [t1 - m if t1 > 0 else np.nan for m in MARG])
                              for c in ("sp", "crudo", "cen", "vel")},
                    "en_tE": {c: cierres(x[c], kf, FPS, [tE + l for l in LAGS]) for c in ("sp", "cen")}})
    return out


def conjunto(sub):
    fns = sorted(glob.glob(os.path.join(e73.DATOS, sub, "*.npy")))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("FASE4_W", "4"))) as ex:
        return list(ex.map(ensayo, fns))


# ---------------------------------------------------------------- AUC agrupado, vectorizado sobre claves
def auc_filas(Z, m):
    """Z: (claves, ensayos, peces); m: (ensayos, peces) entrenado. AUC agrupado de cada clave."""
    z1, z0 = Z[:, m], Z[:, ~m]
    n1, n0 = z1.shape[1], z0.shape[1]
    if n1 == 0 or n0 == 0:
        return np.full(len(Z), np.nan)
    r = rankdata(np.concatenate([z1, z0], axis=1), axis=1)
    return (r[:, :n1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def indices(grupos, sel, rng, n=N_BOOT, dos_niveles=True):
    """Remuestreos de grupos y, dentro de cada grupo elegido, de ensayos (como E73), restringidos a sel."""
    g = {}
    for i in np.flatnonzero(sel):
        g.setdefault(grupos[i], []).append(i)
    gs = sorted(g)
    out = []
    for _ in range(n):
        idx = []
        for k in rng.choice(len(gs), len(gs)):
            lst = np.asarray(g[gs[k]])
            idx.append(lst[rng.integers(0, len(lst), len(lst))] if dos_niveles else lst)
        out.append(np.concatenate(idx))
    return out


def con_ic(Z, M, idxs):
    """AUC de cada clave (filas de Z) y su distribución por remuestreo, con los mismos remuestreos."""
    return np.array([auc_filas(Z[:, i], M[i]) for i in idxs])


def ic(b, axis=0):
    return np.nanpercentile(b, [2.5, 97.5], axis=axis)


def par(v, b):
    return {"auc": float(v), "ic95": [float(q) for q in ic(b)]}


def fuera_de_bolsa(Z, M, grupos, rng, tsel, n=N_BOOT):
    """Z: (nW, nT, ensayos, peces). En cada remuestreo de grupos elige la clave de mayor AUC (global, por
    ventana y entre 0,25 y 0,5 s) y la evalúa en los grupos que no entraron."""
    gs = sorted(set(grupos))
    de = {g: np.flatnonzero(grupos == g) for g in gs}
    nW, nT = Z.shape[:2]
    Zs = Z[:, tsel]
    nS = Zs.shape[1]
    plano = Zs.reshape(nW * nS, *Zs.shape[2:])
    corto = [i for i, w in enumerate(W) if w in (0.25, 0.5)]
    res = {"global": [], "w_global": [], "t_global": [], "por_w": [], "t_por_w": [], "corta": []}
    for _ in range(n):
        ks = rng.choice(len(gs), len(gs))
        dentro = np.concatenate([de[gs[k]] for k in ks])
        sk = set(ks.tolist())
        fuera = np.concatenate([de[g] for i, g in enumerate(gs) if i not in sk] or [np.array([], int)])
        if len(fuera) == 0:
            continue
        a = auc_filas(plano[:, dentro], M[dentro]).reshape(nW, nS)
        af = auc_filas(plano[:, fuera], M[fuera]).reshape(nW, nS)
        i, j = np.unravel_index(np.argmax(a), a.shape)
        res["global"].append(af[i, j]); res["w_global"].append(i); res["t_global"].append(j)
        jj = a.argmax(1)
        res["por_w"].append(af[np.arange(nW), jj]); res["t_por_w"].append(jj)
        ic_ = max(corto, key=lambda q: a[q, jj[q]])
        res["corta"].append(af[ic_, jj[ic_]])
    return {k: np.array(v) for k, v in res.items()}


if __name__ == "__main__":
    os.chdir(e73.RAIZ)
    S, I = conjunto("stimstart"), conjunto("intertrial")
    grupos = np.array([e["grupo"] for e in S]); M = np.array([e["ent"] for e in S])
    gI = np.array([e["grupo"] for e in I]); MI = np.array([e["ent"] for e in I])
    Z = {c: np.stack([e["st"][c] for e in S], 2) for c in CAD}          # (nW, nT, ensayos, peces)
    ZI = {c: np.stack([e["st"][c] for e in I], 2) for c in CAD}
    nW, nT = len(W), len(TG)
    iw = {str(w): i for i, w in enumerate(W)}
    it = {f"{t:g}": j for j, t in enumerate(TG)}
    todos = np.ones(len(S), bool)
    idxs = indices(grupos, todos, np.random.default_rng(741))
    tE = np.array([e["tE"] for e in S]); tD = np.array([e["tD"] for e in S]); t1 = np.array([e["t1"] for e in S])
    out = {"W": [str(w) for w in W], "TG": TG.tolist(), "LAGS": LAGS.tolist(), "MARG": list(MARG)}
    L = ["=== E74 · peces cebra con filtros de sólo pasado (Romero-Ferrero et al. 2023, sesión 11) ==="]
    out["n"] = {"ensayos": len(S), "grupos": len(set(grupos)), "entre_ensayos": len(I),
                "soporte": {"sp": 2 * e73.N_SUAV + 1, "crudo": 3, "cen": 2 * (e73.N_SUAV - 1) + 5},
                "sigma_mediana": {c: float(np.median([e["sigma"][c] for e in S])) for c in ("cen", "sp", "crudo")},
                "quieto_crudo": float(np.mean([e["quieto_crudo"] for e in S]))}
    L.append(f"ensayos {len(S)} en {out['n']['grupos']} grupos; entre ensayos {len(I)}; cuadros por valor: sólo pasado "
             f"{out['n']['soporte']['sp']}, sin suavizar 3, centrado (E73) {out['n']['soporte']['cen']}; sigma mediana (rad/s): "
             + ", ".join(f"{c} {v:.2f}" for c, v in out["n"]["sigma_mediana"].items())
             + f"; pasos crudos sin desplazamiento {100 * out['n']['quieto_crudo']:.2f} %")

    # ------------------------------------------------------------ 1. curso temporal con IC
    out["curso"], B = {}, {}
    for c in CAD:
        plano = Z[c].reshape(nW * nT, *Z[c].shape[2:])
        v = auc_filas(plano, M).reshape(nW, nT)
        out["curso"][c] = {"auc": v.tolist()}
        if c in ("sp", "crudo", "vel"):
            B[c] = con_ic(plano, M, idxs).reshape(len(idxs), nW, nT)
            q = ic(B[c])
            out["curso"][c].update({"lo": q[0].tolist(), "hi": q[1].tolist()})
    for c, nom in (("sp", "sólo pasado"), ("crudo", "sin suavizar"), ("cen", "centrado (E73)")):
        L.append(f"\nAUC entrenado frente a no entrenado, {nom}; filas: cierre (s); columnas: " + "  ".join(str(w) for w in W))
        for j, t in enumerate(TG):
            if abs(t * 8 - round(t * 8)) < 1e-6 or 0 < t <= 1:
                a = np.array(out["curso"][c]["auc"])[:, j]
                L.append(f"{t:+.4f}  " + "  ".join(f"{x:.3f}" for x in a)
                         + ("   IC " + "  ".join(f"[{np.array(out['curso'][c]['lo'])[i, j]:.2f};{np.array(out['curso'][c]['hi'])[i, j]:.2f}]"
                                                 for i in range(nW)) if c in B and (abs(t * 4 - round(t * 4)) < 1e-6) else ""))

    def punto(c, w, t):
        i, j = iw[str(w)], it[f"{t:g}"]
        return par(out["curso"][c]["auc"][i][j], B[c][:, i, j])

    def dif(c1, w1, t1_, c2, w2, t2_):
        a = B[c1][:, iw[str(w1)], it[f"{t1_:g}"]] - B[c2][:, iw[str(w2)], it[f"{t2_:g}"]]
        v = out["curso"][c1]["auc"][iw[str(w1)]][it[f"{t1_:g}"]] - out["curso"][c2]["auc"][iw[str(w2)]][it[f"{t2_:g}"]]
        return {"dif": float(v), "ic95": [float(q) for q in ic(a)]}

    out["puntos"] = {c: {f"{w}|{t:g}": punto(c, w, t) for w in W for t in (-1, 0, 0.25, 0.375, 0.5, 0.625, 0.75, 1, 1.5, 2, 3)}
                     for c in ("sp", "crudo", "vel")}
    out["difs"] = {"sp_0.25@0.5_menos_1@1": dif("sp", 0.25, 0.5, "sp", 1.0, 1),
                   "sp_0.25@0.5_menos_1@0.5": dif("sp", 0.25, 0.5, "sp", 1.0, 0.5),
                   "sp_0.25@0.5_menos_valor@0.375": dif("sp", 0.25, 0.5, "sp", "valor", 0.375),
                   "sp_1@1_menos_valor@1": dif("sp", 1.0, 1, "sp", "valor", 1),
                   "vel_menos_giro_0.25@0.5": dif("vel", 0.25, 0.5, "sp", 0.25, 0.5)}
    L.append("\nPuntos (sólo pasado): " + "; ".join(f"{k} {q['auc']:.3f} [{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}]" for k, q in
                                                   out["puntos"]["sp"].items() if k in ("valor|0.375", "0.25|0.5", "0.5|0.5", "0.5|0.625", "1.0|0.5", "1.0|1", "valor|1", "0.25|1")))
    L.append("Diferencias pareadas: " + "; ".join(f"{k} {q['dif']:+.3f} [{q['ic95'][0]:+.3f}; {q['ic95'][1]:+.3f}]" for k, q in out["difs"].items()))

    # ------------------------------------------------------------ 2. elección dentro de la inferencia
    tsel = np.flatnonzero((TG >= T_SEL[0] - 1e-9) & (TG <= T_SEL[1] + 1e-9))
    out["fuera"] = {}
    for c, sem in (("sp", 742), ("crudo", 743)):
        r = fuera_de_bolsa(Z[c], M, grupos, np.random.default_rng(sem), tsel)
        d_cl = r["corta"] - r["por_w"][:, iw["1.0"]]
        d_cv = r["por_w"][:, iw["0.25"]] - r["por_w"][:, iw["valor"]]
        d_cu = r["por_w"][:, iw["0.25"]] - r["por_w"][:, iw["1.0"]]
        q = {"n": int(len(r["global"])), "global": par(r["global"].mean(), r["global"]),
             "frac_ventana": {str(w): float((r["w_global"] == i).mean()) for i, w in enumerate(W)},
             "t_global": {"mediana": float(np.median(TG[tsel][r["t_global"]])),
                          "iqr": [float(x) for x in np.percentile(TG[tsel][r["t_global"]], [25, 75])]},
             "por_ventana": {str(w): {**par(r["por_w"][:, i].mean(), r["por_w"][:, i]),
                                      "t_mediana": float(np.median(TG[tsel][r["t_por_w"][:, i]]))} for i, w in enumerate(W)},
             "corta": par(r["corta"].mean(), r["corta"]),
             "corta_menos_1s": {"dif": float(d_cl.mean()), "ic95": [float(x) for x in ic(d_cl)]},
             "0.25_menos_valor": {"dif": float(d_cv.mean()), "ic95": [float(x) for x in ic(d_cv)]},
             "0.25_menos_1s": {"dif": float(d_cu.mean()), "ic95": [float(x) for x in ic(d_cu)]}}
        # dejando un grupo fuera
        gs = sorted(set(grupos)); plano = Z[c][:, tsel].reshape(nW * len(tsel), *Z[c].shape[2:])
        lo_ = []
        for g in gs:
            a = auc_filas(plano[:, grupos != g], M[grupos != g])
            k = int(np.argmax(a))
            lo_.append((k // len(tsel), k % len(tsel), float(auc_filas(plano[k:k + 1, grupos == g], M[grupos == g])[0]), int((grupos == g).sum())))
        q["un_grupo_fuera"] = {"media_pond": float(np.average([x[2] for x in lo_], weights=[x[3] for x in lo_])),
                               "sobre_0.5": int(sum(x[2] > 0.5 for x in lo_)),
                               "claves": sorted({f"{W[x[0]]}|{TG[tsel][x[1]]:g}" for x in lo_})}
        out["fuera"][c] = q
        L.append(f"\nElección dentro de la inferencia ({c}; {q['n']} remuestreos de grupos, cierres {T_SEL[0]:g}–{T_SEL[1]:g} s): AUC fuera de bolsa "
                 f"{q['global']['auc']:.3f} [{q['global']['ic95'][0]:.3f}; {q['global']['ic95'][1]:.3f}]; ventana elegida: "
                 + ", ".join(f"{w} {100 * f:.0f} %" for w, f in q["frac_ventana"].items())
                 + f"; cierre elegido mediana {q['t_global']['mediana']:g} s [{q['t_global']['iqr'][0]:g}; {q['t_global']['iqr'][1]:g}]")
        L.append("  por ventana (cierre elegido en la bolsa): " + "; ".join(
            f"{w}: {p['auc']:.3f} [{p['ic95'][0]:.3f}; {p['ic95'][1]:.3f}] (cierre {p['t_mediana']:g})" for w, p in q["por_ventana"].items()))
        L.append(f"  mejor de 0,25 y 0,5 s menos 1 s: {q['corta_menos_1s']['dif']:+.3f} [{q['corta_menos_1s']['ic95'][0]:+.3f}; {q['corta_menos_1s']['ic95'][1]:+.3f}]; "
                 f"0,25 s menos un valor: {q['0.25_menos_valor']['dif']:+.3f} [{q['0.25_menos_valor']['ic95'][0]:+.3f}; {q['0.25_menos_valor']['ic95'][1]:+.3f}]; "
                 f"0,25 s menos 1 s: {q['0.25_menos_1s']['dif']:+.3f} [{q['0.25_menos_1s']['ic95'][0]:+.3f}; {q['0.25_menos_1s']['ic95'][1]:+.3f}]")
        L.append(f"  dejando un grupo fuera: media ponderada {q['un_grupo_fuera']['media_pond']:.3f}; grupos sobre 0,5: {q['un_grupo_fuera']['sobre_0.5']}; "
                 f"claves elegidas {q['un_grupo_fuera']['claves']}")

    # ------------------------------------------------------------ 3. por grupo
    i0, j0 = iw[P[0]], it[P[1]]
    out["grupos"] = {}
    for c in ("sp", "crudo"):
        ag = {g: float(auc_filas(Z[c][i0:i0 + 1, j0][:, grupos == g], M[grupos == g])[0]) for g in sorted(set(grupos))}
        out["grupos"][c] = {"auc": ag, "sobre_0.5": int(sum(a > 0.5 for a in ag.values())), "min": min(ag.values()), "max": max(ag.values())}
        L.append(f"\nPor grupo ({c}, 0,25 s con cierre a +0,5 s): {out['grupos'][c]['sobre_0.5']} de {len(ag)} sobre 0,5; rango "
                 f"{out['grupos'][c]['min']:.2f}–{out['grupos'][c]['max']:.2f}")

    # robustez de la pareja principal: AUC dentro de cada ensayo (media) e IC remuestreando las tandas
    pl0 = Z["sp"][i0:i0 + 1, j0]
    dentro = np.array([auc_filas(pl0[:, k:k + 1], M[k:k + 1])[0] for k in range(len(S))])
    lotes = np.array([e["lote"] for e in S])
    bl = np.array([auc_filas(pl0[:, i], M[i])[0] for i in indices(lotes, todos, np.random.default_rng(748))])
    out["robustez"] = {"dentro_de_ensayo": par(dentro.mean(), np.array([dentro[i].mean() for i in idxs])),
                       "por_tanda": par(out["curso"]["sp"]["auc"][i0][j0], bl), "tandas": int(len(set(lotes)))}
    L.append(f"Robustez (sp, 0,25 s a +0,5 s): media del AUC dentro de cada ensayo {out['robustez']['dentro_de_ensayo']['auc']:.3f} "
             f"[{out['robustez']['dentro_de_ensayo']['ic95'][0]:.3f}; {out['robustez']['dentro_de_ensayo']['ic95'][1]:.3f}]; IC remuestreando las "
             f"{out['robustez']['tandas']} tandas [{out['robustez']['por_tanda']['ic95'][0]:.3f}; {out['robustez']['por_tanda']['ic95'][1]:.3f}]")

    # ------------------------------------------------------------ 4. controles
    out["control"] = {}
    for c in ("sp", "crudo", "vel", "cen"):
        a = auc_filas(ZI[c].reshape(nW * nT, *ZI[c].shape[2:]), MI).reshape(nW, nT)
        post = (TG > 0) & (TG <= 1.5)
        out["control"][c] = {"auc": a.tolist(), "max_desv_0_1.5": float(np.abs(a[:4][:, post] - 0.5).max()),
                             "pre_real_max_desv": float(np.abs(np.array(out["curso"][c]["auc"])[:, TG <= 0] - 0.5).max())}
    idxI = indices(gI, np.ones(len(I), bool), np.random.default_rng(744))
    kI = [(w, t) for w in ("valor", 0.25, 0.5, 1.0) for t in (0.25, 0.5, 1)]
    plI = np.stack([ZI["sp"][iw[str(w)], it[f"{t:g}"]] for w, t in kI])
    bI = con_ic(plI, MI, idxI)
    vI = auc_filas(plI, MI)
    out["control"]["sp"]["puntos"] = {f"{w}|{t:g}": par(vI[k], bI[:, k]) for k, (w, t) in enumerate(kI)}
    L.append("\nControl, encendidos ficticios entre ensayos (sólo pasado): " + "; ".join(
        f"{k} {q['auc']:.3f} [{q['ic95'][0]:.2f}; {q['ic95'][1]:.2f}]" for k, q in out["control"]["sp"]["puntos"].items()))
    L.append("  desviación máxima de 0,5 con cierres en (0; 1,5] s y ventanas hasta 1 s: " + ", ".join(
        f"{c} {out['control'][c]['max_desv_0_1.5']:.3f}" for c in ("sp", "crudo", "vel"))
             + "; en encendidos reales con cierre ≤ 0: " + ", ".join(f"{c} {out['control'][c]['pre_real_max_desv']:.3f}" for c in ("sp", "crudo", "vel")))

    # ------------------------------------------------------------ 5. antes de la respuesta
    out["resp"] = {"tE_mediana": float(np.nanmedian(tE)), "tD_mediana": float(np.nanmedian(tD)), "t1_mediana": float(np.nanmedian(t1)),
                   "n_tE": int(np.isfinite(tE).sum()), "n_tD": int(np.isfinite(tD).sum()), "n_t1": int(np.isfinite(t1).sum()),
                   "t1_iqr": [float(x) for x in np.nanpercentile(t1, [25, 75])],
                   "individual_antes_de_tD": int(sum(e["n_antes"] > 0 for e in S)),
                   "sin_individual_antes": int(((~np.isfinite(t1)) | (t1 >= T_SIN)).sum()),
                   "tD_menor_0.2": int((tD < 0.2).sum()), "t1_menor_0.2": int((t1 < 0.2).sum())}
    L.append(f"\nRespuesta: t_E mediana {out['resp']['tE_mediana']:.2f} s ({out['resp']['n_tE']} ensayos); t_D {out['resp']['tD_mediana']:.2f} s "
             f"({out['resp']['n_tD']}); primer no entrenado individual t_1 {out['resp']['t1_mediana']:.2f} s "
             f"[{out['resp']['t1_iqr'][0]:.2f}; {out['resp']['t1_iqr'][1]:.2f}] ({out['resp']['n_t1']}); algún individuo antes de t_D en "
             f"{out['resp']['individual_antes_de_tD']} de {out['resp']['n_tD']}; sin respuesta individual antes de +{T_SIN:g} s: "
             f"{out['resp']['sin_individual_antes']} ensayos")
    out["antes"] = {}
    for clave, tt, sem in (("en_tD", tD, 745), ("en_t1", t1, 746)):
        sel = np.isfinite(tt) & ((tt > 0) if clave == "en_t1" else True)
        ix = indices(grupos, sel, np.random.default_rng(sem))
        out["antes"][clave] = {"n": int(sel.sum())}
        for c in ("sp", "crudo", "cen", "vel"):
            Zc = np.full((nW, len(MARG), len(S), M.shape[1]), np.nan)
            for k, e in enumerate(S):
                if sel[k]:
                    Zc[:, :, k] = e[clave][c]
            plano = Zc.reshape(nW * len(MARG), len(S), -1)
            v = auc_filas(plano[:, sel], M[sel]).reshape(nW, len(MARG))
            b = np.array([auc_filas(plano[:, i], M[i]) for i in ix]).reshape(len(ix), nW, len(MARG))
            q = {}
            for jm, m in enumerate(MARG):
                q[f"{m:g}"] = {str(w): par(v[i, jm], b[:, i, jm]) for i, w in enumerate(W)}
                for nom, ia, ib in (("1s_menos_valor", iw["1.0"], iw["valor"]), ("0.25_menos_valor", iw["0.25"], iw["valor"]),
                                    ("1s_menos_0.25", iw["1.0"], iw["0.25"])):
                    q[f"{m:g}"][nom] = {"dif": float(v[ia, jm] - v[ib, jm]), "ic95": [float(x) for x in ic(b[:, ia, jm] - b[:, ib, jm])]}
            out["antes"][clave][c] = q
        for c in ("sp", "crudo"):
            for m in MARG:
                q = out["antes"][clave][c][f"{m:g}"]
                L.append(f"{clave} − {m:g} s ({c}, n = {out['antes'][clave]['n']}): " + "  ".join(
                    f"{w}: {q[str(w)]['auc']:.3f} [{q[str(w)]['ic95'][0]:.2f}; {q[str(w)]['ic95'][1]:.2f}]" for w in W)
                         + f"  | 1 s − valor {q['1s_menos_valor']['dif']:+.3f} [{q['1s_menos_valor']['ic95'][0]:+.3f}; {q['1s_menos_valor']['ic95'][1]:+.3f}]"
                         + f"  0,25 s − valor {q['0.25_menos_valor']['dif']:+.3f} [{q['0.25_menos_valor']['ic95'][0]:+.3f}; {q['0.25_menos_valor']['ic95'][1]:+.3f}]")
    sin = (~np.isfinite(t1)) | (t1 >= T_SIN)
    ixs = indices(grupos, sin, np.random.default_rng(747))
    out["antes"]["sin_individual"] = {"n": int(sin.sum()), "grupos": int(len(set(grupos[sin])))}
    for c in ("sp", "crudo", "vel"):
        pl = np.stack([Z[c][iw[str(w)], j0] for w in W])
        v = auc_filas(pl[:, sin], M[sin]); b = np.array([auc_filas(pl[:, i], M[i]) for i in ixs])
        out["antes"]["sin_individual"][c] = {str(w): par(v[i], b[:, i]) for i, w in enumerate(W)}
    L.append(f"sin respuesta individual antes de +{T_SIN:g} s (n = {int(sin.sum())}), cierre a +0,5 s: " + "; ".join(
        f"{c}: " + " ".join(f"{w} {q['auc']:.3f} [{q['ic95'][0]:.2f}; {q['ic95'][1]:.2f}]" for w, q in out["antes"]["sin_individual"][c].items())
        for c in ("sp", "crudo")))

    # ------------------------------------------------------------ 6. rapidez y emparejamiento
    out["rapidez"] = {"0.25@0.5": punto("vel", 0.25, 0.5), "valor@0": punto("vel", "valor", 0), "0.25@0": punto("vel", 0.25, 0),
                      "1@1": punto("vel", 1.0, 1), "1@0": punto("vel", 1.0, 0), "menos_giro": out["difs"]["vel_menos_giro_0.25@0.5"]}
    g_, s_ = Z["sp"][i0, j0], Z["vel"][i0, j0]
    gana = np.zeros(len(S)); masr = np.zeros(len(S))
    for k in range(len(S)):
        en = M[k]
        for i in np.flatnonzero(en):
            j = np.flatnonzero(~en)[np.argmin(np.abs(s_[k][~en] - s_[k][i]))]
            gana[k] += 1.0 if g_[k][i] > g_[k][j] else (0.5 if g_[k][i] == g_[k][j] else 0.0)
            masr[k] += 1.0 if s_[k][i] > s_[k][j] else 0.0
    npar = M.sum(1)
    be = np.array([gana[i].sum() / npar[i].sum() for i in idxs])
    out["rapidez"]["emparejado"] = {"frac": float(gana.sum() / npar.sum()), "ic95": [float(x) for x in ic(be)], "pares": int(npar.sum()),
                                    "frac_entrenado_mas_rapido": float(masr.sum() / npar.sum())}
    L.append(f"\nRapidez relativa (sólo pasado): 0,25 s a +0,5 s {out['rapidez']['0.25@0.5']['auc']:.3f} [{out['rapidez']['0.25@0.5']['ic95'][0]:.3f}; "
             f"{out['rapidez']['0.25@0.5']['ic95'][1]:.3f}]; antes del encendido (0,25 s, cierre 0) {out['rapidez']['0.25@0']['auc']:.3f}; 1 s a +1 s "
             f"{out['rapidez']['1@1']['auc']:.3f}; menos giro {out['rapidez']['menos_giro']['dif']:+.3f} [{out['rapidez']['menos_giro']['ic95'][0]:+.3f}; "
             f"{out['rapidez']['menos_giro']['ic95'][1]:+.3f}]")
    q = out["rapidez"]["emparejado"]
    L.append(f"Emparejado por rapidez: el entrenado supera en giro al no entrenado de rapidez más parecida en {q['frac']:.3f} "
             f"[{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}] de {q['pares']} pares (en {q['frac_entrenado_mas_rapido']:.3f} el entrenado sigue siendo el más rápido)")

    out["rapidez"]["spearman_giro_rapidez"] = {"no_entrenados": float(spearmanr(g_[~M], s_[~M]).statistic),
                                               "entrenados": float(spearmanr(g_[M], s_[M]).statistic)}
    L.append(f"Spearman giro–rapidez (0,25 s a +0,5 s, peces agrupados): no entrenados {out['rapidez']['spearman_giro_rapidez']['no_entrenados']:+.2f}; "
             f"entrenados {out['rapidez']['spearman_giro_rapidez']['entrenados']:+.2f}")

    # ------------------------------------------------------------ 7. variantes
    out["variantes"] = {}
    for c in ("ciclo", "k5"):
        pl = np.stack([Z[c][i0, j0], Z[c][iw["1.0"], it["1"]], Z[c][iw["valor"], it["0.375"]]])
        v = auc_filas(pl, M); b = con_ic(pl, M, idxs)
        out["variantes"][c] = {"0.25@0.5": par(v[0], b[:, 0]), "1@1": par(v[1], b[:, 1]), "valor@0.375": par(v[2], b[:, 2])}
        L.append(f"Variante {c}: 0,25 s a +0,5 s {v[0]:.3f} [{ic(b[:, 0])[0]:.3f}; {ic(b[:, 0])[1]:.3f}]; 1 s a +1 s {v[1]:.3f}; un valor a +0,375 s {v[2]:.3f}")

    # ------------------------------------------------------------ 8. respecto de t_E (descriptivo)
    selE = np.isfinite(tE) & (tE + LAGS[-1] < 7.9)
    out["desde_tE"] = {"n": int(selE.sum())}
    for c in ("cen", "sp"):
        ZE = np.full((nW, len(LAGS), len(S), M.shape[1]), np.nan)
        for k, e in enumerate(S):
            if selE[k]:
                ZE[:, :, k] = e["en_tE"][c]
        a = auc_filas(ZE.reshape(nW * len(LAGS), len(S), -1)[:, selE], M[selE]).reshape(nW, len(LAGS))
        out["desde_tE"][c] = {"auc": a.tolist(), "pico": {str(w): float(LAGS[int(np.argmax(a[i]))]) for i, w in enumerate(W)},
                              "en_tE": {str(w): float(a[i, int(np.argmin(np.abs(LAGS)))]) for i, w in enumerate(W)}}
    L.append(f"\nRespecto de t_E ({int(selE.sum())} ensayos): un valor centrado tiene su máximo {out['desde_tE']['cen']['pico']['valor']:+g} s "
             f"respecto de t_E (AUC {max(out['desde_tE']['cen']['auc'][0]):.3f}; en t_E {out['desde_tE']['cen']['en_tE']['valor']:.3f}); "
             f"sólo pasado, 0,25 s: máximo {out['desde_tE']['sp']['pico']['0.25']:+g} s (AUC {max(out['desde_tE']['sp']['auc'][1]):.3f}; "
             f"cierre en t_E {out['desde_tE']['sp']['en_tE']['0.25']:.3f})")

    # ------------------------------------------------------------ predicciones
    cp = out["control"]["sp"]["puntos"]
    pre0 = [out["curso"]["sp"]["auc"][i][it["0"]] for i in range(nW)]
    f, fc = out["fuera"]["sp"], out["fuera"]["crudo"]
    sinq = out["antes"]["sin_individual"]["sp"]["0.25"]
    emp = out["rapidez"]["emparejado"]
    out["pred"] = {
        "S1": {"ficticios": {k: q["auc"] for k, q in cp.items()}, "reales_cierre_0": pre0,
               "cumple": bool(all(abs(q["auc"] - 0.5) <= 0.05 for q in cp.values()) and all(abs(a - 0.5) <= 0.05 for a in pre0))},
        "S2": {"media": f["global"]["auc"], "p2.5": f["global"]["ic95"][0], "frac_0.25_o_0.5": f["frac_ventana"]["0.25"] + f["frac_ventana"]["0.5"],
               "cumple": bool(f["global"]["auc"] >= 0.60 and f["global"]["ic95"][0] > 0.55
                              and f["frac_ventana"]["0.25"] + f["frac_ventana"]["0.5"] >= 0.90)},
        "S3": {"sobre_0.5": out["grupos"]["sp"]["sobre_0.5"], "cumple": bool(out["grupos"]["sp"]["sobre_0.5"] >= 17)},
        "S4": {"a": f["corta_menos_1s"], "b": f["0.25_menos_valor"],
               "cumple_a": bool(f["corta_menos_1s"]["ic95"][0] > 0),
               "cumple_b": bool(abs(f["0.25_menos_valor"]["dif"]) < 0.05 and f["0.25_menos_valor"]["ic95"][0] <= 0 <= f["0.25_menos_valor"]["ic95"][1])},
        "S5": {**sinq, "n": out["antes"]["sin_individual"]["n"], "cumple": bool(sinq["auc"] >= 0.60 and sinq["ic95"][0] > 0.55)},
        "S6": {"a": fc["global"], "frac_0.25_o_0.5": fc["frac_ventana"]["0.25"] + fc["frac_ventana"]["0.5"],
               "valor": fc["por_ventana"]["valor"], "c_dif": fc["0.25_menos_valor"],
               "cumple_a": bool(fc["global"]["auc"] >= 0.58),
               "cumple_b": bool(fc["frac_ventana"]["0.25"] + fc["frac_ventana"]["0.5"] >= 0.90),
               "cumple_c": bool(fc["por_ventana"]["valor"]["auc"] <= 0.58 and fc["0.25_menos_valor"]["ic95"][0] > 0)},
        "S7": {"rapidez": out["rapidez"]["0.25@0.5"], "menos_giro": out["rapidez"]["menos_giro"], "emparejado": emp,
               "cumple_rapidez": bool(out["rapidez"]["0.25@0.5"]["auc"] >= 0.75 and out["rapidez"]["menos_giro"]["ic95"][0] > 0),
               "cumple_emparejado": bool(emp["frac"] >= 0.58 and emp["ic95"][0] > 0.55)},
        "S8": {"ciclo": out["variantes"]["ciclo"]["0.25@0.5"], "k5": out["variantes"]["k5"]["0.25@0.5"],
               "cumple_ciclo": bool(out["variantes"]["ciclo"]["0.25@0.5"]["auc"] >= 0.58 and out["variantes"]["ciclo"]["0.25@0.5"]["ic95"][0] > 0.55),
               "cumple_k5": bool(out["variantes"]["k5"]["0.25@0.5"]["auc"] >= 0.60 and out["variantes"]["k5"]["0.25@0.5"]["ic95"][0] > 0.55)}}
    for q in out["pred"].values():
        if "cumple" not in q:
            q["cumple"] = bool(all(v for k, v in q.items() if k.startswith("cumple_")))
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if q['cumple'] else 'FALLA'}  {json.dumps(q, default=float)}" for k, q in out["pred"].items()]
    os.makedirs(SAL, exist_ok=True)
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
