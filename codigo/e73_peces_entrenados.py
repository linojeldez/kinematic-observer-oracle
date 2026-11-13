"""Etapa 73 (plan 16): peces cebra con informados entrenados (Romero-Ferrero et al. 2023).

Cinco peces entrenados a asociar una luz con comida y diez desensibilizados nadan juntos (sesión
11; 6 grupos, 159 encendidos, 32 cuadros/s). Se aplica la cadena de los tetras (E54): residuo de la
velocidad angular contra los 3 vecinos más cercanos y estadístico de ventana media |eps|/sigma, y se
mide cuánto separa a los entrenados de los desensibilizados antes y después del encendido, con un
cuadro y con ventanas de 0,25-2 s. Controles: ventanas entre ensayos y antes del primer encendido.

Diseño y predicciones Z1-Z6 en docs/bitacora/resultados_E66-E75.md («E73»).

    PYTHONPATH=codigo python codigo/e73_peces_entrenados.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata, spearmanr

from carga_segura import cargar

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.path.join(RAIZ, "datos", "polavieja2023")
K = 3
N_SUAV = 5
VENT = ("cuadro", 0.25, 0.5, 1.0, 2.0)
TFIN = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
T_BASE = 7.0
N_BOOT = 2000
F_JSON, F_TXT = "resultados/E73_peces_entrenados.json", "resultados/E73_peces_entrenados.txt"


def suav(x, w):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, x)


def cinematica(X, Y, DT):
    """Como e27.cinematica (5 cuadros, unwrap por segmentos contiguos), con el DT del ensayo."""
    Xs, Ys = suav(X, N_SUAV), suav(Y, N_SUAV)
    Vx, Vy = np.gradient(Xs, DT, axis=0), np.gradient(Ys, DT, axis=0)
    th = np.arctan2(Vy, Vx)
    om = np.full_like(th, np.nan)
    for a in range(th.shape[1]):
        idx = np.flatnonzero(np.isfinite(th[:, a]))
        if len(idx) < 100:
            continue
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) < 50:
                continue
            om[seg, a] = np.gradient(np.unwrap(th[seg, a]), DT)
    return Xs, Ys, Vx, Vy, suav(om, N_SUAV)


def residuo(X, Y, om):
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    P = np.stack([X, Y], -1)
    for s in range(T):
        ok = np.isfinite(P[s]).all(1) & np.isfinite(om[s])
        if ok.sum() < K + 2:
            continue
        ii = np.flatnonzero(ok)
        D = np.linalg.norm(P[s][ii][:, None] - P[s][ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        eps[s, ii] = om[s, ii] - om[s, ii][np.argsort(D, 1)[:, :K]].mean(1)
    return eps


def t_respuesta(Xs, Ys, Vx, Vy, quien, comedero, kf, FPS):
    """Primer instante tras el encendido en que la velocidad media de `quien` hacia el comedero supera
    la media + 3 DE de los 7 s previos y se mantiene 0,5 s. NaN si no ocurre."""
    dx, dy = comedero[0] - Xs[:, quien], comedero[1] - Ys[:, quien]
    d = np.hypot(dx, dy)
    vr = np.nanmean((Vx[:, quien] * dx + Vy[:, quien] * dy) / np.maximum(d, 1e-9), axis=1)
    b = vr[kf - int(T_BASE * FPS):kf]
    umbral = np.nanmean(b) + 3 * np.nanstd(b)
    sobre = vr[kf:] > umbral
    n = int(0.5 * FPS)
    for i in range(len(sobre) - n):
        if sobre[i:i + n].all():
            return i / FPS
    return np.nan


def ensayo(fn):
    d = cargar(fn)
    tr = np.asarray(d["trajectories"], float)
    FPS = int(d["frames_per_second"])
    DT = 1.0 / FPS
    X, Y = tr[:, :, 0], tr[:, :, 1]
    kf = int(d["key_frame"])
    Xs, Ys, Vx, Vy, om = cinematica(X, Y, DT)
    eps = residuo(Xs, Ys, om)
    base = eps[kf - int(T_BASE * FPS):kf]
    base = base[np.isfinite(base)]
    sig = float(np.median(np.abs(base - np.median(base))) * 1.4826)
    ae = np.abs(eps) / sig
    N = tr.shape[1]
    ent = np.zeros(N, bool)
    ent[list(int(i) for i in d["identities"]["trained"])] = True
    st = {}
    for w in VENT:
        n = 1 if w == "cuadro" else int(round(w * FPS))
        for tf in TFIN:
            e = kf + int(round(tf * FPS))
            with np.errstate(all="ignore"):
                st[f"{w}|{tf:g}"] = np.nanmean(ae[e - n:e], axis=0).tolist()
    out = {"fn": os.path.basename(fn), "grupo": d["video_code"], "lote": d["experiment"], "video": d["video_code"],
           "sesion": int(d["session_number"]), "fps": int(d["frames_per_second"]), "N": int(N),
           "entrenados": ent.tolist(), "sigma": sig, "st": st, "nan_frac": float(np.isnan(tr).mean()),
           "lado": d.get("light_side"), "tipo": d.get("key_frame_name")}
    if d.get("light_side") in ("left", "right") and "setup_points" in d:
        com = np.asarray(d["setup_points"][f"{d['light_side']}_feeder"], float).ravel()
        tE = t_respuesta(Xs, Ys, Vx, Vy, ent, com, kf, FPS)
        tD = t_respuesta(Xs, Ys, Vx, Vy, ~ent, com, kf, FPS)
        out.update({"tE": tE, "tD": tD})
        if np.isfinite(tD):
            e = kf + int(round(tD * FPS))
            for w in VENT:
                n = 1 if w == "cuadro" else int(round(w * FPS))
                with np.errstate(all="ignore"):
                    out["st"][f"{w}|tD"] = np.nanmean(ae[e - n:e], axis=0).tolist()
    return out


def auc_rango(x1, x0):
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    r = rankdata(np.concatenate([x1, x0]))
    return float((r[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0)))


def auc_de(ens, clave):
    x1 = np.concatenate([np.asarray(e["st"][clave])[np.asarray(e["entrenados"])] for e in ens if clave in e["st"]])
    x0 = np.concatenate([np.asarray(e["st"][clave])[~np.asarray(e["entrenados"])] for e in ens if clave in e["st"]])
    return auc_rango(x1, x0)


def remuestras(ens, rng, n=N_BOOT):
    """Bootstrap de dos niveles: grupos y, dentro de cada grupo elegido, ensayos."""
    g = {}
    for e in ens:
        g.setdefault(e["grupo"], []).append(e)
    gs = sorted(g)
    for _ in range(n):
        out = []
        for k in rng.choice(len(gs), len(gs)):
            lst = g[gs[k]]
            out += [lst[i] for i in rng.integers(0, len(lst), len(lst))]
        yield out


def con_ic(ens, claves, rng):
    """AUC de cada clave y sus IC, con los mismos remuestreos para todas (para diferencias)."""
    v = {c: auc_de(ens, c) for c in claves}
    B = {c: [] for c in claves}
    for m in remuestras(ens, rng):
        for c in claves:
            B[c].append(auc_de(m, c))
    return v, {c: np.array(b) for c, b in B.items()}


def ic(b):
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


def cargar_conjunto(sub):
    fns = sorted(glob.glob(os.path.join(DATOS, sub, "*.npy")))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("FASE4_W", "4"))) as ex:
        return list(ex.map(ensayo, fns))


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(73)
    S = cargar_conjunto("stimstart")
    I = cargar_conjunto("intertrial")
    out = {"n": {}, "auc": {}, "ic": {}, "grupos": {}, "control_intertrial": {}, "resp": {}, "pred": {}}
    L = ["=== E73 · peces cebra con 5 entrenados y 10 desensibilizados (Romero-Ferrero et al. 2023, sesión 11) ==="]
    grupos = sorted({e["grupo"] for e in S})
    out["n"] = {"ensayos": len(S), "grupos": {g: sum(e["grupo"] == g for e in S) for g in grupos},
                "fps": sorted({e["fps"] for e in S}), "N": sorted({e["N"] for e in S}),
                "entrenados_por_ensayo": sorted({int(sum(e["entrenados"])) for e in S}),
                "nan_frac_max": max(e["nan_frac"] for e in S), "intertrial": len(I),
                "sigma_mediana": float(np.median([e["sigma"] for e in S]))}
    L.append(f"ensayos {len(S)} en {len(grupos)} grupos {out['n']['grupos']}; fps {out['n']['fps']}; N {out['n']['N']}; "
             f"entrenados por ensayo {out['n']['entrenados_por_ensayo']}; NaN máx {out['n']['nan_frac_max']:.3f}; "
             f"sigma mediana {out['n']['sigma_mediana']:.2f} rad/s")
    claves = [f"{w}|{tf:g}" for w in VENT for tf in TFIN]
    v, B = con_ic(S, claves, rng)
    for c in claves:
        out["auc"][c], out["ic"][c] = v[c], ic(B[c])
    L.append("\nAUC entrenado frente a desensibilizado (IC 95 % por grupos y ensayos); filas: ventana; columnas: t_fin (s)")
    L.append("ventana   " + "  ".join(f"{tf:>+16g}" for tf in TFIN))
    for w in VENT:
        L.append(f"{str(w):9s} " + "  ".join(f"{v[f'{w}|{tf:g}']:.3f} [{ic(B[f'{w}|{tf:g}'])[0]:.2f};{ic(B[f'{w}|{tf:g}'])[1]:.2f}]"
                                           for tf in TFIN))
    # diferencias pareadas: 1 s frente a un cuadro
    out["dif_1s_cuadro"] = {}
    for tf in TFIN:
        dd = B[f"1.0|{tf:g}"] - B[f"cuadro|{tf:g}"]
        out["dif_1s_cuadro"][f"{tf:g}"] = {"dif": v[f"1.0|{tf:g}"] - v[f"cuadro|{tf:g}"], "ic95": ic(dd)}
    L.append("\nAUC(1 s) − AUC(1 cuadro): " + "  ".join(
        f"{tf:+g} s: {q['dif']:+.3f} [{q['ic95'][0]:+.3f}; {q['ic95'][1]:+.3f}]" for tf, q in
        ((float(k), q) for k, q in out["dif_1s_cuadro"].items())))
    # por grupo
    for g in grupos:
        sub = [e for e in S if e["grupo"] == g]
        out["grupos"][g] = {c: auc_de(sub, c) for c in ("1.0|1", "cuadro|1", "1.0|0", "1.0|2")}
    L.append(f"\npor grupo, {len(grupos)} grupos (AUC 1 s a +1 s; un cuadro a +1 s; 1 s a 0 s): " + "; ".join(
        f"{g[-6:]} {q['1.0|1']:.2f}/{q['cuadro|1']:.2f}/{q['1.0|0']:.2f}" for g, q in out["grupos"].items()))
    # control: ventanas entre ensayos
    vi, Bi = con_ic(I, [f"{w}|{tf:g}" for w in ("cuadro", 1.0) for tf in TFIN], np.random.default_rng(74))
    out["control_intertrial"] = {c: {"auc": vi[c], "ic95": ic(Bi[c])} for c in vi}
    L.append("\ncontrol entre ensayos (encendido ficticio), AUC 1 s: " + "  ".join(
        f"{tf:+g}: {vi[f'1.0|{tf:g}']:.3f}" for tf in TFIN))
    # tiempos de respuesta
    tE = np.array([e.get("tE", np.nan) for e in S]); tD = np.array([e.get("tD", np.nan) for e in S])
    out["resp"] = {"tE_mediana": float(np.nanmedian(tE)), "tD_mediana": float(np.nanmedian(tD)),
                   "tE_iqr": [float(np.nanpercentile(tE, 25)), float(np.nanpercentile(tE, 75))],
                   "tD_iqr": [float(np.nanpercentile(tD, 25)), float(np.nanpercentile(tD, 75))],
                   "frac_tE": float(np.isfinite(tE).mean()), "frac_tD": float(np.isfinite(tD).mean()),
                   "E_antes_D": float(np.nanmean((tE < tD)[np.isfinite(tE) & np.isfinite(tD)]))}
    cl_tD = [f"{w}|tD" for w in VENT]
    con_tD = [e for e in S if "1.0|tD" in e["st"]]
    vt, Bt = con_ic(con_tD, cl_tD, np.random.default_rng(75))
    out["auc_tD"] = {c: {"auc": vt[c], "ic95": ic(Bt[c])} for c in cl_tD}
    L.append(f"\nrespuesta hacia el comedero iluminado: entrenados t_E mediana {out['resp']['tE_mediana']:.2f} s "
             f"[{out['resp']['tE_iqr'][0]:.2f}; {out['resp']['tE_iqr'][1]:.2f}] ({100 * out['resp']['frac_tE']:.0f} % de ensayos); "
             f"desensibilizados t_D {out['resp']['tD_mediana']:.2f} s [{out['resp']['tD_iqr'][0]:.2f}; {out['resp']['tD_iqr'][1]:.2f}] "
             f"({100 * out['resp']['frac_tD']:.0f} %); t_E < t_D en {100 * out['resp']['E_antes_D']:.0f} %")
    dd = Bt["1.0|tD"] - Bt["cuadro|tD"]
    out["posterior_dif_tD"] = {"dif": vt["1.0|tD"] - vt["cuadro|tD"], "ic95": ic(dd), "n_ensayos": len(con_tD),
                               "nota": "análisis posterior, no declarado"}
    L.append("AUC con ventanas que terminan en t_D: " + "  ".join(
        f"{c.split('|')[0]}: {q['auc']:.3f} [{q['ic95'][0]:.2f}; {q['ic95'][1]:.2f}]" for c, q in out["auc_tD"].items()))
    L.append(f"(posterior, no declarado) AUC(1 s) − AUC(1 cuadro) en t_D: {out['posterior_dif_tD']['dif']:+.3f} "
             f"[{out['posterior_dif_tD']['ic95'][0]:+.3f}; {out['posterior_dif_tD']['ic95'][1]:+.3f}] ({len(con_tD)} ensayos)")
    # Z6: el estadístico de los entrenados predice la latencia de los desensibilizados
    def rho(ens, c):
        m = [(np.nanmax(np.asarray(e["st"][c])[np.asarray(e["entrenados"])]), e["tD"]) for e in ens if np.isfinite(e.get("tD", np.nan))]
        m = np.array(m)
        return float(spearmanr(m[:, 0], m[:, 1]).statistic) if len(m) > 5 else np.nan
    r1, rc = rho(S, "1.0|1"), rho(S, "cuadro|1")
    b1, bc = [], []
    for m in remuestras(S, np.random.default_rng(76), 1000):
        b1.append(rho(m, "1.0|1")); bc.append(rho(m, "cuadro|1"))
    b1, bc = np.array(b1), np.array(bc)
    out["Z6"] = {"rho_1s": r1, "ic_1s": ic(b1), "rho_cuadro": rc, "ic_cuadro": ic(bc),
                 "dif_abs": float(abs(r1) - abs(rc)), "ic_dif_abs": ic(np.abs(b1) - np.abs(bc))}
    L.append(f"Spearman(estadístico de los entrenados en el 1.er s, t_D): 1 s {r1:+.3f} [{ic(b1)[0]:+.2f}; {ic(b1)[1]:+.2f}]; "
             f"un cuadro {rc:+.3f} [{ic(bc)[0]:+.2f}; {ic(bc)[1]:+.2f}]")
    # variante conservadora: bootstrap por tanda (6 fechas) en lugar de por grupo (20)
    SL = [dict(e, grupo=e["lote"]) for e in S]
    vl, Bl = con_ic(SL, ["1.0|1", "cuadro|1"], np.random.default_rng(77))
    SLt = [e for e in SL if "1.0|tD" in e["st"]]
    vlt, Blt = con_ic(SLt, ["1.0|tD", "cuadro|tD"], np.random.default_rng(78))
    out["por_tanda"] = {"auc_1s_+1": {"auc": vl["1.0|1"], "ic95": ic(Bl["1.0|1"])},
                        "dif_+1": {"dif": vl["1.0|1"] - vl["cuadro|1"], "ic95": ic(Bl["1.0|1"] - Bl["cuadro|1"])},
                        "dif_tD": {"dif": vlt["1.0|tD"] - vlt["cuadro|tD"], "ic95": ic(Blt["1.0|tD"] - Blt["cuadro|tD"])}}
    L.append("\n(conservador, IC por las 6 tandas) AUC 1 s a +1 s " + f"{vl['1.0|1']:.3f} [{ic(Bl['1.0|1'])[0]:.3f}; {ic(Bl['1.0|1'])[1]:.3f}]; "
             f"1 s − 1 cuadro a +1 s {out['por_tanda']['dif_+1']['dif']:+.3f} [{out['por_tanda']['dif_+1']['ic95'][0]:+.3f}; "
             f"{out['por_tanda']['dif_+1']['ic95'][1]:+.3f}]; en t_D {out['por_tanda']['dif_tD']['dif']:+.3f} "
             f"[{out['por_tanda']['dif_tD']['ic95'][0]:+.3f}; {out['por_tanda']['dif_tD']['ic95'][1]:+.3f}]")
    # predicciones
    A = out["auc"]
    mejor2 = max(("0.25", "0.5", "1.0", "2.0"), key=lambda w: A[f"{w}|2"])
    out["pred"] = {
        "Z1": {"auc": A["1.0|1"], "ic": out["ic"]["1.0|1"], "grupos_sobre_0.5": sum(q["1.0|1"] > 0.5 for q in out["grupos"].values()),
               "n_grupos": len(out["grupos"]),
               "cumple": bool(A["1.0|1"] >= 0.65 and out["ic"]["1.0|1"][0] > 0.5
                              and sum(q["1.0|1"] > 0.5 for q in out["grupos"].values()) >= 5 / 6 * len(out["grupos"]))},
        "Z2": {"auc_pre": A["1.0|0"], "intertrial": {k: q["auc"] for k, q in out["control_intertrial"].items() if k.startswith("1.0")},
               "cumple": bool(abs(A["1.0|0"] - 0.5) <= 0.10 and all(abs(q["auc"] - 0.5) <= 0.10
                                                                    for k, q in out["control_intertrial"].items() if k.startswith("1.0")))},
        "Z3": {**out["dif_1s_cuadro"]["1"], "cumple": bool(out["dif_1s_cuadro"]["1"]["ic95"][0] > 0)},
        "Z4": {"tE": out["resp"]["tE_mediana"], "tD": out["resp"]["tD_mediana"], "auc_tD": out["auc_tD"]["1.0|tD"]["auc"],
               "cumple": bool(out["resp"]["tE_mediana"] < out["resp"]["tD_mediana"] and out["auc_tD"]["1.0|tD"]["auc"] >= 0.60)},
        "Z5": {"mejor_ventana_+2s": mejor2, "auc": {w: A[f"{w}|2"] for w in ("0.25", "0.5", "1.0", "2.0")},
               "cumple": bool(mejor2 in ("0.25", "0.5", "1.0"))},
        "Z6": {**out["Z6"], "cumple": bool(out["Z6"]["ic_1s"][1] < 0 and abs(r1) > abs(rc))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if q['cumple'] else 'FALLA'}  {json.dumps(q, default=float)}" for k, q in out["pred"].items()]
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
