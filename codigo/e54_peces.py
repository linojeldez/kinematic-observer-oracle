"""Etapa 54 (B1 a B3 del plan de fortalecimiento): tetras de Lecheval, con el estimador y
la ventana de las grajillas.

B1  Huella T95 (el estimador principal de E51) con ventana de 2 s y k = 3; con ventana de
    5 s como cota por censura; curva AUC frente a la ventana en rejilla fina (un cuadro;
    0,1 a 1,0 s en pasos de 0,1; 1,25, 1,5 y 2 s); ventana más corta dentro del 95 % del
    máximo; diferencia pareada AUC(1 s) - AUC(un cuadro). IC por remuestreo de experimentos.
B2  Agilidad con el estimador de aceleración (a_perp / v) y su IC.
B3  Sensibilidad a la detección: umbral de giro colectivo y separación mínima.

Usa las funciones de e27_tetras.py sin modificarlas; con la configuración por defecto debe
reproducir E27 y E28 (ancla). Diseño y predicciones en docs/bitacora/resultados_E46-E55.md.

    PYTHONPATH=codigo python codigo/e54_peces.py
"""

from __future__ import annotations

import glob
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e27_tetras as e27
from e51_huella_aves import fwhm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT = e27.DT
K, Q = 3, 0.25
TAUS = ("cuadro", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0)
BASE = (0.3, 10.0)
CONFIGS = [(o, s) for s in (10.0, 5.0) for o in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
N_SUAV = int(round(0.2 / DT))            # 0,2 s, como en E51
N_BOOT = 2000


def _dphi(X, Y):
    cx, cy = np.nanmean(X), np.nanmean(Y)
    phi = np.arctan2(Y - cy, X - cx)
    dphi = np.full_like(phi, np.nan)
    for a in range(phi.shape[1]):
        idx = np.flatnonzero(np.isfinite(phi[:, a]))
        if len(idx) < 100:
            continue
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) < 50:
                continue
            dphi[seg, a] = np.gradient(np.unwrap(phi[seg, a]), DT)
    return dphi


def detectar(dphi, om_min, sep_min):
    """La detección de e27.uturns, con el umbral y la separación como parámetros."""
    Om = e27.suav(np.nanmean(dphi, axis=1), int(1.0 / DT))
    w = int(2.0 / DT)
    eventos, ultimo = [], -10**9
    for s in np.flatnonzero(np.diff(np.sign(Om)) != 0):
        if s - ultimo < sep_min / DT or s < 3 * w or s > len(Om) - 3 * w:
            continue
        antes = np.nanmean(Om[s - int(3 / DT):s - int(1 / DT)])
        despues = np.nanmean(Om[s + int(1 / DT):s + int(3 / DT)])
        if np.sign(antes) != np.sign(despues) and abs(antes) > om_min and abs(despues) > om_min:
            eventos.append(int(s))
            ultimo = s
    return eventos


def suav_nan(x, n):
    k = np.ones(n)
    f = np.isfinite(x)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, np.where(f, x, 0.0))
    den = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, f.astype(float))
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def experimento(arg):
    fn, grupo = arg
    min_aves = 3 if grupo == "4" else None
    min_finitos = 4 if grupo == "4" else None
    rng = np.random.default_rng(zlib.crc32(os.path.basename(fn).encode()))
    X, Y = e27.cargar(fn)
    Vx, Vy, om = e27.cinematica(X, Y, 1.0)
    dphi = _dphi(X, Y)
    T, N = om.shape
    eps = np.full_like(om, np.nan)
    Pxy = np.stack([X, Y], -1)
    for s in range(T):
        p, o = Pxy[s], om[s]
        okk = np.isfinite(p).all(1) & np.isfinite(o)
        if okk.sum() < (min_finitos if min_finitos else K + 2):
            continue
        ii = np.flatnonzero(okk)
        D = np.linalg.norm(p[ii][:, None] - p[ii][None, :], axis=2)
        np.fill_diagonal(D, np.inf)
        eps[s, ii] = o[ii] - o[ii][np.argsort(D, 1)[:, :K]].mean(1)
    aeps = np.abs(eps)
    aeps_s = suav_nan(aeps, N_SUAV)
    dphis = e27.suav(dphi, int(0.5 / DT))

    salida = {"fn": os.path.basename(fn), "grupo": grupo, "dur_min": T * DT / 60.0, "cfg": {}}
    for om_min, sep in CONFIGS:
        eventos = detectar(dphi, om_min, sep)
        base = (om_min, sep) == BASE
        r = {"n_det": len(eventos), "ev": []}
        if not eventos:
            salida["cfg"][f"{om_min:g}|{sep:g}"] = r
            continue
        lejos = np.ones(T, bool)
        for s in eventos:
            lejos[max(0, s - int(5 / DT)):s + int(5 / DT)] = False
        fb = np.isfinite(eps) & lejos[:, None]
        if fb.sum() < 100:
            salida["cfg"][f"{om_min:g}|{sep:g}"] = r
            continue
        sig = float(np.nanmedian(np.abs(eps[fb] - np.nanmedian(eps[fb]))) * 1.4826)
        r["sigma"] = sig
        if base:
            r.update({"omega": [], "omega_alat": [], "hue2": [], "hue5": [], "t95_2": [], "t95_5": [],
                      "t95_2_todos": [], "t95_nulo": [], "curvas": []})
        for s0 in eventos:
            vent = slice(max(0, s0 - int(3 / DT)), min(T, s0 + int(3 / DT)))
            t_rev = np.full(N, np.nan)
            for a in range(N):
                tr = dphis[vent, a]
                okk = np.isfinite(tr)
                if okk.sum() < 20:
                    continue
                cz = np.flatnonzero(np.diff(np.sign(tr[okk])) != 0)
                if len(cz):
                    t_rev[a] = (vent.start + np.flatnonzero(okk)[cz[0]]) * DT
            okr = np.isfinite(t_rev)
            if okr.sum() < (min_aves if min_aves else max(4, int(N * 0.6))):
                continue
            orden = np.argsort(t_rev)
            d_rel = t_rev - np.nanmin(t_rev)
            lim_i, lim_s = np.nanquantile(d_rel[okr], Q), np.nanquantile(d_rel[okr], 1 - Q)
            ini = [a for a in range(N) if okr[a] and d_rel[a] <= lim_i]
            seg = [a for a in range(N) if okr[a] and d_rel[a] >= lim_s]
            fila = {"ini": ini, "seg": seg, "st": {}}
            for tau in TAUS:
                n = 1 if tau == "cuadro" else int(tau / DT)
                if s0 - n < 0:
                    continue
                with np.errstate(all="ignore"):
                    fila["st"][str(tau)] = (np.nanmean(aeps[s0 - n:s0], axis=0) / sig).tolist()
            r["ev"].append(fila)
            if not base:
                continue
            vturn = slice(max(0, s0 - int(1 / DT)), min(T, s0 + int(2 / DT)))
            omv = np.abs(om[vturn])
            if np.isfinite(omv).any():
                r["omega"].append(float(np.nanpercentile(omv, 90)))
            # como en e27.analizar: la aceleración se deriva dentro de la ventana del giro
            vx, vy = Vx[vturn], Vy[vturn]
            ax_, ay_ = np.gradient(vx, DT, axis=0), np.gradient(vy, DT, axis=0)
            rp = np.sqrt(vx**2 + vy**2)
            oma = (np.abs(ax_ * (-vy) + ay_ * vx) / np.maximum(rp, 1e-9)) / np.maximum(rp, 1e-9)
            if np.isfinite(oma).any():
                r["omega_alat"].append(float(np.nanpercentile(oma, 90)))
            a0 = int(orden[0])
            s_ini = int(round(t_rev[a0] / DT))
            # operador anterior (E27 y E28), para el ancla
            for cap, dest in ((2.0, "hue2"), (5.0, "hue5")):
                tramo = aeps[s_ini:s_ini + int(cap / DT), a0] / sig
                if np.isfinite(tramo).mean() > 0.7:
                    sm = e27.suav(np.nan_to_num(tramo, nan=0.0), 9)
                    ip = int(np.argmax(sm))
                    dur = cap
                    for j in range(ip, len(sm) - 9):
                        if sm[j] < 1.5 and np.all(sm[j:j + 9] < 1.5):
                            dur = j * DT
                            break
                    r[dest].append(dur)
            # T95: el foco frente a los peces fuera del cuartil iniciador, en los mismos cuadros
            no_ini = np.array([a for a in range(N) if okr[a] and a not in ini])
            if len(no_ini) >= 2:
                for cap, dest in ((2.0, "t95_2"), (5.0, "t95_5")):
                    n = int(cap / DT)
                    if s_ini + n > T:
                        continue
                    foco = aeps_s[s_ini:s_ini + n, a0] / sig
                    if np.isfinite(foco).mean() <= 0.7:
                        continue
                    ref = aeps_s[s_ini:s_ini + n][:, no_ini] / sig
                    thr = np.nanpercentile(ref, 95)
                    r[dest].append(float(np.nansum(foco > thr) * DT))
                    if cap == 2.0:
                        m = int(rng.choice(no_ini))
                        resto = no_ini[no_ini != m]
                        thr_n = np.nanpercentile(aeps_s[s_ini:s_ini + n][:, resto] / sig, 95)
                        r["t95_nulo"].append(float(np.nansum(aeps_s[s_ini:s_ini + n, m] / sig > thr_n) * DT))
                        vals = []
                        for a in ini:
                            sa = int(round(t_rev[a] / DT))
                            if sa + n > T:
                                continue
                            fa = aeps_s[sa:sa + n, a] / sig
                            if np.isfinite(fa).mean() <= 0.7:
                                continue
                            th = np.nanpercentile(aeps_s[sa:sa + n][:, no_ini] / sig, 95)
                            vals.append(float(np.nansum(fa > th) * DT))
                        if vals:
                            r["t95_2_todos"].append(float(np.median(vals)))
            n_pre, n_post = int(0.3 / DT), int(2.0 / DT)
            if s_ini - n_pre >= 0 and s_ini + n_post <= T:
                c = aeps[s_ini - n_pre:s_ini + n_post, a0] / sig
                if np.isfinite(c).mean() > 0.7:
                    r["curvas"].append(c)
        if base and r["curvas"]:
            cur = np.nanmedian(np.vstack(r["curvas"]), axis=0)
            tt = (np.arange(-int(0.3 / DT), int(2.0 / DT))) * DT
            w, cens, pico, tp = fwhm(tt, cur, N_SUAV)
            r["fwhm"], r["fwhm_cens"], r["n_curvas"] = w, cens, len(r["curvas"])
        r.pop("curvas", None)
        salida["cfg"][f"{om_min:g}|{sep:g}"] = r
    return salida


def auc(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    return float(((x1[:, None] > x0[None, :]).sum()
                  + 0.5 * (x1[:, None] == x0[None, :]).sum()) / (len(x1) * len(x0)))


def pares(exps, cfg):
    """Por experimento y ventana: (valores de iniciadores, valores de seguidores)."""
    out = []
    for e in exps:
        r = e["cfg"].get(cfg, {})
        d = {str(t): ([], []) for t in TAUS}
        for ev in r.get("ev", []):
            for t, vals in ev["st"].items():
                v = np.array(vals)
                d[t][0].extend(v[ev["ini"]].tolist())
                d[t][1].extend(v[ev["seg"]].tolist())
        out.append({t: (np.array(a), np.array(b)) for t, (a, b) in d.items()})
    return out


def curva(pp, idx=None):
    idx = range(len(pp)) if idx is None else idx
    return {t: auc(np.concatenate([pp[i][str(t)][0] for i in idx]),
                   np.concatenate([pp[i][str(t)][1] for i in idx])) for t in TAUS}


def tau95(c):
    fin = {t: a for t, a in c.items() if t != "cuadro" and np.isfinite(a)}
    tope = max(fin.values())
    return min(t for t, a in fin.items() if a >= 0.95 * tope)


def resumen(exps, cfg, rng, boot=True):
    exps = [e for e in exps if e["cfg"].get(cfg, {}).get("ev")]
    pp = pares(exps, cfg)
    c = curva(pp)
    out = {"n_exp": len(exps), "n_eventos": int(sum(len(e["cfg"][cfg]["ev"]) for e in exps)),
           "n_detectados": int(sum(e["cfg"][cfg]["n_det"] for e in exps)),
           "curva": {str(t): a for t, a in c.items()}, "tau95": tau95(c),
           "auc_max": max(a for t, a in c.items() if t != "cuadro"),
           "delta_1s_cuadro": c[1.0] - c["cuadro"]}
    if not boot:
        return out
    B = rng.integers(0, len(exps), (N_BOOT, len(exps)))
    cb = [curva(pp, b) for b in B]
    ic = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out["curva_ic95"] = {str(t): ic([x[t] for x in cb]) for t in TAUS}
    out["tau95_ic95"] = ic([tau95(x) for x in cb])
    out["delta_ic95"] = ic([x[1.0] - x["cuadro"] for x in cb])
    for nom in ("omega", "omega_alat", "hue2", "hue5", "t95_2", "t95_5", "t95_2_todos", "t95_nulo"):
        por = [np.array(e["cfg"][cfg].get(nom, [])) for e in exps]
        todo = np.concatenate(por) if por else np.array([])
        if not len(todo):
            continue
        med = [np.median(np.concatenate([por[i] for i in b])) for b in B]
        out[nom] = {"mediana": float(np.median(todo)), "media": float(todo.mean()), "ic95": ic(med),
                    "n": int(len(todo)), "_boot": med}
    fw = np.array([e["cfg"][cfg].get("fwhm", np.nan) for e in exps], float)
    fw = fw[np.isfinite(fw)]
    if len(fw):
        bb = rng.integers(0, len(fw), (N_BOOT, len(fw)))
        out["fwhm_por_experimento"] = {"mediana": float(np.median(fw)),
                                       "ic95": ic(np.median(fw[bb], axis=1)), "n": int(len(fw)),
                                       "censuradas": int(sum(bool(e["cfg"][cfg].get("fwhm_cens"))
                                                             for e in exps))}
    return out


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(54)
    jobs = [(fn, g) for g in ("4", "5", "8", "10")
            for fn in sorted(glob.glob(os.path.join(e27.DATA, g, "*.csv")))]
    with ProcessPoolExecutor(max_workers=8) as ex:
        exps = list(ex.map(experimento, jobs))
    L, out = ["=== E54 · tetras: misma ventana, mismo estimador y mismo k que las grajillas ==="], {}
    P = L.append
    cfg0 = f"{BASE[0]:g}|{BASE[1]:g}"
    g810 = [e for e in exps if e["grupo"] in ("8", "10")]

    # ---- ancla contra E27 y E28
    r = resumen(g810, cfg0, rng)
    e27j = json.load(open("resultados/E27_tetras.json"))
    e28j = json.load(open("resultados/E28_robustez.json"))
    dm = max(abs(r["curva"][str(t)] - e27j["curva_auc"][str(t)]) for t in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0))
    anc = {"n_eventos": [r["n_eventos"], e27j["n_eventos"]], "curva_delta_max": dm,
           "omega": [r["omega"]["mediana"], e27j["omega_max"]],
           "omega_alat": [r["omega_alat"]["mediana"], e28j["base"]["omega_alat"]],
           "huella_2s": [r["hue2"]["mediana"], e27j["huella"]],
           "huella_5s": [r["hue5"]["mediana"], e28j["cap5_huella"]]}
    anc["cumple"] = bool(anc["n_eventos"][0] == anc["n_eventos"][1] and dm < 1e-9
                         and all(abs(a - b) < 1e-9 for a, b in (anc["omega"], anc["omega_alat"],
                                                               anc["huella_2s"], anc["huella_5s"])))
    out["ancla"] = anc
    P(f"ancla (N = 8 y 10, configuración por defecto) frente a E27 y E28: {json.dumps(anc)}")

    def informe(nombre, sub, clave):
        r = resumen(sub, cfg0, rng)
        out[clave] = {k: ({kk: vv for kk, vv in v.items() if kk != "_boot"} if isinstance(v, dict) else v)
                      for k, v in r.items()}
        P(f'\n--- {nombre}: {r["n_exp"]} experimentos, {r["n_eventos"]} giros analizados '
          f'({r["n_detectados"]} detectados) ---')
        P("  AUC: " + " | ".join(f'{t}: {r["curva"][str(t)]:.3f} [{r["curva_ic95"][str(t)][0]:.3f}; '
                                 f'{r["curva_ic95"][str(t)][1]:.3f}]' for t in TAUS))
        P(f'  ventana más corta dentro del 95 % del máximo: {r["tau95"]:g} s {r["tau95_ic95"]}; '
          f'Δ = AUC(1 s) − AUC(un cuadro) = {r["delta_1s_cuadro"]:+.3f} [{r["delta_ic95"][0]:+.3f}; '
          f'{r["delta_ic95"][1]:+.3f}]')
        for nom, et in (("t95_2", "T95, ventana de 2 s (primer pez)"), ("t95_nulo", "T95 nulo (un pez no iniciador)"),
                        ("t95_2_todos", "T95, ventana de 2 s (cuartil iniciador)"),
                        ("t95_5", "T95, ventana de 5 s"), ("hue2", "operador anterior, 2 s"),
                        ("hue5", "operador anterior, 5 s"), ("omega_alat", "agilidad a_perp/v"),
                        ("omega", "agilidad por rumbo")):
            if nom in r:
                v = r[nom]
                P(f'  {et}: mediana {v["mediana"]:.3f}, media {v["media"]:.3f}, IC 95 % de la mediana '
                  f'[{v["ic95"][0]:.3f}; {v["ic95"][1]:.3f}] (n = {v["n"]})')
        if "fwhm_por_experimento" in r:
            v = r["fwhm_por_experimento"]
            P(f'  anchura a media altura de la curva mediana por experimento: {v["mediana"]:.2f} s '
              f'[{v["ic95"][0]:.2f}; {v["ic95"][1]:.2f}] ({v["n"]} experimentos, {v["censuradas"]} censuradas)')
        return r

    r810 = informe("N = 8 y 10", g810, "N8_10")
    for g in ("4", "5", "8", "10"):
        informe(f"N = {g}", [e for e in exps if e["grupo"] == g], f"N{g}")

    # ---- comparación con las grajillas (E51 y E52), mismo estimador, misma ventana, k = 3
    e51 = json.load(open("resultados/E51_huella_aves.json"))
    e52 = json.load(open("resultados/E52_auc_ventana_aves.json"))
    aves = np.array(e51["k3_min12"]["T95"]["por_evento"])
    ba = np.median(aves[rng.integers(0, len(aves), (N_BOOT, len(aves)))], axis=1)
    bt = np.array(r810["t95_2"]["_boot"])
    coc = bt / np.maximum(ba, 1e-9)
    dif = bt - ba
    out["especies"] = {
        "T95_grajilla": {"mediana": float(np.median(aves)), "ic95": e51["k3_min12"]["T95"]["ic95"],
                         "nulo": e51["k3_min12"]["T95_nulo_media"]["mediana"]},
        "T95_tetra": {"mediana": r810["t95_2"]["mediana"], "ic95": r810["t95_2"]["ic95"],
                      "nulo": r810["t95_nulo"]["mediana"]},
        "diferencia_tetra_menos_grajilla": {"valor": float(r810["t95_2"]["mediana"] - np.median(aves)),
                                            "ic95": [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))]},
        "cociente_definido_en": float(np.mean(ba > 0.02)),
        "tau95_grajilla": e52["k3|1|todos"]["tau95"], "tau95_tetra": [r810["tau95"], r810["tau95_ic95"]],
        "delta_grajilla": e52["k3|1|todos"]["delta"], "delta_tetra": [r810["delta_1s_cuadro"], r810["delta_ic95"]]}
    P("\n--- especies, mismo estimador (T95, ventana de 2 s, k = 3) ---")
    P(json.dumps(out["especies"], ensure_ascii=False))

    # ---- B3: sensibilidad a la detección
    P("\n--- B3: sensibilidad a la detección (N = 8 y 10) ---")
    P(f'{"umbral":>7s} {"sep":>4s} {"N=8 det":>8s} {"N=10 det":>9s} {"analiz.":>8s} {"AUC 1 s":>8s} '
      f'{"tau95":>6s} {"AUC cuadro":>10s}')
    out["deteccion"] = []
    for om_min, sep in CONFIGS:
        c = f"{om_min:g}|{sep:g}"
        rr = resumen(g810, c, rng, boot=False)
        n8 = sum(e["cfg"][c]["n_det"] for e in exps if e["grupo"] == "8")
        n10 = sum(e["cfg"][c]["n_det"] for e in exps if e["grupo"] == "10")
        out["deteccion"].append({"om_min": om_min, "sep": sep, "det_N8": int(n8), "det_N10": int(n10),
                                 "analizados": rr["n_eventos"], "auc_1s": rr["curva"]["1.0"],
                                 "tau95": rr["tau95"], "auc_cuadro": rr["curva"]["cuadro"]})
        P(f'{om_min:7.2f} {sep:4.0f} {n8:8d} {n10:9d} {rr["n_eventos"]:8d} {rr["curva"]["1.0"]:8.3f} '
          f'{rr["tau95"]:6.1f} {rr["curva"]["cuadro"]:10.3f}')
    d = out["deteccion"]
    P(f'rango: AUC a 1 s {min(x["auc_1s"] for x in d):.3f} a {max(x["auc_1s"] for x in d):.3f}; '
      f'ventana {min(x["tau95"] for x in d):g} a {max(x["tau95"] for x in d):g} s; giros con N = 8: '
      f'{min(x["det_N8"] for x in d)} a {max(x["det_N8"] for x in d)}; con N = 10: '
      f'{min(x["det_N10"] for x in d)} a {max(x["det_N10"] for x in d)}')
    dur = [e["dur_min"] for e in g810]
    out["duracion_min_8_10"] = [float(min(dur)), float(np.median(dur)), float(max(dur))]
    with open("resultados/E54_peces.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    with open("resultados/E54_peces.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
