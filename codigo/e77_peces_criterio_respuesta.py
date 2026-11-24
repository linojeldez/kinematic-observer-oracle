"""Etapa 77 (revisión del 18-nov, comentario 3): ¿depende del criterio de respuesta individual el
AUC de los ensayos de peces cebra «sin respuesta individual antes de +0,6 s»?

Mismos 159 ensayos y misma cadena de sólo pasado de E74 (ventana de 0,25 s con cierre a +0,5 s).
La respuesta de cada pez no entrenado se recalcula con umbral de 2, 3 y 4 desviaciones estándar
sobre la media de los 7 s previos y duración sostenida de 0,125, 0,25 y 0,5 s (E73 y E74 usan
3 y 0,5 s). En cada variante se toman los ensayos en que ningún no entrenado responde antes de
+0,6 s y se calcula el AUC agrupado, con remuestreo de grupos y de ensayos.

Diseño y predicciones R1-R3 en docs/bitacora/resultados_E76-E85.md («E77»).

    PYTHONPATH=codigo python codigo/e77_peces_criterio_respuesta.py
"""
from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e73_peces_entrenados as e73
import e74_peces_solo_pasado as e74
from carga_segura import cargar

SAL = os.environ.get("E77_SALIDA", "resultados")
F_JSON, F_TXT = os.path.join(SAL, "E77_peces_criterio_respuesta.json"), os.path.join(SAL, "E77_peces_criterio_respuesta.txt")
UMBRALES = (2.0, 3.0, 4.0)            # desviaciones estándar sobre la media basal
SOSTENIDOS = (0.125, 0.25, 0.5)       # s que debe mantenerse
REF = (3.0, 0.5)                      # criterio de E73 y E74
T_SIN = e74.T_SIN                     # 0,6 s
IW = e74.W.index(0.25)
N_BOOT = e74.N_BOOT


def t_resp(vr, kf, FPS, n_sd, sost):
    """Como e73.t_respuesta, con el umbral y la duración como parámetros."""
    b = vr[kf - int(e73.T_BASE * FPS):kf]
    sobre = vr[kf:] > np.nanmean(b) + n_sd * np.nanstd(b)
    n = int(sost * FPS)
    for i in range(len(sobre) - n):
        if sobre[i:i + n].all():
            return i / FPS
    return np.nan


def ensayo(fn):
    d = cargar(fn)
    tr = np.asarray(d["trajectories"], float)
    FPS = int(d["frames_per_second"]); DT = 1.0 / FPS
    kf = int(d["key_frame"]); nb = int(e73.T_BASE * FPS)
    X, Y = tr[:, :, 0], tr[:, :, 1]
    N = tr.shape[1]
    ent = np.zeros(N, bool); ent[[int(i) for i in d["identities"]["trained"]]] = True
    Xc, Yc, Vxc, Vyc, _ = e73.cinematica(X, Y, DT)
    Xs, Ys, _, _, oms = e74.cinematica_sp(X, Y, DT)
    eps = e73.residuo(Xs, Ys, oms)
    x = np.abs(eps) / e74.ancho(eps, kf, nb)
    st = e74.cierres(x, kf, FPS, [0.5])[IW, 0]                      # ventana 0,25 s, cierre +0,5 s
    com = np.asarray(d["setup_points"][f"{d['light_side']}_feeder"], float).ravel()
    dx, dy = com[0] - Xc, com[1] - Yc
    vr = (Vxc * dx + Vyc * dy) / np.maximum(np.hypot(dx, dy), 1e-9)   # velocidad de cada pez hacia el comedero
    t1 = {}
    for u in UMBRALES:
        for s in SOSTENIDOS:
            ind = np.array([t_resp(vr[:, j], kf, FPS, u, s) for j in np.flatnonzero(~ent)])
            t1[f"{u:g}|{s:g}"] = float(np.nanmin(ind)) if np.isfinite(ind).any() else np.nan
    # control: la variante de referencia es la de E73
    ref = []
    for j in np.flatnonzero(~ent):
        q = np.zeros(N, bool); q[j] = True
        ref.append(e73.t_respuesta(Xc, Yc, Vxc, Vyc, q, com, kf, FPS))
    ref = np.array(ref)
    t1_e73 = float(np.nanmin(ref)) if np.isfinite(ref).any() else np.nan
    return {"grupo": d["video_code"], "ent": ent, "st": st, "t1": t1, "t1_e73": t1_e73}


if __name__ == "__main__":
    t0 = time.time()
    os.chdir(e73.RAIZ)
    fns = sorted(glob.glob(os.path.join(e73.DATOS, "stimstart", "*.npy")))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("FASE4_W", "4"))) as ex:
        S = list(ex.map(ensayo, fns))
    grupos = np.array([e["grupo"] for e in S]); M = np.array([e["ent"] for e in S])
    Z = np.stack([e["st"] for e in S])[None]                         # (1, ensayos, peces)
    kref = f"{REF[0]:g}|{REF[1]:g}"
    a, b = np.array([e["t1"][kref] for e in S]), np.array([e["t1_e73"] for e in S])
    assert np.array_equal(np.isfinite(a), np.isfinite(b)) and np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], atol=0, rtol=0)
    out = {"n_ensayos": len(S), "n_grupos": len(set(grupos)), "t_sin": T_SIN, "n_boot": N_BOOT,
           "umbrales": list(UMBRALES), "sostenidos": list(SOSTENIDOS), "ref": kref, "variantes": {}}
    todos = np.ones(len(S), bool)
    v = e74.auc_filas(Z, M)[0]
    bt = e74.con_ic(Z, M, e74.indices(grupos, todos, np.random.default_rng(741)))[:, 0]
    out["todos"] = e74.par(v, bt)
    L = ["=== E77 · peces cebra: sensibilidad del AUC condicional al criterio de respuesta individual ===",
         f"{len(S)} ensayos, {len(set(grupos))} grupos; ventana 0,25 s con cierre a +0,5 s (sólo pasado); "
         f"ensayos sin respuesta individual de ningún no entrenado antes de +{T_SIN:g} s",
         f"todos los ensayos: AUC {out['todos']['auc']:.4f} [{out['todos']['ic95'][0]:.3f}; {out['todos']['ic95'][1]:.3f}]",
         "umbral (DE) | sostenido (s) | ensayos | grupos | AUC [IC95]"]
    for i, u in enumerate(UMBRALES):
        for j, s in enumerate(SOSTENIDOS):
            k = f"{u:g}|{s:g}"
            t1 = np.array([e["t1"][k] for e in S])
            sin = (~np.isfinite(t1)) | (t1 >= T_SIN)
            rng = np.random.default_rng(747 if k == kref else 7700 + 10 * i + j)   # 747: la semilla de E74
            ixs = e74.indices(grupos, sin, rng)
            v = e74.auc_filas(Z[:, sin], M[sin])[0]
            bt = np.array([e74.auc_filas(Z[:, ix], M[ix])[0] for ix in ixs])
            q = {**e74.par(v, bt), "n": int(sin.sum()), "grupos": int(len(set(grupos[sin]))), "umbral": u, "sostenido": s}
            out["variantes"][k] = q
            L.append(f"{u:g} | {s:g} | {q['n']} | {q['grupos']} | {q['auc']:.4f} [{q['ic95'][0]:.3f}; {q['ic95'][1]:.3f}]"
                     + ("   <- criterio de E73 y E74" if k == kref else ""))
    V = out["variantes"]
    aucs = [q["auc"] for q in V.values()]; ns = [q["n"] for q in V.values()]
    out["resumen"] = {"auc_min": min(aucs), "auc_max": max(aucs), "n_min": min(ns), "n_max": max(ns),
                      "ic_inf_min": min(q["ic95"][0] for q in V.values())}
    e74a = json.load(open("resultados/E74_peces_solo_pasado.json"))["antes"]["sin_individual"]
    r = V[kref]
    laxa, estricta = V[f"{UMBRALES[0]:g}|{SOSTENIDOS[0]:g}"], V[f"{UMBRALES[-1]:g}|{SOSTENIDOS[-1]:g}"]
    out["pred"] = {
        "R1": {"n": r["n"], "auc": r["auc"], "ic95": r["ic95"], "e74": e74a["sp"]["0.25"], "n_e74": e74a["n"],
               "cumple": bool(r["n"] == e74a["n"] and abs(r["auc"] - e74a["sp"]["0.25"]["auc"]) < 1e-12
                              and np.allclose(r["ic95"], e74a["sp"]["0.25"]["ic95"], atol=1e-12))},
        "R2": {**out["resumen"], "cumple": bool(min(aucs) >= 0.60 and out["resumen"]["ic_inf_min"] > 0.5)},
        "R3": {"n_laxa": laxa["n"], "n_estricta": estricta["n"], "auc_laxa": laxa["auc"], "auc_ref": r["auc"],
               "cumple": bool(30 <= min(ns) <= 50 and 140 <= max(ns) <= 159 and laxa["auc"] >= r["auc"] - 0.03)}}
    L.append(f"\nAUC entre {min(aucs):.3f} y {max(aucs):.3f}; ensayos elegibles entre {min(ns)} y {max(ns)}; "
             f"menor extremo inferior de los intervalos {out['resumen']['ic_inf_min']:.3f}")
    for k, p in out["pred"].items():
        L.append(f"{k}: {'CUMPLE' if p['cumple'] else 'FALLA'}  " + json.dumps({a: b for a, b in p.items() if a != 'cumple'})[:300])
    out["segundos"] = time.time() - t0
    L.append(f"\n{out['segundos']:.0f} s")
    os.makedirs(SAL, exist_ok=True)
    json.dump(out, open(F_JSON, "w"), indent=1)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
