"""Etapa 75b: cuándo se parte la bandada en vuelo sin informados ni falsas alarmas.

E64 guardó la cohesión sólo al final de los 600 s; en la réplica 0 (semilla 11) vale 0,556 con el
integrador de pi0 = 0,173 y 0,3 y con los sin memoria x6 y x12. Aquí se repite esa réplica con el
mismo protocolo (job_P de E64) guardando cada 5 s la fracción del componente mayor, el número de
componentes y la polarización. El rumbo debe reproducir la serie guardada.

Diseño y predicción F7 en docs/bitacora/resultados_E66-E75.md («E75b»).

    PYTHONPATH=codigo python codigo/e75b_cohesion_sin_alarmas.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params
from parametros import P0
from e64_persistencia import clase, condiciones, T_P, T_BURN, DT_S, SEM_P

RAIZ = os.environ.get("E75_RAIZ") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.environ.get("E75_SALIDA", "resultados")
NOMS = ("ViSt", "adaptado_pi0.05", "adaptado_pi0.173", "adaptado_pi0.3", "amplitud_pi0.173", "inst_m6_pi0.173")
REP, CADA = 0, 5.0


def componentes(f):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = np.array(list(cKDTree(f.pos).query_pairs(2.5 * f.p.d0)), dtype=int)
    n = f.p.N
    if not pairs.size:
        return n, 1.0 / n
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    return int(nc), float(np.bincount(lab).max() / n)


def brecha(f):
    """Arista más larga del árbol de expansión mínima (en unidades de d0) y tamaño del lado menor."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    from scipy.spatial import cKDTree
    n = f.p.N
    tr = cKDTree(f.pos)
    d, j = tr.query(f.pos, k=min(n, 40))
    i = np.repeat(np.arange(n), d.shape[1] - 1)
    g = coo_matrix((d[:, 1:].ravel(), (i, j[:, 1:].ravel())), shape=(n, n)).tocsr()
    g = g.maximum(g.T)
    nc, _ = connected_components(g, directed=False)
    if nc > 1:                                   # con 39 vecinos no basta: grafo completo
        D = np.linalg.norm(f.pos[:, None, :] - f.pos[None, :, :], axis=2)
        g = coo_matrix(D).tocsr()
    t = minimum_spanning_tree(g).tocoo()
    m = int(np.argmax(t.data))
    keep = np.ones(len(t.data), bool); keep[m] = False
    h = coo_matrix((t.data[keep], (t.row[keep], t.col[keep])), shape=(n, n))
    _, lab = connected_components(h, directed=False)
    return float(t.data[m] / f.p.d0), int(np.bincount(lab).min())


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "lambda_fp": 0.0, "tau_W": 4.0, "seed": SEM_P + REP})
    t0 = time.time()
    flocking.Flock = clase(a["factor"], a["tau_base"])
    try:
        f = flocking.Flock(p, a["model"])
    finally:
        flocking.Flock = Flock
    for _ in range(int(round(T_BURN / p.dt))):
        f.step()
    every = int(round(DT_S / p.dt))
    n_s = int(round(T_P / DT_S))
    V = f.V()
    prev = unw = float(np.arctan2(V[1], V[0]))
    phi = np.empty(n_s + 1)
    phi[0] = unw
    nc0, fr0 = componentes(f)
    t, frac, ncomp, pol = [0.0], [fr0], [nc0], [float(f.polarization())]
    b0 = brecha(f)
    dst, lado = [b0[0]], [b0[1]]
    m = int(round(CADA / DT_S))
    for s in range(n_s):
        for _ in range(every):
            f.step()
        V = f.V()
        a_ = float(np.arctan2(V[1], V[0]))
        unw += float(np.angle(np.exp(1j * (a_ - prev))))
        prev = a_
        phi[s + 1] = unw
        if s % m == m - 1:
            nc, fr = componentes(f)
            t.append((s + 1) * DT_S); frac.append(fr); ncomp.append(nc); pol.append(float(f.polarization()))
            bb = brecha(f)
            dst.append(bb[0]); lado.append(bb[1])
    return {"nom": a["nom"], "rep": REP, "phi": (phi - phi[0]).astype(np.float32).tolist(), "t": t, "frac": frac,
            "ncomp": ncomp, "pol": pol, "dstar": dst, "lado_menor": lado, "coh_final": float(f.cohesion()["frac_max"]), "seg": time.time() - t0}


if __name__ == "__main__":
    os.chdir(RAIZ)
    C = {c[0]: c for c in condiciones()}
    jobs = [{"nom": n, "model": C[n][1], "kw": C[n][2], "factor": C[n][3], "tau_base": C[n][4]} for n in NOMS]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("FASE4_W", "5"))) as ex:
        res = list(ex.map(job, jobs))
    e64 = json.load(open("resultados/E64_persistencia_crudo.json"))["P"]
    out = {"rep": REP, "cada_s": CADA, "cond": {}}
    L = ["=== E75b · cohesión en el tiempo, réplica 0 de E64 (sin informados ni falsas alarmas) ==="]
    for x in res:
        g = next(y for y in e64 if y["nom"] == x["nom"] and y["rep"] == REP)
        dphi = float(np.max(np.abs(np.asarray(x["phi"]) - np.asarray(g["phi"]))))
        fr, t = np.array(x["frac"]), np.array(x["t"])
        bajo = np.where(fr < 0.9)[0]
        q = {"max_dif_rumbo": dphi, "coh_final": x["coh_final"], "coh_guardada": g["coh"], "frac_min": float(fr.min()),
             "t_primera_bajo_0.9": float(t[bajo[0]]) if len(bajo) else None,
             "vuelve_a_1": bool(len(bajo) and (fr[bajo[0]:] >= 0.999).any()),
             "frac_tiempo_bajo_0.9": float((fr < 0.9).mean()), "ncomp_max": int(max(x["ncomp"])),
             "pol_media": float(np.mean(x["pol"])), "t": x["t"], "frac": x["frac"], "ncomp": x["ncomp"], "seg": x["seg"],
             "dstar": x["dstar"], "lado_menor": x["lado_menor"]}
        ds, ld = np.array(x["dstar"]), np.array(x["lado_menor"])
        tarde = t >= 350
        im = int(np.argmax(np.where(tarde, ds, -1)))
        q.update({"dstar_inicial": float(ds[0]), "dstar_max": float(ds.max()), "dstar_max_tras_350": float(ds[tarde].max()),
                  "t_dstar_max_tras_350": float(t[im]), "lado_menor_en_ese_max": int(ld[im]),
                  "dstar_mediana_antes_300": float(np.median(ds[t < 300])), "frac_tiempo_dstar_sobre_2": float((ds > 2.0).mean())})
        out["cond"][x["nom"]] = q
        L.append(f"{x['nom']:18s} máx|Δrumbo| {dphi:.1e}  cohesión final {x['coh_final']:.3f} (guardada {g['coh']:.3f})  "
                 f"mínima {q['frac_min']:.3f}  primera vez < 0,9: {q['t_primera_bajo_0.9']}  vuelve a 1: {q['vuelve_a_1']}  "
                 f"tiempo bajo 0,9: {q['frac_tiempo_bajo_0.9']:.2f}  componentes máx {q['ncomp_max']}  pol {q['pol_media']:.3f}  "
                 f"d*/d0 inicial {q['dstar_inicial']:.2f}, mediana antes de 300 s {q['dstar_mediana_antes_300']:.2f}, máx tras 350 s "
                 f"{q['dstar_max_tras_350']:.2f} (t = {q['t_dstar_max_tras_350']:g} s, lado menor {q['lado_menor_en_ese_max']})")
    part = [n for n in ("adaptado_pi0.173", "adaptado_pi0.3", "inst_m6_pi0.173") if out["cond"][n]["t_primera_bajo_0.9"] is not None]
    out["pred"] = {"F7": {
        "t_primera": {n: out["cond"][n]["t_primera_bajo_0.9"] for n in NOMS},
        "a": bool(len(part) == 3 and all(out["cond"][n]["t_primera_bajo_0.9"] < 60 and not out["cond"][n]["vuelve_a_1"] for n in part)),
        "b": bool(all(out["cond"][n]["frac_min"] >= 0.999 for n in ("ViSt", "adaptado_pi0.05", "amplitud_pi0.173")))}}
    out["pred"]["F7"]["cumple"] = bool(out["pred"]["F7"]["a"] and out["pred"]["F7"]["b"])
    v = out["cond"]["ViSt"]
    out["pred"]["F8"] = {"dstar_max_tras_350": v["dstar_max_tras_350"], "lado_menor": v["lado_menor_en_ese_max"],
                         "cumple": bool(v["dstar_max_tras_350"] > 2.0 and v["lado_menor_en_ese_max"] == 111)}
    L.append("F7: " + ("CUMPLE" if out["pred"]["F7"]["cumple"] else "FALLA") + "  " + json.dumps({k: v for k, v in out["pred"]["F7"].items()}))
    L.append("F8: " + ("CUMPLE" if out["pred"]["F8"]["cumple"] else "FALLA") + "  " + json.dumps(out["pred"]["F8"]))
    json.dump(out, open(os.path.join(SALIDA, "E75b_cohesion_sin_alarmas.json"), "w"), indent=1)
    open(os.path.join(SALIDA, "E75b_cohesion_sin_alarmas.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
