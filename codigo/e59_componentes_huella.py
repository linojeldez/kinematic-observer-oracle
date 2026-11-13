"""Etapa 59 (plan 14, objeción mayor 3): componentes de la huella.

Mismo protocolo y semillas que E47 (`e15_helpers.huella`), registrando además el giro absoluto
del agente con intención, el de sus vecinos, su rapidez y su desalineación con la dirección
deseada. Diseño y predicciones D1-D3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e59_componentes_huella.py
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from flocking import Flock, Params

CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
BASE = dict(N=200, k=12, sigma_xi=np.sqrt(2.0), seed=11)
MODELOS = ("CoLd", "WiSt")
ANGULOS = (1.0, 2.0, 2.67, 3.0)
N_REP, AG = 16, 7
D_STAR = np.deg2rad(14.4)
F_JSON, F_TXT = "resultados/E59_componentes_huella.json", "resultados/E59_componentes_huella.txt"


def descenso(sm, thr, w, desde):
    for j in range(desde, len(sm)):
        if sm[j] < thr and np.all(sm[j:j + w] < thr):
            return j
    return None


def uno(a):
    m, ang, rep = a
    q = Params(**{**BASE, **CFG, "seed": BASE["seed"] + rep})
    f = Flock(q, m)
    for _ in range(int(round(6.0 / q.dt))):
        f.step()
    base = []
    for _ in range(int(round(2.0 / q.dt))):
        f.step()
        i, _ = f._neighbors()
        base.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    sig = float(np.median(np.concatenate(base)) * 1.4826)
    thr = 1.5 * max(sig, 1e-9)
    d_des = np.array([np.cos(ang), np.sin(ang)])
    f.will[AG] = 1.0
    f.d_des[AG] = d_des
    t, eps, omL, omV, vel, des = [], [], [], [], [], []
    for s in range(int(round(5.0 / q.dt))):
        f.step()
        f.will[AG] = 1.0
        if s % 5 == 0:
            i, _ = f._neighbors()
            e = np.abs(f.omega - f.omega[i].mean(axis=1))
            vec = [x for x in i[AG] if x != AG]
            t.append(s * q.dt)
            eps.append(float(e[AG]))
            omL.append(abs(float(f.omega[AG])))
            omV.append(abs(float(f.omega[vec].mean())))
            v = f.vel[AG]
            vel.append(float(np.linalg.norm(v)) / q.v0)
            des.append(float(np.arccos(np.clip(v @ d_des / max(np.linalg.norm(v), 1e-9), -1, 1))))
    t, eps, omL, omV, vel, des = map(np.array, (t, eps, omL, omV, vel, des))
    w = max(3, len(t) // 25)
    sm = lambda x: np.convolve(x, np.ones(w) / w, mode="same")
    se, sL, sV = sm(eps), sm(omL), sm(omV)
    ip = int(np.argmax(se))
    j = descenso(se, thr, w, ip)
    t_hu = float(t[j]) if j is not None else float(t[-1])
    jl = int(np.argmax(se >= thr)) if (se >= thr).any() else None
    t_lat = float(t[jl]) if jl is not None else float("nan")
    ipl = int(np.argmax(sL))
    jp = descenso(sL, thr, w, ipl)
    t_propio = float(t[jp]) if jp is not None else float(t[-1])
    ja = np.flatnonzero(des < D_STAR)
    t_alin = float(t[ja[0]]) if len(ja) else float("nan")
    k_hu = j if j is not None else len(t) - 1
    clase = "propio" if sL[k_hu] < thr else "social"
    n3 = int(round(3.0 / (5 * q.dt)))
    return {"modelo": m, "ang": ang, "rep": rep, "t_hu": t_hu, "t_lat": t_lat, "t_propio": t_propio,
            "t_alin": t_alin, "clase": clase, "v_min": float(vel[:n3].min()),
            "omL_en_hu": float(sL[k_hu] / thr), "omV_en_hu": float(sV[k_hu] / thr),
            "serie": {"t": t[::2].tolist(), "eps": (se[::2] / thr).tolist(), "omL": (sL[::2] / thr).tolist(),
                      "omV": (sV[::2] / thr).tolist(), "v": vel[::2].tolist()}}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with Pool(8) as pool:
        res = pool.map(uno, [(m, a, r) for m in MODELOS for a in ANGULOS for r in range(N_REP)], chunksize=2)
    e47 = json.load(open("resultados/E47_huella_modelos.json"))["amplitud"]
    L = ["=== E59 · componentes de la huella (N = 200, k = 12, 16 réplicas; semillas de E47) ==="]
    out = {"por_celda": {}, "ancla": {}}
    dm = 0.0
    for m in MODELOS:
        for a in ANGULOS:
            g = f"{np.rad2deg(a):.0f}"
            v = [x for x in res if x["modelo"] == m and x["ang"] == a]
            med = lambda k: float(np.nanmedian([x[k] for x in v]))
            c = {k: med(k) for k in ("t_hu", "t_lat", "t_propio", "t_alin", "v_min", "omL_en_hu", "omV_en_hu")}
            c["n_propio"] = int(sum(x["clase"] == "propio" for x in v))
            c["n"] = len(v)
            c["v_min_rango"] = [float(min(x["v_min"] for x in v)), float(max(x["v_min"] for x in v))]
            out["por_celda"][f"{m}|{g}"] = c
            d = abs(c["t_hu"] - e47[m][g]["mediana"])
            dm = max(dm, d)
            L.append(f"{m} {g:>3s}°: huella {c['t_hu']:.3f} s (E47 {e47[m][g]['mediana']:.3f}) | latencia {c['t_lat']:.2f} s | "
                     f"giro propio bajo umbral {c['t_propio']:.2f} s | alineado (<14,4°) {c['t_alin']:.2f} s | "
                     f"v_min {c['v_min']:.2f} v0 | en la huella: |ω_L| {c['omL_en_hu']:.2f} y |ω_vec| {c['omV_en_hu']:.2f} umbrales | "
                     f"«propio» {c['n_propio']}/{c['n']}")
    out["ancla"] = {"delta_max_medianas": dm, "cumple": bool(dm < 1e-9)}
    P = out["por_celda"]
    out["pred"] = {
        "D1": out["ancla"],
        "D2": {"CoLd_propio": [P[f"CoLd|{g}"]["n_propio"] for g in ("57", "115", "153", "172")],
               "WiSt_social": [P[f"WiSt|{g}"]["n"] - P[f"WiSt|{g}"]["n_propio"] for g in ("115", "153", "172")],
               "cumple": bool(all(P[f"CoLd|{g}"]["n_propio"] >= 12 for g in ("57", "115", "153", "172"))
                              and all(P[f"WiSt|{g}"]["n"] - P[f"WiSt|{g}"]["n_propio"] >= 9 for g in ("115", "153", "172")))},
        "D3": {"v_min": {k: v["v_min"] for k, v in P.items()}, "t_lat_172": [P["CoLd|172"]["t_lat"], P["WiSt|172"]["t_lat"]],
               "cumple": bool(all(P[f"{m}|{g}"]["v_min"] < 0.5 for m in MODELOS for g in ("153", "172"))
                              and all(P[f"{m}|57"]["v_min"] > 0.8 for m in MODELOS)
                              and all(P[f"{m}|172"]["t_lat"] < 1.0 for m in MODELOS))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    out["series_mediana"] = {}
    for m in MODELOS:
        for a in ANGULOS:
            v = [x["serie"] for x in res if x["modelo"] == m and x["ang"] == a]
            out["series_mediana"][f"{m}|{np.rad2deg(a):.0f}"] = {
                "t": v[0]["t"], **{k: np.median(np.array([s[k] for s in v]), axis=0).tolist() for k in ("eps", "omL", "omV", "v")}}
    out["crudo"] = [{k: x[k] for k in x if k != "serie"} for x in res]
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
