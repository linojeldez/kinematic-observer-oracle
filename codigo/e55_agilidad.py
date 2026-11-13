"""Etapa 55 (plan 13): intervención sobre la agilidad en el modelo (H3).

a_max en {5, 10, 20, 40} m/s^2 para todos los agentes; barrido de la ventana del observador
(0,25 a 4 s) y huella. Diseño, ancla y predicciones en docs/bitacora/resultados_E46-E55.md.

    PYTHONPATH=codigo python codigo/e55_agilidad.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from flocking import Flock, Params, phi_grid
from parametros import P0
from fase4 import (D_PHI, T_N, W, trabajo, correr, clave, crudos_previos, pareado, limpio, fmt)

AS = (5.0, 10.0, 20.0, 40.0)
VENT = (0.25, 0.5, 1.0, 2.0, 4.0)
N_REP, TAU_W = 4, 4.0
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))
F_CRUDO, F_JSON, F_TXT = ("resultados/E55_agilidad_crudo.json", "resultados/E55_agilidad.json",
                          "resultados/E55_agilidad.txt")
NOM_PREV = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s", 4.0: "ciclo 4 s"}


def _anchura(args):
    a, seed = args
    p = Params(**{**asdict(P0), "N": 250, "a_max": a, "seed": 100 + seed})
    f = Flock(p, "ViSt")
    for _ in range(300):
        f.step()
    out = []
    for s in range(1200):
        f.step()
        if s % 5 == 0:
            idx, _ = f._neighbors()
            out.append(np.abs(f.omega - f.omega[idx].mean(axis=1)))
    return a, float(1.4826 * np.median(np.concatenate(out)))


def _huella(args):
    """Operador de e15_helpers y T95 de E51 para el agente con intención (N = 200, k = 12)."""
    m, a, rep = args
    CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
    q = Params(N=200, k=12, sigma_xi=float(np.sqrt(2.0)), seed=11 + rep, a_max=a, **CFG)
    f = Flock(q, m)
    for _ in range(600):
        f.step()
    base = []
    for _ in range(200):
        f.step()
        i, _ = f._neighbors()
        base.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    sig = float(np.median(np.array(base)) * 1.4826)
    ag = 7
    f.will[ag] = 1.0
    f.d_des[ag] = np.array([np.cos(2.0), np.sin(2.0)])
    serie = []
    for s in range(500):
        f.step()
        f.will[ag] = 1.0
        if s % 5 == 0:
            i, _ = f._neighbors()
            serie.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    S = np.array(serie) / sig
    t = np.arange(len(S)) * 0.05
    w = 4
    sm = np.apply_along_axis(lambda x: np.convolve(x, np.ones(w) / w, mode="same"), 0, S)
    ip = int(np.argmax(sm[:, ag]))
    dur = float(t[-1])
    for j in range(ip, len(sm)):
        if sm[j, ag] < 1.5 and np.all(sm[j:j + w, ag] < 1.5):
            dur = float(t[j]); break
    otros = np.array([j for j in range(q.N) if j != ag])
    thr = np.percentile(sm[:40][:, otros], 95)
    return {"modelo": m, "a_max": a, "rep": rep, "dur": dur, "T95": float((sm[:40, ag] > thr).sum() * 0.05),
            "pico": float(sm[:, ag].max())}


def trabajos(anch):
    phis = phi_grid(D_PHI)
    out = []
    for a in AS:
        if a == 10.0:
            cal = {}
        else:
            cal = dict(a_max=a, sigma_theta=anch[str(a)], sigma_W=0.811 * a / 10.0)
        for etq, pulso, ref in PROT:
            mecs = [(ref, ref, {}), ("oraculo", "oraculo", {}), ("amplitud", "amplitud", {})]
            mecs += [(f"ciclo{v:g}", "adaptado", {"tau_det": v}) for v in VENT]
            for nom, base, kw in mecs:
                kw2 = {**kw, **({"a_max": a} if base in ("ViSt", "CoLd", "oraculo") and a != 10.0 else {}),
                       **(cal if base in ("amplitud", "adaptado") else {})}
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        out.append(trabajo(f"a{a:g}", nom, 250, TAU_W, pulso, i, phi, rep, kw=kw2, base=base))
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with ProcessPoolExecutor(max_workers=W) as ex:
        an = list(ex.map(_anchura, [(a, s) for a in AS for s in range(8)]))
        hu = list(ex.map(_huella, [(m, a, r) for m in ("WiSt", "CoLd") for a in AS for r in range(16)]))
    anch = {str(a): round(float(np.mean([w for aa, w in an if aa == a])), 4) for a in AS}
    res, seg = correr(trabajos(anch), F_CRUDO + ".parcial", "E55")
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")

    L = ["=== E55 · agilidad intervenida en el modelo (N = 250, k = 15, tau_W = 4 s, 4 réplicas) ==="]
    out = {"anchuras": anch, "huella": {}, "rec": {}, "mejor": {}}
    # ---- ancla a_max = 10 frente a E42, E42b y E44b
    prev = {}
    for x in crudos_previos():
        prev[(x["nom"], x["N"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"])] = x
    for f in ("E42b_barrido_ventana",):
        for x in json.load(open(f"resultados/{f}_crudo.json"))["resp"]:
            prev[(x["nom"], x["N"], x["tau_W"], x["pulso"], x["i_phi"], x["rep"])] = x
    dm, n = 0.0, 0
    mapa = {**{f"ciclo{v:g}": NOM_PREV[v] for v in VENT}, "ViSt": "ViSt", "CoLd": "CoLd",
            "oraculo": "oraculo", "amplitud": "amplitud"}
    for x in res:
        if x["celda"] != "a10":
            continue
        y = prev.get((mapa[x["nom"]], 250, TAU_W, x["pulso"], x["i_phi"], x["rep"]))
        if y is None:
            continue
        for w in x["r"]:
            if w in y["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    out["ancla"] = {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)}
    L.append(f"ancla a_max = 10 frente a E42, E42b y E44b: |Δ| máx = {dm:.2e} en {n} comparaciones")
    L.append("anchura del residuo (rad/s): " + ", ".join(f"a_max {k}: {v}" for k, v in anch.items()))
    for m in ("WiSt", "CoLd"):
        for a in AS:
            v = [h for h in hu if h["modelo"] == m and h["a_max"] == a]
            r = {k: float(np.median([h[k] for h in v])) for k in ("dur", "T95", "pico")}
            out["huella"][f"{m}|{a:g}"] = r
            L.append(f"huella {m} a_max {a:4.0f}: operador {r['dur']:.2f} s, T95 {r['T95']:.2f} s, pico {r['pico']:.1f} σ")
    for a in AS:
        for etq, pulso, ref in PROT:
            fila = {}
            for nom in ["amplitud"] + [f"ciclo{v:g}" for v in VENT]:
                fila[nom] = limpio(pareado(res, f"a{a:g}", nom, ref, 250, pulso, 15.8, TAU_W))
            vs = {v: fila[f"ciclo{v:g}"]["rec"] for v in VENT}
            mejor = max(vs, key=vs.get)
            out["rec"][f"{etq}|{a:g}"] = fila
            out["mejor"][f"{etq}|{a:g}"] = {"ventana": mejor, "rec": vs[mejor]}
            L.append(f"{etq:9s} a_max {a:4.0f}: ventaja {fila['ciclo1']['ventaja']:.3f} | sin memoria "
                     f"{fila['amplitud']['rec']:.3f} | " + " | ".join(f"{v:g} s {vs[v]:.3f}" for v in VENT)
                     + f" | mejor {mejor:g} s")
    pr = {}
    for m in ("WiSt", "CoLd"):
        q = out["huella"][f"{m}|5"]["dur"] / max(out["huella"][f"{m}|40"]["dur"], 1e-9)
        pr[f"P1_{m}"] = {"cociente_5_40": q, "cumple": bool(2 <= q <= 5)}
    for etq, _, _ in PROT:
        seq = [out["mejor"][f"{etq}|{a:g}"]["ventana"] for a in AS]
        q = seq[0] / seq[-1]
        pr[f"P2_{etq}"] = {"ventanas": seq, "cociente_5_40": q,
                           "cumple": bool(all(seq[i] >= seq[i + 1] for i in range(3)) and 2 <= q <= 8)}
        pr[f"P3_{etq}"] = {"rec_5": out["mejor"][f"{etq}|5"]["rec"], "rec_10": out["mejor"][f"{etq}|10"]["rec"],
                           "cumple": bool(out["mejor"][f"{etq}|5"]["rec"] < out["mejor"][f"{etq}|10"]["rec"])}
        pr[f"regla_{etq}"] = {"cambia_mas_de_x2": bool(q > 2 or q < 0.5)}
    out["pred"] = pr
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v.get('cumple') else ('—' if 'cumple' not in v else 'FALLA')}  {json.dumps(v)}" for k, v in pr.items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, ensure_ascii=False, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
