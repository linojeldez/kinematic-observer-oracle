"""Etapa 49 (C2 y C3 del plan de fortalecimiento): sensibilidad a k y al ruido, y un
observador que estima su propio suelo de ruido.

Por qué. Las cifras de recuperación del Paper A corto valen con k = 15 y un nivel de ruido,
y el observador adaptado usa una anchura del residuo calibrada fuera de la corrida (E22: al
trasplantar calibraciones baja de 72 % a 12-44 %). Aquí se mide (C2) la recuperación en una
malla k x ruido con N = 250 y tau_W = 1 s, y (C3) un observador que estima la anchura con
una línea base local y móvil (`fase4.FlockAuto`).

Cada celda de la malla se mide con tres observadores (sin memoria y adaptado de 1 s):
  fijo           sigma_theta = 0,361 rad/s (la calibración de las grajillas; la del paper)
  calibrado      sigma_theta = anchura robusta del residuo del motor en esa celda
  autocalibrado  FlockAuto, tau_base = 5 s (2 y 10 s como sensibilidad)

Diseño completo, anclas y predicciones S1 a S7 en docs/bitacora/resultados_E46-E55.md,
declaradas antes de correr.

    PYTHONPATH=codigo python codigo/e49_sensibilidad.py
    PYTHONPATH=codigo E49_RAPIDO=1 python codigo/e49_sensibilidad.py        # humo
    PYTHONPATH=codigo E49_SOLO_ANALISIS=1 python codigo/e49_sensibilidad.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

import fase4
from flocking import Flock, Params, phi_grid
from parametros import P0
from fase4 import (D_PHI, T_N, W, _w, trabajo, correr, clave, crudos_previos, pareado, fmt,
                   limpio, R_rep)

RAPIDO = bool(os.environ.get("E49_RAPIDO"))
SOLO_ANALISIS = bool(os.environ.get("E49_SOLO_ANALISIS"))
S2 = float(np.sqrt(2.0))
KS, FACS = (7, 15, 30), (0.5, 1.0, 2.0)
N_REP = 1 if RAPIDO else 8
N_SEM_ANCH = 2 if RAPIDO else 8
TAU_BASE = 5.0
SIGMA_FIJA = P0.sigma_theta
CABECERA = (("forzado", False, 4.0, "ViSt"), ("maniobra", True, 4.0, "CoLd"),
            ("maniobra", True, 1.0, "CoLd"))

SUF = "_rapido" if RAPIDO else ""
F_CRUDO = f"resultados/E49_sensibilidad_crudo{SUF}.json"
F_JSON = f"resultados/E49_sensibilidad{SUF}.json"
F_TXT = f"resultados/E49_sensibilidad{SUF}.txt"
F_PARCIAL = F_CRUDO + ".parcial"


def celda(k, f):
    return f"k{k}|f{f:g}"


BASE = celda(15, 1.0)


def _anchura(args):
    """Anchura robusta del residuo en vuelo sin líder (ViSt, N = 250): 3 s + 12 s."""
    k, f, seed = args
    p = Params(**{**asdict(P0), "N": 250, "k": k, "sigma_xi": f * S2, "seed": 100 + seed})
    fl = Flock(p, "ViSt")
    for _ in range(300):
        fl.step()
    out = []
    for s in range(1200):
        fl.step()
        if s % 5 == 0:
            idx, _ = fl._neighbors()
            out.append(np.abs(fl.omega - fl.omega[idx].mean(axis=1)))
    return k, f, float(1.4826 * np.median(np.concatenate(out))), float(fl.polarization())


def medir_anchuras():
    jobs = [(k, f, s) for k in KS for f in FACS for s in range(N_SEM_ANCH)]
    with ProcessPoolExecutor(max_workers=W) as ex:
        res = list(ex.map(_anchura, jobs))
    out = {}
    for k in KS:
        for f in FACS:
            v = np.array([w for kk, ff, w, _ in res if kk == k and ff == f])
            pol = np.array([q for kk, ff, _, q in res if kk == k and ff == f])
            out[celda(k, f)] = {"sigma": round(float(v.mean()), 4), "sd": float(v.std()),
                                "pol": float(pol.mean()), "semillas": len(v)}
    return out


def trabajos(anch):
    phis = phi_grid(D_PHI)
    phis_i = [(4, phis[4]), (13, phis[13])] if RAPIDO else list(enumerate(phis))
    out = []

    def add(c, nom, N, tau, pulso, base, kw, tau_base=None):
        for i, phi in phis_i:
            for rep in range(N_REP):
                out.append(trabajo(c, nom, N, tau, pulso, i, phi, rep, kw=kw, base=base,
                                   tau_base=tau_base))

    for k in KS:
        for f in FACS:
            c = celda(k, f)
            kw = dict(k=k, sigma_xi=f * S2)
            cal = dict(kw, sigma_theta=anch[c]["sigma"])
            add(c, "CoLd", 250, 1.0, True, "CoLd", kw)
            add(c, "oraculo", 250, 1.0, True, "oraculo", kw)
            for obs in ("amplitud", "adaptado"):
                add(c, f"{obs}_fijo", 250, 1.0, True, obs, kw)
                add(c, f"{obs}_cal", 250, 1.0, True, obs, cal)
                add(c, f"{obs}_auto", 250, 1.0, True, obs, kw, TAU_BASE)
            if k == 15 or f == 1.0:                       # cruz con líder forzado
                add(c, "ViSt", 250, 4.0, False, "ViSt", kw)
                add(c, "oraculo", 250, 4.0, False, "oraculo", kw)
                add(c, "amplitud_cal", 250, 4.0, False, "amplitud", cal)
                add(c, "adaptado_cal", 250, 4.0, False, "adaptado", cal)
                add(c, "adaptado_auto", 250, 4.0, False, "adaptado", kw, TAU_BASE)
    # celdas de cabecera con N = 250 (celda base de la malla)
    kw = dict(k=15, sigma_xi=S2)
    add(BASE, "adaptado_fijo", 250, 4.0, False, "adaptado", kw)
    for nom, b in (("CoLd", "CoLd"), ("oraculo", "oraculo"), ("adaptado_fijo", "adaptado"),
                   ("amplitud_fijo", "amplitud")):
        add(BASE, nom, 250, 4.0, True, b, kw)
    add(BASE, "adaptado_auto", 250, 4.0, True, "adaptado", kw, TAU_BASE)
    add(BASE, "amplitud_auto", 250, 4.0, True, "amplitud", kw, TAU_BASE)
    for tb in (2.0, 10.0):
        for _, pulso, tau, _ in CABECERA:
            add(BASE, f"adaptado_auto{tb:g}", 250, tau, pulso, "adaptado", kw, tb)
    if not RAPIDO:                                        # N = 1000, referencias guardadas
        for _, pulso, tau, _ in CABECERA:
            add("base", "adaptado_auto", 1000, tau, pulso, "adaptado", {}, TAU_BASE)
    out.sort(key=lambda a: a["N"] * a["T"])
    return out


def cociente_ic(res, c, num, den, ref, N, pulso, win, tau, c_ref=None, den_c=None,
                n_boot=4000, seed=43):
    """Cociente de recuperaciones num/den con IC 95 % remuestreando réplicas pareadas."""
    cr = c_ref or c
    cd = den_c or c
    series = [R_rep(res, c, num, N, pulso, win, tau), R_rep(res, cd, den, N, pulso, win, tau),
              R_rep(res, cr, ref, N, pulso, win, tau)]
    com = sorted(set(series[0][0]) & set(series[1][0]) & set(series[2][0]))
    if not com:
        return {"cociente": float("nan"), "ic95": [float("nan")] * 2, "n_rep": 0}
    A, B, V = (np.array([R[ids.index(k)] for k in com]) for ids, R in series)
    q = (A.mean() - V.mean()) / (B.mean() - V.mean())
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(com), (n_boot, len(com)))
    qs = (A[b].mean(1) - V[b].mean(1)) / (B[b].mean(1) - V[b].mean(1))
    lo, hi = np.percentile(qs, [2.5, 97.5])
    return {"cociente": float(q), "ic95": [float(lo), float(hi)], "n_rep": len(com)}


def analizar(res, anch):
    prev = crudos_previos()
    idx_prev = {clave(x): x for x in prev}
    L = ["=== E49 · sensibilidad a k y al ruido; observador autocalibrado ==="]
    P = L.append
    out = {"config": {"N": 250, "k": list(KS), "ruido_factor": list(FACS), "replicas": N_REP,
                      "tau_base": TAU_BASE, "sigma_fija": SIGMA_FIJA, "rapido": RAPIDO},
           "anchuras": anch, "ancla": {}, "malla": {}, "cruz": {}, "cabecera": {},
           "sigma_auto": {}, "pred": {}}

    P("\n=== anchura robusta del residuo del motor (ViSt, N = 250), rad/s ===")
    for c, v in anch.items():
        P(f'{c:10s} {v["sigma"]:.4f} ± {v["sd"]:.4f}  polarización {v["pol"]:.4f}')

    # ---- anclas: la celda base reproduce los crudos guardados
    dm, n_cmp = 0.0, 0
    for x in res:
        if x["celda"] != BASE or x["N"] != 250:
            continue
        nom = {"CoLd": "CoLd", "oraculo": "oraculo", "amplitud_fijo": "amplitud",
               "adaptado_fijo": "adaptado"}.get(x["nom"])
        if nom is None:
            continue
        y = idx_prev.get(clave({**x, "celda": "base", "nom": nom}))
        if y is None:
            continue
        for w in x["r"]:
            if w in y["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w]))
                n_cmp += 1
    out["ancla"] = {"delta_max": dm, "comparaciones": n_cmp,
                    "cumple": bool(n_cmp > 0 and dm < 1e-12)}
    P(f"\nancla: celda base frente a los crudos de E42, E43 y E44b: |Δ| máx = {dm:.2e} en "
      f"{n_cmp} comparaciones")

    OBS = ("amplitud_fijo", "amplitud_cal", "amplitud_auto", "adaptado_fijo", "adaptado_cal",
           "adaptado_auto")
    win = T_N[250]
    P("\n=== malla: maniobra única, tau_W = 1 s, corrida entera; recuperación desde CoLd ===")
    for k in KS:
        for f in FACS:
            c = celda(k, f)
            fila = {o: limpio(pareado(res, c, o, "CoLd", 250, True, win, 1.0)) for o in OBS}
            fila["cociente_auto_cal"] = cociente_ic(res, c, "adaptado_auto", "adaptado_cal",
                                                    "CoLd", 250, True, win, 1.0)
            out["malla"][c] = fila
            q = fila["adaptado_cal"]
            P(f'\n{c}: ventaja del oráculo sobre CoLd {q.get("ventaja", float("nan")):.4f} '
              f'{q.get("ventaja_ic95")}')
            for o in OBS:
                P(f'   {o:14s} {fmt(fila[o])}')
            P(f'   autocalibrado / calibrado (adaptado): {fila["cociente_auto_cal"]["cociente"]:.3f} '
              f'{fila["cociente_auto_cal"]["ic95"]}')

    P("\n=== cruz con líder forzado, corrida entera; recuperación desde ViSt ===")
    for k in KS:
        for f in FACS:
            if not (k == 15 or f == 1.0):
                continue
            c = celda(k, f)
            fila = {o: limpio(pareado(res, c, o, "ViSt", 250, False, win, 4.0))
                    for o in ("amplitud_cal", "adaptado_cal", "adaptado_auto")}
            out["cruz"][c] = fila
            P(f'{c}: ventaja {fila["adaptado_cal"].get("ventaja", float("nan")):.4f} | sin memoria cal. '
              f'{fmt(fila["amplitud_cal"])} | adaptado cal. {fmt(fila["adaptado_cal"])} | '
              f'adaptado auto {fmt(fila["adaptado_auto"])}')

    P("\n=== C3 en las celdas de cabecera: autocalibrado frente a fijo ===")
    todo = prev + res
    for N in (250, 1000):
        for etq, pulso, tau, ref in CABECERA:
            w_ = T_N[N]
            if N == 250:
                c, fijo = BASE, "adaptado_fijo"
            else:
                c, fijo = "base", "adaptado"
            fila = {"fijo": limpio(pareado(todo, c, fijo, ref, N, pulso, w_, tau)),
                    "auto": limpio(pareado(todo, c, "adaptado_auto", ref, N, pulso, w_, tau)),
                    "cociente": cociente_ic(todo, c, "adaptado_auto", fijo, ref, N, pulso, w_, tau)}
            if N == 250:
                for tb in ("2", "10"):
                    fila[f"auto{tb}"] = limpio(pareado(todo, c, f"adaptado_auto{tb}", ref, N,
                                                       pulso, w_, tau))
            out["cabecera"][f"{etq}|{N}|{tau:g}"] = fila
            P(f'{etq:9s} N = {N:4d} tau_W = {tau:g} s | fijo {fmt(fila["fijo"])} | auto (5 s) '
              f'{fmt(fila["auto"])} | cociente {fila["cociente"]["cociente"]:.3f} '
              f'{fila["cociente"]["ic95"]}'
              + (f' | auto 2 s {fila["auto2"]["rec"]:.3f} | auto 10 s {fila["auto10"]["rec"]:.3f}'
                 if N == 250 else ""))

    P("\n=== anchura estimada por el autocalibrado (adaptado, tau_base = 5 s) ===")
    for k in KS:
        for f in FACS:
            c = celda(k, f)
            v = [(x["sig_ini"], x["sig_max"]) for x in res
                 if x["celda"] == c and x["nom"] == "adaptado_auto" and x["pulso"]
                 and abs(x["tau_W"] - 1.0) < 1e-9 and x.get("sig_ini")]
            if not v:
                continue
            ini = float(np.mean([a for a, _ in v]))
            mx = float(np.mean([b for _, b in v]))
            out["sigma_auto"][c] = {"al_informarse": ini, "maximo": mx,
                                    "sobre_medida": ini / anch[c]["sigma"],
                                    "maximo_sobre_inicial": mx / ini}
            P(f'{c:10s} al informarse el líder {ini:.4f} ({ini / anch[c]["sigma"]:.3f} de la '
              f'medida); máximo durante la corrida {mx:.4f} ({mx / ini:.3f} del inicial)')

    # ---- predicciones
    pr = out["pred"]
    m = out["malla"]
    try:
        pr["S1"] = {"sin_memoria_cal": {c: m[c]["amplitud_cal"]["rec"] for c in m},
                    "adaptado_cal": {c: m[c]["adaptado_cal"]["rec"] for c in m},
                    "fallan": [c for c in m if not (m[c]["amplitud_cal"]["rec"] < 0.15
                                                    and m[c]["adaptado_cal"]["rec"] > 0.90)]}
        pr["S1"]["cumple"] = not pr["S1"]["fallan"]
        s2 = {c: m[c]["adaptado_fijo"]["rec"] for c in m}
        pr["S2"] = {"adaptado_fijo": s2, "cumple": bool(
            all(s2[celda(k, 0.5)] > 0.90 for k in KS) and all(s2[celda(k, 2.0)] < 0.50 for k in KS)
            and all(abs(s2[celda(k, 1.0)] - m[celda(k, 1.0)]["adaptado_cal"]["rec"]) < 0.10
                    for k in (7, 30)))}
        cab = out["cabecera"]
        s3 = {k: cab[k]["cociente"]["cociente"] for k in cab if "|250|" in k}
        pr["S3"] = {"cociente_auto_fijo": s3, "cumple": bool(all(v >= 2 / 3 for v in s3.values()))}
        s4 = {c: m[c]["cociente_auto_cal"]["cociente"] for c in m}
        s4b = {c: m[c]["amplitud_auto"]["rec"] for c in m}
        pr["S4"] = {"cociente_auto_cal": s4, "sin_memoria_auto": s4b,
                    "cumple": bool(all(v >= 2 / 3 for v in s4.values())
                                   and all(v < 0.15 for v in s4b.values()))}
        s5 = {k: cab[k]["cociente"]["cociente"] for k in cab if "|1000|" in k}
        pr["S5"] = {"cociente_auto_fijo": s5,
                    "cumple": bool(s5 and all(v >= 2 / 3 for v in s5.values()))}
        s6 = {k: [cab[k]["auto2"]["rec"] - cab[k]["auto"]["rec"],
                  cab[k]["auto10"]["rec"] - cab[k]["auto"]["rec"]] for k in cab if "|250|" in k}
        pr["S6"] = {"diferencia_2s_10s": s6,
                    "cumple": bool(all(abs(x) < 0.10 for v in s6.values() for x in v))}
        cz = out["cruz"]
        pr["S7"] = {"sin_memoria_cal": {c: cz[c]["amplitud_cal"]["rec"] for c in cz},
                    "adaptado_cal": {c: cz[c]["adaptado_cal"]["rec"] for c in cz},
                    "cumple": bool(all(cz[c]["amplitud_cal"]["rec"] < 0.15
                                       and cz[c]["adaptado_cal"]["rec"] > 0.50 for c in cz))}
    except KeyError as e:
        pr["incompleto"] = str(e)
    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        if isinstance(v, dict):
            P(f'{k}: {"CUMPLE" if v.get("cumple") else "FALLA"}  {json.dumps(v, ensure_ascii=False)[:700]}')
        else:
            P(f"{k}: {v}")
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if RAPIDO:
        fase4.N_PHI = 2
    f_anch = F_JSON.replace(".json", "_anchuras.json")
    if SOLO_ANALISIS:
        with open(f_anch) as fh:
            anch = json.load(fh)
        with open(F_CRUDO if os.path.exists(F_CRUDO) else F_PARCIAL) as fh:
            d = json.load(fh)
        res, seg = (d["resp"] if isinstance(d, dict) else d), 0.0
    else:
        if os.path.exists(f_anch):                       # reanudación: mismas calibraciones
            with open(f_anch) as fh:
                anch = json.load(fh)
        else:
            anch = medir_anchuras()
            with open(f_anch, "w") as fh:
                json.dump(anch, fh, indent=1)
        res, seg = correr(trabajos(anch), F_PARCIAL, "E49")
        with open(F_CRUDO, "w") as fh:
            json.dump({"resp": res}, fh)
        if os.path.exists(F_PARCIAL):
            os.remove(F_PARCIAL)
    out, L = analizar(res, anch)
    L.append(f"\n{seg:.0f} s")
    out["segundos"] = seg
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
