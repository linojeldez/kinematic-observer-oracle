"""Etapa 42: la escalera de recuperación frente al tamaño de la bandada.

Por qué. El número de cabecera del Paper A (el detector adaptado recupera 72 % de la
ventaja del oráculo; 60 % con una sola maniobra) se midió con N = 250. E41 encontró, sin
haberlo predicho y con 4 réplicas, que con N = 1000 baja a 0,38 (líder forzado) y 0,46
(una maniobra). Antes de fijar el número del paper corto hay que medirlo con 8 réplicas
en tres tamaños y saber si la ventana de 1 s sigue siendo la mejor en una bandada grande.

Diseño (banco de `run_response`, el de E20 y E41; nada del motor cambia):
  - N en {250, 500, 1000}; lambda_fp = 0, tau_W = 4 s, k = 15, sigma_xi^2 = 2.
  - Mecanismos: ViSt, oráculo, amplitud (sin memoria) y adaptado (ciclo de 1 s).
  - Protocolos: líder forzado (tractor) y maniobra única (pulso).
  - Rejilla de 10 grados en puntos medios (18 ángulos); 8 réplicas; semilla
    7 + 1000 * i_phi + réplica (la de E20 y E41: mismas corridas, anclas exactas).
  - Cada corrida dura sqrt(N) s (15,8; 22,4; 31,6). Responsividad en ventanas de 4, 8,
    15,8 s y sqrt(N) s sobre la misma corrida, y el curso de la proyección cada 0,5 s.
  - Barrido de ventana del detector: ciclo de 0,5 s y de 2 s con N = 250 y N = 1000,
    4 réplicas, los dos protocolos.
  - Recuperación = (R - R_ViSt) / (R_oráculo - R_ViSt); IC 95 % remuestreando réplicas,
    emparejadas entre mecanismos por semilla (4000 remuestreos).

Las predicciones se declaran en docs/bitacora/resultados_E36-E45.md antes de correr.

    PYTHONPATH=codigo python codigo/e42_escalera_N.py
    PYTHONPATH=codigo E42_RAPIDO=1 python codigo/e42_escalera_N.py        # humo
    PYTHONPATH=codigo E42_SOLO_ANALISIS=1 python codigo/e42_escalera_N.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from flocking import Params, run_response, phi_grid
from parametros import P0, MECANISMOS

RAPIDO = bool(os.environ.get("E42_RAPIDO"))
SOLO_ANALISIS = bool(os.environ.get("E42_SOLO_ANALISIS"))
W = int(os.environ.get("E42_W", "8"))

SEMILLA_R = 7
D_PHI = 10.0
TAU_W = 4.0
NS = (250, 500, 1000)
T_N = {250: 15.8, 500: 22.4, 1000: 31.6}
WINS = (4.0, 8.0, 15.8, 22.4, 31.6)
N_REP = 1 if RAPIDO else 8
N_REP_BARRIDO = 1 if RAPIDO else 4
N_BARRIDO = (250, 1000)
T_MUESTRA = 0.5

MEC = {n: (m, kw) for n, m, kw in MECANISMOS if n != "canal"}
BARRIDO = {
    "ciclo 0.5 s": ("WiSt", dict(will_mode="bayes", detector="ciclo", tau_det=0.5)),
    "ciclo 2 s": ("WiSt", dict(will_mode="bayes", detector="ciclo", tau_det=2.0)),
}
TODOS = {**MEC, **BARRIDO}

SUF = "_rapido" if RAPIDO else ""
F_CRUDO = f"resultados/E42_escalera_N_crudo{SUF}.json"
F_JSON = f"resultados/E42_escalera_N{SUF}.json"
F_TXT = f"resultados/E42_escalera_N{SUF}.txt"
F_PARCIAL = F_CRUDO + ".parcial"


def _w(t):
    return f"{t:g}"


def job(args):
    """Una corrida de respuesta. Devuelve r por ventana y el curso de la proyección."""
    nom, model, kw, N, tau_W, pulso, i_phi, phi, rep, T = args
    p = Params(**{**asdict(P0), **kw, "N": N, "lambda_fp": 0.0, "tau_W": tau_W,
                  "seed": SEMILLA_R + 1000 * i_phi + rep})
    t0 = time.time()
    r = run_response(p, model, float(phi), T, every_coh=100, pulse=pulso)
    proj = r["proj"]
    m = int(round(T_MUESTRA / p.dt))
    n = (len(proj) // m) * m
    return {"nom": nom, "N": N, "tau_W": tau_W, "pulso": bool(pulso), "i_phi": i_phi,
            "rep": rep, "pol": r["pol"], "frac_min": r["frac_min"],
            "r": {_w(w): float(proj[:int(round(w / p.dt))].mean())
                  for w in WINS if w <= T + 1e-9},
            "curso": [float(x) for x in proj[:n].reshape(-1, m).mean(axis=1)],
            "seg": time.time() - t0}


def clave(a):
    nom, model, kw, N, tau_W, pulso, i_phi, phi, rep, T = a
    return f"{nom}|{N}|{tau_W:g}|{int(pulso)}|{i_phi}|{rep}"


def clave_res(x):
    return f'{x["nom"]}|{x["N"]}|{x["tau_W"]:g}|{int(x["pulso"])}|{x["i_phi"]}|{x["rep"]}'


def trabajos():
    phis = phi_grid(D_PHI)
    if RAPIDO:
        phis_i = [(0, phis[0]), (9, phis[9])]
    else:
        phis_i = list(enumerate(phis))
    out = []
    for N in NS:
        for nom, (model, kw) in MEC.items():
            for pulso in (False, True):
                for i, phi in phis_i:
                    for rep in range(N_REP):
                        out.append((nom, model, kw, N, TAU_W, pulso, i, float(phi), rep, T_N[N]))
    for N in N_BARRIDO:
        for nom, (model, kw) in BARRIDO.items():
            for pulso in (False, True):
                for i, phi in phis_i:
                    for rep in range(N_REP_BARRIDO):
                        out.append((nom, model, kw, N, TAU_W, pulso, i, float(phi), rep, T_N[N]))
    out.sort(key=lambda a: -a[3] * a[9])          # los largos primero
    return out


def correr(jobs, f_parcial, etiqueta):
    """Corre los trabajos con reanudación: guarda parciales y se salta lo ya hecho."""
    hechos = {}
    if os.path.exists(f_parcial):
        with open(f_parcial) as fh:
            for x in json.load(fh):
                hechos[clave_res(x)] = x
        print(f"[{etiqueta}] reanudando: {len(hechos)} trabajos ya hechos", flush=True)
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[{etiqueta}] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    if faltan:
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(job, a) for a in faltan]
            for i, fu in enumerate(as_completed(futs), 1):
                x = fu.result()
                hechos[clave_res(x)] = x
                if time.time() - ult > 120 or i == len(futs):
                    with open(f_parcial, "w") as fh:
                        json.dump(list(hechos.values()), fh)
                    ult = time.time()
                    print(f"[{etiqueta}] {i}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    return [hechos[clave(a)] for a in jobs], time.time() - t0


# ----------------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------------


def R_rep(res, nom, N, pulso, win, tau_W=TAU_W):
    """R por réplica (media sobre ángulos). Array (n_rep,), NaN si falta."""
    d = {}
    for x in res:
        if (x["nom"] == nom and x["N"] == N and x["pulso"] == pulso
                and abs(x["tau_W"] - tau_W) < 1e-9 and _w(win) in x["r"]):
            d.setdefault(x["rep"], []).append(x["r"][_w(win)])
    if not d:
        return np.array([])
    return np.array([float(np.mean(d[k])) for k in sorted(d)])


def rec_ic(Rm, Rv, Ro, n_boot=4000, seed=42):
    """Recuperación y su IC 95 % remuestreando réplicas emparejadas."""
    n = min(len(Rm), len(Rv), len(Ro))
    if n == 0:
        return {"rec": float("nan"), "ic95": [float("nan")] * 2, "n_rep": 0}
    Rm, Rv, Ro = Rm[:n], Rv[:n], Ro[:n]
    rec = (Rm.mean() - Rv.mean()) / (Ro.mean() - Rv.mean())
    rng = np.random.default_rng(seed)
    b = rng.integers(0, n, (n_boot, n))
    den = Ro[b].mean(axis=1) - Rv[b].mean(axis=1)
    recs = (Rm[b].mean(axis=1) - Rv[b].mean(axis=1)) / den
    lo, hi = np.percentile(recs, [2.5, 97.5])
    return {"rec": float(rec), "ic95": [float(lo), float(hi)], "n_rep": int(n),
            "R": float(Rm.mean()), "ventaja_oraculo": float(Ro.mean() - Rv.mean()),
            "_boot": recs}


def curso_medio(res, nom, N, pulso, tau_W=TAU_W):
    cs = [x["curso"] for x in res if x["nom"] == nom and x["N"] == N
          and x["pulso"] == pulso and abs(x["tau_W"] - tau_W) < 1e-9]
    if not cs:
        return []
    n = min(len(c) for c in cs)
    return [float(v) for v in np.mean([c[:n] for c in cs], axis=0)]


def analizar(res, crudo_e41=None, e20=None):
    out = {"config": {"N": list(NS), "T_N": {str(k): v for k, v in T_N.items()},
                      "ventanas": list(WINS), "replicas": N_REP,
                      "replicas_barrido": N_REP_BARRIDO, "tau_W": TAU_W,
                      "semilla": "7 + 1000 * i_phi + réplica", "rapido": RAPIDO},
           "rec": {}, "curso": {}, "pred": {}}
    L = []
    P = L.append
    P("=== E42 · escalera de recuperación frente a N ===")
    P(f"lambda_fp = 0, tau_W = {TAU_W:g} s, k = 15; rejilla de 10 grados; {N_REP} réplicas "
      f"({N_REP_BARRIDO} en el barrido de ventana); IC 95 % remuestreando réplicas")

    boots = {}
    for pulso, etq in ((False, "tractor"), (True, "pulso")):
        P(f"\n=== {etq} ===")
        P(f'{"N":>5s} {"ventana":>8s} | {"R ViSt":>7s} {"R orác.":>7s} {"ventaja":>7s} | '
          f'{"amplitud":>22s} | {"adaptado":>22s} | {"adapt/ampl":>10s}')
        for N in NS:
            for win in sorted({4.0, 8.0, 15.8, T_N[N]}):
                Rv, Ro = R_rep(res, "ViSt", N, pulso, win), R_rep(res, "oraculo", N, pulso, win)
                fila = {}
                for nom in ("amplitud", "adaptado"):
                    q = rec_ic(R_rep(res, nom, N, pulso, win), Rv, Ro)
                    boots[(etq, N, win, nom)] = q.pop("_boot", None)
                    fila[nom] = q
                clave_o = f"{etq}|{N}|{_w(win)}"
                out["rec"][clave_o] = {"R_vist": float(Rv.mean()) if len(Rv) else None,
                                       "R_oraculo": float(Ro.mean()) if len(Ro) else None,
                                       **fila}
                a, d = fila["amplitud"], fila["adaptado"]
                coc = d["rec"] / a["rec"] if a["rec"] else float("nan")
                P(f'{N:5d} {win:8.1f} | {Rv.mean():7.4f} {Ro.mean():7.4f} '
                  f'{Ro.mean() - Rv.mean():7.4f} | '
                  f'{a["rec"]:6.3f} [{a["ic95"][0]:5.3f}; {a["ic95"][1]:5.3f}] | '
                  f'{d["rec"]:6.3f} [{d["ic95"][0]:5.3f}; {d["ic95"][1]:5.3f}] | {coc:10.1f}')
        for N in NS:
            for nom in MEC:
                out["curso"][f"{etq}|{N}|{nom}"] = curso_medio(res, nom, N, pulso)

    P("\n=== barrido de la ventana del detector (recuperación con ventana sqrt(N)) ===")
    P(f'{"N":>5s} {"protocolo":>9s} | {"0,5 s":>22s} | {"1 s":>22s} | {"2 s":>22s} | mejor')
    for N in N_BARRIDO:
        for pulso, etq in ((False, "tractor"), (True, "pulso")):
            win = T_N[N]
            Rv, Ro = R_rep(res, "ViSt", N, pulso, win), R_rep(res, "oraculo", N, pulso, win)
            fila = {}
            for nom, tau in (("ciclo 0.5 s", 0.5), ("adaptado", 1.0), ("ciclo 2 s", 2.0)):
                Rm = R_rep(res, nom, N, pulso, win)
                n = min(len(Rm), N_REP_BARRIDO)          # mismas réplicas en las tres
                q = rec_ic(Rm[:n], Rv[:n], Ro[:n])
                q.pop("_boot", None)
                fila[_w(tau)] = q
            mejor = max(fila, key=lambda k: fila[k]["rec"])
            out["rec"][f"barrido|{etq}|{N}"] = {**fila, "mejor": mejor}
            P(f'{N:5d} {etq:>9s} | ' + " | ".join(
                f'{fila[k]["rec"]:6.3f} [{fila[k]["ic95"][0]:5.3f}; {fila[k]["ic95"][1]:5.3f}]'
                for k in ("0.5", "1", "2")) + f" | {mejor} s")

    # ---- anclas
    P("\n=== anclas ===")
    anc = {}
    if e20 is not None:
        dm = 0.0
        for etq20, pulso in (("SOSTENIDO", False), ("PULSO", True)):
            for nom20, nom in (("amplitud", "amplitud"), ("ciclo_1s", "adaptado")):
                a = np.array(e20[etq20][nom20]["R"])
                b = R_rep(res, nom, 250, pulso, 15.8)
                n = min(len(a), len(b))
                if n:
                    dm = max(dm, float(np.abs(a[:n] - b[:n]).max()))
            for k20, nom in (("R_vist", "ViSt"), ("R_orac", "oraculo")):
                a = np.array(e20[etq20][k20])
                b = R_rep(res, nom, 250, pulso, 15.8)
                n = min(len(a), len(b))
                if n:
                    dm = max(dm, float(np.abs(a[:n] - b[:n]).max()))
        anc["E20_N250"] = dm
        P(f"contra E20_pulso.json (N = 250, R por réplica): |Δ| máx = {dm:.2e}")
    if crudo_e41 is not None:
        idx = {clave_res({**x, "tau_W": TAU_W}): x for x in crudo_e41["resp"]}
        dm, n_cmp = 0.0, 0
        for x in res:
            y = idx.get(clave_res(x))
            if y is None or x["nom"] not in MEC:
                continue
            for w in ("15.8", "31.6"):
                if w in x["r"] and w in y["r"]:
                    dm = max(dm, abs(x["r"][w] - y["r"][w]))
                    n_cmp += 1
        anc["E41_crudo"] = {"delta_max": dm, "comparaciones": n_cmp}
        P(f"contra E41_credulidad_causal_crudo.json (corrida por corrida): |Δ| máx = {dm:.2e} "
          f"en {n_cmp} comparaciones")
    out["anclas"] = anc

    # ---- predicciones declaradas
    def g(etq, N, win, nom):
        return out["rec"][f"{etq}|{N}|{_w(win)}"][nom]["rec"]

    pr = out["pred"]
    try:
        pr["N0"] = {"E20": anc.get("E20_N250"), "E41": anc.get("E41_crudo"),
                    "cumple": bool(anc.get("E20_N250", 1) < 1e-9
                                   and anc.get("E41_crudo", {}).get("delta_max", 1) < 1e-9)}
        t500, t1000 = g("tractor", 500, 22.4, "adaptado"), g("tractor", 1000, 31.6, "adaptado")
        t250 = g("tractor", 250, 15.8, "adaptado")
        pr["N1"] = {"rec": {"250": t250, "500": t500, "1000": t1000},
                    "cumple": bool(t250 > t500 > t1000 and 0.45 <= t500 <= 0.65
                                   and 0.30 <= t1000 <= 0.46)}
        p250, p500, p1000 = (g("pulso", 250, 15.8, "adaptado"), g("pulso", 500, 22.4, "adaptado"),
                             g("pulso", 1000, 31.6, "adaptado"))
        pr["N2"] = {"rec": {"250": p250, "500": p500, "1000": p1000},
                    "cumple": bool(0.47 <= p500 <= 0.60 and 0.40 <= p1000 <= 0.51)}
        amp, coc = [], []
        for etq in ("tractor", "pulso"):
            for N in NS:
                a, d = g(etq, N, T_N[N], "amplitud"), g(etq, N, T_N[N], "adaptado")
                amp.append(a)
                coc.append(d / a)
        pr["N3"] = {"amplitud_max": max(amp), "cociente_min": min(coc),
                    "cumple": bool(max(amp) < 0.10 and min(coc) >= 5.0)}
        f1000 = g("tractor", 1000, 15.8, "adaptado")
        pr["N4"] = {"fija_15.8": f1000, "sqrtN": t1000, "cumple": bool(f1000 > t1000)}
        mej = {k: out["rec"][k]["mejor"] for k in out["rec"] if k.startswith("barrido|")}
        pr["N5"] = {"mejor": mej,
                    "cumple": bool(all(v == "1" for k, v in mej.items() if k.endswith("|1000")))}
        r8, r158 = g("tractor", 1000, 8.0, "adaptado"), g("tractor", 1000, 15.8, "adaptado")
        pr["N6"] = {"rec_8s": r8, "rec_15.8s": r158, "rec_31.6s": t1000,
                    "cumple": bool(r8 > r158)}
    except KeyError as e:                                # corrida de humo incompleta
        pr["incompleto"] = str(e)

    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        if isinstance(v, dict) and "cumple" in v:
            P(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v, ensure_ascii=False)[:300]}')
        else:
            P(f"{k}: {v}")
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if SOLO_ANALISIS:
        with open(F_CRUDO) as fh:
            res = json.load(fh)["resp"]
        seg = 0.0
    else:
        res, seg = correr(trabajos(), F_PARCIAL, "E42")
        with open(F_CRUDO, "w") as fh:
            json.dump({"resp": res}, fh)
        if os.path.exists(F_PARCIAL):
            os.remove(F_PARCIAL)
    crudo_e41 = e20 = None
    if os.path.exists("resultados/E41_credulidad_causal_crudo.json"):
        with open("resultados/E41_credulidad_causal_crudo.json") as fh:
            crudo_e41 = json.load(fh)
    if os.path.exists("resultados/E20_pulso.json"):
        with open("resultados/E20_pulso.json") as fh:
            e20 = json.load(fh)
    out, L = analizar(res, crudo_e41, e20)
    L.append(f"\n{seg:.0f} s")
    out["segundos"] = seg
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
