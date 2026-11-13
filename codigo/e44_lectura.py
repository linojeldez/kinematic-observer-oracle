"""Etapa 44: ¿de dónde sale lo que el oráculo gana sobre el detector adaptado?

Por qué. Con tau_W = 4 s la recuperación del detector adaptado baja con el tamaño de la
bandada (E42: 0,72 / 0,54 / 0,40 con líder forzado; 0,60 / 0,49 / 0,45 con maniobra
única). La lectura del paper corto es que el oráculo sigue pesando al líder cuando su
huella cinemática ya terminó. E43 la apoya, pero al acortar tau_W cambia dos cosas a la
vez: cuánto insiste el líder y cuánto tiempo lo leen los demás. Aquí se interviene sólo
la LECTURA: una subclase de Flock modifica `_perceived_will` (lo que los vecinos leen) y
nada más. El estado, la dinámica del líder y el motor quedan intactos.

Brazos (tau_W = 4 s, N = 250 y 1000, líder forzado y maniobra única, 18 ángulos, 8
réplicas, semilla 7 + 1000 * i_phi + réplica, como E42):
  G1, G2, G4, G8  oráculo con compuerta: los vecinos leen el will sólo durante los
                  primeros T s desde que el líder se informa.
  D               detector adaptado (ciclo de 1 s) cuyo posterior se conserva sólo
                  sobre el líder. Quita relevos y falsas alarmas.
  R               el mismo detector con el posterior sobre el líder anulado.
Además D y R con tau_W = 1 s y maniobra única (el resultado central de E43 y E43b).
En D y R se registra el posterior sobre el líder cada 0,1 s.

ViSt, adaptado y oráculo salen de los crudos de E42, E43 y E43b (mismas semillas).
Anclas exactas: la subclase sin intervención frente al adaptado de E42 y la compuerta
con T infinito frente al oráculo de E42, réplica 0.

Las predicciones (M0 a M7) están en docs/bitacora/resultados_E36-E45.md, declaradas
antes de correr.

    PYTHONPATH=codigo python codigo/e44_lectura.py
    PYTHONPATH=codigo E44_RAPIDO=1 python codigo/e44_lectura.py        # humo
    PYTHONPATH=codigo E44_SOLO_ANALISIS=1 python codigo/e44_lectura.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

import flocking
from flocking import Flock, Params, run_response, phi_grid
from parametros import P0
from e42_escalera_N import (MEC, D_PHI, SEMILLA_R, T_N, WINS, T_MUESTRA, _w, R_rep,
                            rec_ic, clave_res, curso_medio)

RAPIDO = bool(os.environ.get("E44_RAPIDO"))
SOLO_ANALISIS = bool(os.environ.get("E44_SOLO_ANALISIS"))
W = int(os.environ.get("E44_W", "8"))

N_REP = 1 if RAPIDO else 8
NS = (250,) if RAPIDO else (250, 1000)
COMPUERTAS = (1.0, 2.0, 4.0, 8.0)
DT_REG = 0.1                    # muestreo del posterior sobre el líder (s)
BRAZOS = tuple(f"G{t:g}" for t in COMPUERTAS) + ("D", "R")

SUF = "_rapido" if RAPIDO else ""
F_CRUDO = f"resultados/E44_lectura_crudo{SUF}.json"
F_JSON = f"resultados/E44_lectura{SUF}.json"
F_TXT = f"resultados/E44_lectura{SUF}.txt"
F_PARCIAL = F_CRUDO + ".parcial"


class FlockLectura(Flock):
    """Flock con la LECTURA del will intervenida; estado y dinámica no cambian.

    modo = None           sin intervención (ancla)
    modo = 'compuerta'    los vecinos leen sólo durante T_g s desde que alguien se informa
    modo = 'solo_lider'   lo percibido se conserva sólo sobre el líder
    modo = 'sin_lider'    lo percibido sobre el líder se anula
    La columna propia (la intención de uno mismo) nunca se toca.
    """

    modo = None
    T_g = np.inf
    ultimo = None

    def __init__(self, p, model):
        super().__init__(p, model)
        self._t_on = None
        self.serie_wL = []
        self._n_reg = max(int(round(DT_REG / p.dt)), 1)
        FlockLectura.ultimo = self

    def _perceived_will(self, idx, dist):
        w = super()._perceived_will(idx, dist)
        if self.modo is None:
            return w
        if self._t_on is None and self.will.max() > 0.5:
            self._t_on = self.t
        own = idx == np.arange(self.p.N)[:, None]
        if self.modo == "compuerta":
            if self._t_on is not None and self.t - self._t_on >= self.T_g - 1e-9:
                w = np.where(own, w, 0.0)
            return w
        es_lider = idx == self.leader
        if self._t_on is not None and self._paso % self._n_reg == 0:
            v = w[es_lider & ~own]
            self.serie_wL.append(float(v.mean()) if v.size else float("nan"))
        if self.modo == "solo_lider":
            return np.where(own | es_lider, w, 0.0)
        if self.modo == "sin_lider":
            return np.where(es_lider & ~own, 0.0, w)
        raise ValueError(self.modo)


def job(args):
    """Una corrida de respuesta con la lectura intervenida."""
    nom, model, kw, N, tau_W, pulso, i_phi, phi, rep, T, modo, T_g = args
    p = Params(**{**asdict(P0), **kw, "N": N, "lambda_fp": 0.0, "tau_W": tau_W,
                  "seed": SEMILLA_R + 1000 * i_phi + rep})
    flocking.Flock = type("FlockLecturaJob", (FlockLectura,), {"modo": modo, "T_g": T_g})
    t0 = time.time()
    try:
        r = run_response(p, model, float(phi), T, every_coh=100, pulse=pulso)
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    m = int(round(T_MUESTRA / p.dt))
    n = (len(proj) // m) * m
    out = {"nom": nom, "N": N, "tau_W": tau_W, "pulso": bool(pulso), "i_phi": i_phi,
           "rep": rep, "pol": r["pol"], "frac_min": r["frac_min"],
           "r": {_w(w): float(proj[:int(round(w / p.dt))].mean())
                 for w in WINS if w <= T + 1e-9},
           "curso": [float(x) for x in proj[:n].reshape(-1, m).mean(axis=1)],
           "seg": time.time() - t0}
    if modo in ("solo_lider", "sin_lider"):
        s = np.array(FlockLectura.ultimo.serie_wL, dtype=float)
        ok = np.isfinite(s)
        out["T_lect"] = float((s[ok] > 0.5).sum() * DT_REG)
        k = int(round(T_MUESTRA / DT_REG))
        nb = (len(s) // k) * k
        out["wL_curso"] = [float(x) for x in np.nanmean(s[:nb].reshape(-1, k), axis=1)]
    return out


def clave(a):
    return f"{a[0]}|{a[3]}|{a[4]:g}|{int(a[5])}|{a[6]}|{a[8]}"


def trabajos():
    phis = phi_grid(D_PHI)
    phis_i = [(0, phis[0]), (9, phis[9])] if RAPIDO else list(enumerate(phis))
    m_or, kw_or = MEC["oraculo"]
    m_ad, kw_ad = MEC["adaptado"]
    inf = float("inf")
    out = []
    for N in NS:
        T = T_N[N]
        for pulso in (False, True):
            for i, phi in phis_i:
                phi = float(phi)
                for rep in range(N_REP):
                    for tg in COMPUERTAS:
                        out.append((f"G{tg:g}", m_or, kw_or, N, 4.0, pulso, i, phi, rep, T,
                                    "compuerta", tg))
                    out.append(("D", m_ad, kw_ad, N, 4.0, pulso, i, phi, rep, T, "solo_lider", inf))
                    out.append(("R", m_ad, kw_ad, N, 4.0, pulso, i, phi, rep, T, "sin_lider", inf))
                # anclas exactas, réplica 0
                out.append(("ancla_oraculo", m_or, kw_or, N, 4.0, pulso, i, phi, 0, T,
                            "compuerta", inf))
                if N == 250:
                    out.append(("ancla_adaptado", m_ad, kw_ad, N, 4.0, pulso, i, phi, 0, T,
                                None, inf))
        for i, phi in phis_i:                             # tau_W = 1 s, maniobra única
            phi = float(phi)
            for rep in range(N_REP):
                out.append(("D", m_ad, kw_ad, N, 1.0, True, i, phi, rep, T, "solo_lider", inf))
                out.append(("R", m_ad, kw_ad, N, 1.0, True, i, phi, rep, T, "sin_lider", inf))
    out.sort(key=lambda a: -a[3] * a[9])                  # los largos primero
    return out


def correr(jobs, f_parcial, etiqueta):
    """Corre con reanudación: guarda parciales y se salta lo ya hecho."""
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


def cargar_base():
    """ViSt, amplitud, adaptado y oráculo ya corridos, con las mismas semillas."""
    with open("resultados/E42_escalera_N_crudo.json") as fh:
        e42 = json.load(fh)["resp"]
    with open("resultados/E43_escalera_tauW_crudo.json") as fh:
        e43 = json.load(fh)["resp"]
    with open("resultados/E43b_tauW_N1000_crudo.json") as fh:
        e43b = json.load(fh)["resp"]
    base = [x for x in e42 if x["nom"] in ("ViSt", "amplitud", "adaptado", "oraculo")]
    base += [x for x in e43 if x["N"] == 250 and x["pulso"] and abs(x["tau_W"] - 1.0) < 1e-9]
    base += [x for x in e43b if x["nom"] != "ViSt"]
    # ViSt no usa la voluntad: el de E42 vale para tau_W = 1 s (verificado en E43b)
    base += [{**x, "tau_W": 1.0} for x in e42
             if x["nom"] == "ViSt" and x["N"] == 1000 and x["pulso"]]
    return base, e42


def _quita(q):
    q = dict(q)
    q.pop("_boot", None)
    return q


def _fmt(q):
    return f'{q["rec"]:6.3f} [{q["ic95"][0]:5.3f}; {q["ic95"][1]:5.3f}]'


def t_equivalente(rec_x, recs_g):
    """Duración de compuerta que iguala rec_x, interpolando en log T entre compuertas."""
    ts = list(COMPUERTAS)
    if rec_x <= recs_g[0]:
        return f"<{ts[0]:g}"
    if rec_x >= recs_g[-1]:
        return f">{ts[-1]:g}"
    for a in range(len(ts) - 1):
        if recs_g[a] <= rec_x <= recs_g[a + 1]:
            f = (rec_x - recs_g[a]) / max(recs_g[a + 1] - recs_g[a], 1e-12)
            return float(np.exp(np.log(ts[a]) + f * (np.log(ts[a + 1]) - np.log(ts[a]))))
    return float("nan")


def analizar(nuevos, base, e42):
    res = base + [x for x in nuevos if not x["nom"].startswith("ancla")]
    out = {"config": {"N": list(NS), "T_N": {str(k): T_N[k] for k in NS},
                      "compuertas_s": list(COMPUERTAS), "replicas": N_REP,
                      "semilla": "7 + 1000 * i_phi + réplica", "rapido": RAPIDO},
           "anclas": {}, "rec": {}, "razon": {}, "lectura": {}, "curso": {}, "wL_curso": {},
           "pred": {}}
    L = []
    P = L.append
    P("=== E44 · intervención sobre la lectura del will ===")
    P(f"lambda_fp = 0, k = 15, rejilla de 10 grados, {N_REP} réplicas; IC 95 % remuestreando "
      "réplicas emparejadas por semilla")

    # ---- anclas exactas
    idx = {clave_res(x): x for x in e42}
    P("\n=== anclas (réplica 0, corrida por corrida) ===")
    for anc, ref in (("ancla_oraculo", "oraculo"), ("ancla_adaptado", "adaptado")):
        dm, n_cmp = 0.0, 0
        for x in nuevos:
            if x["nom"] != anc:
                continue
            y = idx.get(clave_res({**x, "nom": ref}))
            if y is None:
                continue
            for w in x["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w]))
                n_cmp += 1
            dm = max(dm, float(np.abs(np.array(x["curso"]) - np.array(y["curso"])).max()))
        out["anclas"][anc] = {"delta_max": dm, "comparaciones": n_cmp}
        P(f"{anc} frente a «{ref}» de E42: |Δ| máx = {dm:.2e} en {n_cmp} comparaciones")

    # ---- recuperación, tau_W = 4 s
    noms = ("adaptado",) + BRAZOS
    for pulso, etq in ((False, "tractor"), (True, "pulso")):
        for N in NS:
            for win in (4.0, T_N[N]):
                Rv = R_rep(res, "ViSt", N, pulso, win, 4.0)
                Ro = R_rep(res, "oraculo", N, pulso, win, 4.0)
                fila = {nom: _quita(rec_ic(R_rep(res, nom, N, pulso, win, 4.0), Rv, Ro))
                        for nom in noms}
                out["rec"][f"{etq}|{N}|{_w(win)}"] = {
                    "R_vist": float(Rv.mean()), "R_oraculo": float(Ro.mean()), **fila}
            for nom in BRAZOS:
                out["curso"][f"{etq}|{N}|{nom}"] = curso_medio(res, nom, N, pulso, 4.0)

    def g(etq, N, win, nom):
        return out["rec"][f"{etq}|{N}|{_w(win)}"][nom]

    for titulo, fwin in (("ventana sqrt(N)", lambda N: T_N[N]), ("primeros 4 s", lambda N: 4.0)):
        P(f"\n=== tau_W = 4 s · recuperación, {titulo} ===")
        P(f'{"celda":<16s} | {"adaptado":>22s} | ' + " | ".join(f"{n:>22s}" for n in BRAZOS))
        for pulso, etq in ((False, "tractor"), (True, "pulso")):
            for N in NS:
                P(f"{etq + ' N=' + str(N):<16s} | " + " | ".join(
                    _fmt(g(etq, N, fwin(N), nom)) for nom in noms))

    # ---- razones D/adaptado y R/adaptado, compuerta equivalente
    P("\n=== tau_W = 4 s · qué parte del detector es el líder (ventana sqrt(N)) ===")
    P(f'{"celda":<16s} | {"D / adaptado":>22s} | {"R / adaptado":>22s} | '
      f'{"T equiv. adaptado":>17s} | {"T equiv. D":>10s} | {"T_lect (D)":>18s}')
    for pulso, etq in ((False, "tractor"), (True, "pulso")):
        for N in NS:
            win = T_N[N]
            Rv = R_rep(res, "ViSt", N, pulso, win, 4.0)
            Rm = R_rep(res, "adaptado", N, pulso, win, 4.0)
            qd = _quita(rec_ic(R_rep(res, "D", N, pulso, win, 4.0), Rv, Rm))
            qr = _quita(rec_ic(R_rep(res, "R", N, pulso, win, 4.0), Rv, Rm))
            recs_g = [g(etq, N, win, f"G{t:g}")["rec"] for t in COMPUERTAS]
            te_m = t_equivalente(g(etq, N, win, "adaptado")["rec"], recs_g)
            te_d = t_equivalente(g(etq, N, win, "D")["rec"], recs_g)
            tl = np.array([x["T_lect"] for x in nuevos if x["nom"] == "D" and x["N"] == N
                           and x["pulso"] == pulso and abs(x["tau_W"] - 4.0) < 1e-9])
            lect = {"media": float(tl.mean()), "mediana": float(np.median(tl)),
                    "p10": float(np.percentile(tl, 10)), "p90": float(np.percentile(tl, 90)),
                    "n": int(len(tl))}
            rd = g(etq, N, win, "D")["rec"]
            cerca = min(COMPUERTAS, key=lambda t: abs(g(etq, N, win, f"G{t:g}")["rec"] - rd))
            out["razon"][f"{etq}|{N}"] = {"D_sobre_adaptado": qd, "R_sobre_adaptado": qr,
                                          "T_equiv_adaptado": te_m, "T_equiv_D": te_d,
                                          "compuerta_mas_cercana_a_D": cerca}
            out["lectura"][f"{etq}|{N}|4"] = lect
            wl = [x["wL_curso"] for x in nuevos if x["nom"] == "D" and x["N"] == N
                  and x["pulso"] == pulso and abs(x["tau_W"] - 4.0) < 1e-9]
            nmin = min(len(c) for c in wl)
            out["wL_curso"][f"{etq}|{N}|4"] = [float(v) for v in
                                              np.nanmean([c[:nmin] for c in wl], axis=0)]
            f_te = lambda v: v if isinstance(v, str) else f"{v:.2f}"
            P(f"{etq + ' N=' + str(N):<16s} | {_fmt(qd)} | {_fmt(qr)} | {f_te(te_m):>15s} s | "
              f"{f_te(te_d):>8s} s | {lect['media']:5.2f} s (p10–p90 {lect['p10']:.1f}–{lect['p90']:.1f})")

    P("\n=== posterior medio sobre el líder en D (tau_W = 4 s), por tramo de 0,5 s ===")
    for k_, v in out["wL_curso"].items():
        P(f"{k_:<16s} " + " ".join(f"{x:4.2f}" for x in v[:16]) + "  …  " + f"{v[-1]:4.2f}")

    # ---- tau_W = 1 s, maniobra única
    P("\n=== tau_W = 1 s · maniobra única (ventana sqrt(N)) ===")
    P(f'{"N":>5s} | {"ventaja":>7s} | {"adaptado":>22s} | {"D":>22s} | {"R":>22s} | T_lect (D)')
    for N in NS:
        win = T_N[N]
        Rv = R_rep(res, "ViSt", N, True, win, 1.0)
        Ro = R_rep(res, "oraculo", N, True, win, 1.0)
        fila = {nom: _quita(rec_ic(R_rep(res, nom, N, True, win, 1.0), Rv, Ro))
                for nom in ("adaptado", "D", "R")}
        tl = np.array([x["T_lect"] for x in nuevos if x["nom"] == "D" and x["N"] == N
                       and x["pulso"] and abs(x["tau_W"] - 1.0) < 1e-9])
        out["rec"][f"pulso|{N}|{_w(win)}|tauW1"] = {"R_vist": float(Rv.mean()),
                                                   "R_oraculo": float(Ro.mean()), **fila}
        out["lectura"][f"pulso|{N}|1"] = {"media": float(tl.mean()),
                                         "mediana": float(np.median(tl)), "n": int(len(tl))}
        P(f"{N:5d} | {Ro.mean() - Rv.mean():7.4f} | " + " | ".join(
            _fmt(fila[n]) for n in ("adaptado", "D", "R")) + f" | {tl.mean():5.2f} s")

    # ---- predicciones declaradas
    pr = out["pred"]
    celdas = [(etq, N) for etq in ("tractor", "pulso") for N in NS]
    try:
        a = out["anclas"]
        pr["M0"] = {"anclas": a, "cumple": bool(
            all(v["comparaciones"] > 0 and v["delta_max"] == 0.0 for v in a.values()))}
        esc = {f"{e}|{N}": [g(e, N, T_N[N], f"G{t:g}")["rec"] for t in COMPUERTAS]
               for e, N in celdas}
        pr["M1"] = {"rec_G": esc, "cumple": bool(all(
            v[0] < v[1] < v[2] < v[3] < 1.0 for v in esc.values()))}
        m2 = {f"{e}|{N}": {"G2": g(e, N, T_N[N], "G2")["rec"],
                           "adaptado": g(e, N, T_N[N], "adaptado")["rec"],
                           "G8": g(e, N, T_N[N], "G8")["rec"]} for e, N in celdas}
        pr["M2"] = {"rec": m2, "cumple": bool(all(
            v["G2"] <= v["adaptado"] <= v["G8"] for v in m2.values()))}
        if 1000 in NS:
            m3 = {}
            for pulso, etq, umbral in ((False, "tractor", 0.10), (True, "pulso", 0.05)):
                b = []
                for N, sem in ((250, 42), (1000, 43)):
                    q = rec_ic(R_rep(res, "G4", N, pulso, T_N[N], 4.0),
                               R_rep(res, "ViSt", N, pulso, T_N[N], 4.0),
                               R_rep(res, "oraculo", N, pulso, T_N[N], 4.0), seed=sem)
                    b.append((q["rec"], q["_boot"]))
                dif = b[0][0] - b[1][0]
                lo, hi = np.percentile(b[0][1] - b[1][1], [2.5, 97.5])
                m3[etq] = {"G4_250": b[0][0], "G4_1000": b[1][0], "caida": float(dif),
                           "ic95": [float(lo), float(hi)], "umbral": umbral,
                           "cumple": bool(dif >= umbral)}
            pr["M3"] = {**m3, "cumple": bool(all(v["cumple"] for v in m3.values()))}
            p1 = {str(N): out["rec"][f"pulso|{N}|{_w(T_N[N])}|tauW1"]["D"]["rec"] for N in NS}
            pr["M5"] = {"rec_D_tauW1": p1, "cumple": bool(all(v >= 0.80 for v in p1.values()))}
            tl = {str(N): out["lectura"][f"pulso|{N}|4"]["media"] for N in NS}
            pr["M6"] = {"T_lect_pulso": tl,
                        "T_lect_tractor": {str(N): out["lectura"][f"tractor|{N}|4"]["media"]
                                           for N in NS},
                        "cumple": bool(all(1.0 <= v <= 4.0 for v in tl.values())
                                       and abs(tl["250"] - tl["1000"]) < 0.5)}
        m4 = {f"{e}|{N}": {"D": out["razon"][f"{e}|{N}"]["D_sobre_adaptado"]["rec"],
                           "R": out["razon"][f"{e}|{N}"]["R_sobre_adaptado"]["rec"]}
              for e, N in celdas}
        pr["M4"] = {"razon_sobre_adaptado": m4, "cumple": bool(all(
            v["D"] >= 0.8 and v["R"] <= 0.2 for v in m4.values()))}
        m7 = {}
        for e, N in celdas:
            c = out["razon"][f"{e}|{N}"]["compuerta_mas_cercana_a_D"]
            t = out["lectura"][f"{e}|{N}|4"]["media"]
            m7[f"{e}|{N}"] = {"compuerta": c, "T_lect": t,
                              "factor": float(max(c / max(t, 1e-9), t / c))}
        pr["M7"] = {"celdas": m7, "cumple": bool(all(v["factor"] < 2.0 for v in m7.values()))}
    except KeyError as e:                                 # corrida de humo incompleta
        pr["incompleto"] = str(e)

    P("\n=== Predicciones declaradas ===")
    for k_ in sorted(pr):
        v = pr[k_]
        if isinstance(v, dict) and "cumple" in v:
            P(f'{k_}: {"CUMPLE" if v["cumple"] else "FALLA"}  '
              f'{json.dumps(v, ensure_ascii=False)[:420]}')
        else:
            P(f"{k_}: {v}")
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if SOLO_ANALISIS:
        with open(F_CRUDO) as fh:
            nuevos = json.load(fh)["resp"]
        seg = 0.0
    else:
        nuevos, seg = correr(trabajos(), F_PARCIAL, "E44")
        with open(F_CRUDO, "w") as fh:
            json.dump({"resp": nuevos}, fh)
        if os.path.exists(F_PARCIAL):
            os.remove(F_PARCIAL)
    base, e42 = cargar_base()
    out, L = analizar(nuevos, base, e42)
    L.append(f"\n{seg:.0f} s")
    out["segundos"] = seg
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
