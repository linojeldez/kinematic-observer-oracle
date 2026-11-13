"""Etapa 66 (plan 15, punto 5): 16 réplicas en las celdas que tenían 4.

Barrido de la ventana (E42 y E42b; N = 250 y 1000), agilidad (E55), ruido (E56) y peso fijo
(E61), todos con tau_W = 4 s y los dos protocolos. Se agregan las réplicas 4-15 con las mismas
semillas (7 + 1000 i_phi + réplica) y la misma construcción de trabajos de cada etapa; las
referencias que ya tienen 32 réplicas (E48) no se repiten. Ancla: réplica 0, ángulos 4 y 13 de
cada familia, que debe reproducir lo guardado byte a byte.

Criterio declarado (docs/bitacora/resultados_E56-E65.md, «Plan 15 · E66»): la mejor ventana se
mantiene o pasa a una adyacente de la rejilla.

    PYTHONPATH=codigo python codigo/e66_replicas.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import fase4
from flocking import phi_grid
from fase4 import D_PHI, T_N, W, trabajo, clave, crudos_previos, R_rep, rec_ic, limpio
import e55_agilidad as e55
import e56_ventana_ruido as e56
import e61_ventana_peso_fijo as e61

N_REP, NUEVAS = 16, range(4, 16)
ANCLA_PHI = (4, 13)
VENT = (0.25, 0.5, 1.0, 2.0, 4.0)
NOM_BARRIDO = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s", 4.0: "ciclo 4 s"}
PROT = (("forzado", False, "ViSt"), ("maniobra", True, "CoLd"))
F_CRUDO, F_JSON, F_TXT = ("resultados/E66_replicas_crudo.json", "resultados/E66_replicas.json",
                          "resultados/E66_replicas.txt")


def _filtrar(jobs, fam):
    out = []
    for a in jobs:
        if a["rep"] in NUEVAS or (a["rep"] == 0 and a["i_phi"] in ANCLA_PHI):
            out.append({**a, "fam": fam})
    return out


def trabajos():
    phis = phi_grid(D_PHI)
    jobs = []
    # barrido de la ventana: E42 (0,5, 1, 2 s) y E42b (0,25, 4 s); la de 1 s ya tiene 32 réplicas
    bar = []
    for N in (250, 1000):
        for v, nom in NOM_BARRIDO.items():
            if v == 1.0:
                continue
            for etq, pulso, ref in PROT:
                for i, phi in enumerate(phis):
                    for rep in range(N_REP):
                        bar.append(trabajo("base", nom, N, 4.0, pulso, i, phi, rep, kw={"tau_det": v}, base="adaptado"))
    jobs += _filtrar(bar, "barrido")
    # agilidad (E55): las celdas con a_max distinto de 10 (la de 10 es la celda base)
    anch55 = json.load(open("resultados/E55_agilidad.json"))["anchuras"]
    e55.N_REP = N_REP
    jobs += _filtrar([a for a in e55.trabajos(anch55) if a["celda"] != "a10"], "agilidad")
    # ruido (E56)
    anch49 = json.load(open("resultados/E49_sensibilidad_anchuras.json"))
    e56.N_REP = N_REP
    jobs += _filtrar(e56.trabajos(anch49), "ruido")
    # peso fijo (E61); su ancla con factor 1 no se repite
    e61.N_REP = N_REP
    jobs += _filtrar([a for a in e61.trabajos(anch55, anch49) if not a["celda"].startswith("ancla")], "peso_fijo")
    jobs.sort(key=lambda a: -a["N"] * a["T"])
    return jobs


def job(a):
    if "factor" in a:
        return {**e61.job(a), "fam": a["fam"]}
    return {**fase4.job(a), "fam": a["fam"]}


def previos():
    out = crudos_previos()
    out += [{**x, "celda": "base"} for x in json.load(open("resultados/E48_replicas_crudo.json"))["resp"]
            if x.get("celda", "base") == "base"]
    for x in json.load(open("resultados/E42b_barrido_ventana_crudo.json"))["resp"]:
        out.append({**x, "celda": "base"})
    for f in ("E55_agilidad", "E56_ventana_ruido", "E61_ventana_peso_fijo"):
        out += json.load(open(f"resultados/{f}_crudo.json"))["resp"]
    return list({clave(x): x for x in out}.values())


def familias():
    """(familia, celda, N, {ventana: nombre}) para el análisis."""
    # con N = 1000 la ventana de 0,25 s no tiene réplicas 0-3 (E42b no la corrió): se analiza 0,5-4 s
    out = [("barrido", "base", 250, NOM_BARRIDO),
           ("barrido", "base", 1000, {v: n for v, n in NOM_BARRIDO.items() if v != 0.25})]
    for a in (5.0, 20.0, 40.0):
        out.append(("agilidad", f"a{a:g}", 250, {v: f"ciclo{v:g}" for v in VENT}))
    for f in (0.5, 2.0):
        out.append(("ruido", f"f{f:g}", 250, {v: f"ciclo{v:g}" for v in VENT}))
    for c in ("a5", "a20", "a40", "f0.5", "f2"):
        out.append(("peso_fijo", c, 250, {v: (f"ciclo{v:g}" if v == 1.0 else f"ciclo{v:g}_n12") for v in VENT}))
    return out


def recup(todo, celda, nom, ref, N, pulso, reps, celda_ref=None):
    cr = celda_ref or celda
    w = T_N[N]
    im, Rm = R_rep(todo, celda, nom, N, pulso, w, 4.0, reps)
    iv, Rv = R_rep(todo, cr, ref, N, pulso, w, 4.0, reps)
    io, Ro = R_rep(todo, cr, "oraculo", N, pulso, w, 4.0, reps)
    com = sorted(set(im) & set(iv) & set(io))
    g = lambda ids, R: np.array([R[ids.index(k)] for k in com])
    return g(im, Rm), g(iv, Rv), g(io, Ro), com


def mejor_boot(series, n_boot=2000, seed=61):
    """Frecuencia con que cada ventana es la mejor al remuestrear réplicas (las mismas en todas)."""
    rng = np.random.default_rng(seed)
    vs = list(series)
    n = len(next(iter(series.values()))[0])
    cuenta = {v: 0 for v in vs}
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        rec = {v: (m[b].mean() - r[b].mean()) / (o[b].mean() - r[b].mean()) for v, (m, r, o) in series.items()}
        cuenta[max(rec, key=rec.get)] += 1
    return {f"{v:g}": c / n_boot for v, c in cuenta.items()}


def analizar(todo):
    out, L = {}, ["=== E66 · 16 réplicas en las celdas de 4 (tau_W = 4 s) ==="]
    for fam, celda, N, noms in familias():
        for etq, pulso, ref in PROT:
            c = f"{fam}|{celda}|N{N}|{etq}"
            fila = {}
            for n_rep in (4, N_REP):
                series = {}
                for v, nom in noms.items():
                    m, r, o, com = recup(todo, celda, nom, ref, N, pulso, list(range(n_rep)))
                    if len(com) == n_rep:
                        series[v] = (m, r, o)
                if len(series) < len(noms):
                    fila[f"n{n_rep}"] = {"incompleta": sorted(f"{v:g}" for v in noms if v not in series)}
                    continue
                rec = {f"{v:g}": limpio(rec_ic(*s)) for v, s in series.items()}
                mejor = max(rec, key=lambda v: rec[v]["rec"])
                fila[f"n{n_rep}"] = {"rec": rec, "mejor": mejor, "boot_mejor": mejor_boot(series)}
            out[c] = fila
            if "mejor" in fila.get(f"n{N_REP}", {}) and "mejor" in fila.get("n4", {}):
                r16 = fila[f"n{N_REP}"]["rec"]
                L.append(f"{c:34s} mejor 4 rép.: {fila['n4']['mejor']:>4s} s | 16 rép.: {fila[f'n{N_REP}']['mejor']:>4s} s | "
                         + "  ".join(f"{v}: {q['rec']:.3f}" for v, q in r16.items())
                         + f" | boot: " + " ".join(f"{v}:{p:.2f}" for v, p in fila[f"n{N_REP}"]["boot_mejor"].items() if p > 0))
            else:
                L.append(f"{c:34s} incompleta: {fila}")
    # sin memoria con 16 réplicas en las celdas de agilidad y ruido (tablas del suplemento)
    amp = {}
    for fam, celda, N, _ in familias():
        if fam not in ("agilidad", "ruido"):
            continue
        for etq, pulso, ref in PROT:
            m, r, o, com = recup(todo, celda, "amplitud", ref, N, pulso, list(range(N_REP)))
            if len(com) == N_REP:
                amp[f"{fam}|{celda}|N{N}|{etq}"] = limpio(rec_ic(m, r, o))
    # criterio: la mejor ventana se mantiene o pasa a una adyacente
    rej = [f"{v:g}" for v in VENT]
    pares = [(c, f["n4"]["mejor"], f[f"n{N_REP}"]["mejor"]) for c, f in out.items()
             if "mejor" in f.get("n4", {}) and "mejor" in f.get(f"n{N_REP}", {})]
    igual = [p for p in pares if p[1] == p[2]]
    adyac = [p for p in pares if p[1] != p[2] and abs(rej.index(p[1]) - rej.index(p[2])) == 1]
    lejos = [p for p in pares if abs(rej.index(p[1]) - rej.index(p[2])) > 1]
    pred = {"V1": {"iguales": len(igual), "adyacentes": len(adyac), "mas_lejos": lejos, "total": len(pares),
                   "cumple": bool(pares and not lejos)}}
    L.append(f"\nmejor ventana: igual en {len(igual)}, adyacente en {len(adyac)}, más lejos en {len(lejos)} de {len(pares)}")
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in pred.items()]
    return {"celdas": out, "amplitud": amp, "pred": pred}, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analisis":       # rehace sólo el análisis desde el crudo
        res = json.load(open(F_CRUDO))["resp"]
        prev = previos(); idx = {clave(x): x for x in prev}
        dm, n = 0.0, 0
        for x in res:
            if x["rep"] == 0 and clave(x) in idx:
                y = idx[clave(x)]
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
        todo = list({clave(x): x for x in prev + [x for x in res if x["rep"] >= 4]}.values())
        out, L = analizar(todo)
        out["ancla"] = {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)}
        L.insert(1, f"ancla (réplica 0 frente a lo guardado): |Δ| máx = {dm:.2e} en {n} comparaciones")
        L.append("\n(análisis rehecho desde el crudo: con N = 1000 la rejilla es 0,5-4 s)")
        json.dump(out, open(F_JSON, "w"), indent=1, default=float)
        open(F_TXT, "w").write("\n".join(L) + "\n")
        print("\n".join(L)); sys.exit(0)
    jobs = trabajos()
    f_par = F_CRUDO + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[E66] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for k, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or k == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w"))
                ult = time.time()
                print(f"[E66] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    seg = time.time() - t0
    res = [hechos[clave(a)] for a in jobs]
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    prev = previos()
    idx = {clave(x): x for x in prev}
    dm, n = 0.0, 0
    for x in res:
        if x["rep"] == 0:
            y = idx.get(clave(x))
            if y:
                for w in x["r"]:
                    if w in y["r"]:
                        dm = max(dm, abs(x["r"][w] - y["r"][w])); n += 1
    nuevos = [x for x in res if x["rep"] >= 4]
    todo = list({clave(x): x for x in prev + nuevos}.values())
    out, L = analizar(todo)
    out["ancla"] = {"delta_max": dm, "comparaciones": n, "cumple": bool(n > 0 and dm < 1e-12)}
    L.insert(1, f"ancla (réplica 0 frente a lo guardado): |Δ| máx = {dm:.2e} en {n} comparaciones")
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
