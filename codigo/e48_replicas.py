"""Etapa 48 (C1 del plan de fortalecimiento): 32 réplicas pareadas en las celdas de cabecera.

Por qué. Las cifras del resumen del Paper A corto (sin memoria: menos de un décimo; ventana
de 1 s: 37 a 72 % con tau_W = 4 s y todo con tau_W = 1 s) salen de 8 réplicas pareadas. Se
agregan las réplicas 8 a 31 con las mismas semillas (7 + 1000 * i_phi + réplica) y se
comparan las 8 originales, las 24 nuevas y las 32 juntas.

Celdas: líder forzado (N = 250, 1000; ViSt, oráculo, sin memoria, adaptado) y maniobra
única (N = 250, 1000; tau_W = 1 y 4 s; CoLd, oráculo, sin memoria, adaptado). Referencia de
la recuperación: CoLd (con líder forzado, ViSt, que es la misma corrida).

Predicciones P1 a P3 y ancla A0 en docs/bitacora/resultados_E46-E55.md, antes de correr.

    PYTHONPATH=codigo python codigo/e48_replicas.py                  # todo
    PYTHONPATH=codigo E48_NS=250 python codigo/e48_replicas.py       # sólo N = 250
    PYTHONPATH=codigo E48_SOLO_ANALISIS=1 python codigo/e48_replicas.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from fase4 import (D_PHI, T_N, _w, trabajo, correr, clave, crudos_previos, pareado, fmt,
                   limpio)

NS = tuple(int(x) for x in os.environ.get("E48_NS", "250,1000").split(","))
SOLO_ANALISIS = bool(os.environ.get("E48_SOLO_ANALISIS"))
REPS_NUEVAS = range(8, 32)
CELDAS = [("forzado", False, 4.0, "ViSt"), ("maniobra", True, 4.0, "CoLd"),
          ("maniobra", True, 1.0, "CoLd")]
F_CRUDO = "resultados/E48_replicas_crudo.json"
F_JSON = "resultados/E48_replicas.json"
F_TXT = "resultados/E48_replicas.txt"
F_PARCIAL = F_CRUDO + ".parcial"


def trabajos(ns):
    phis = phi_grid(D_PHI)
    out = []
    for N in ns:
        for _, pulso, tau, ref in CELDAS:
            for nom in (ref, "oraculo", "amplitud", "adaptado"):
                for i, phi in enumerate(phis):
                    for rep in REPS_NUEVAS:
                        out.append(trabajo("base", nom, N, tau, pulso, i, phi, rep))
                # ancla: réplica 0 en dos ángulos (uno si N = 1000)
                for i in ((4, 13) if N == 250 else (9,)):
                    out.append(trabajo("ancla", nom, N, tau, pulso, i, phis[i], 0))
    out.sort(key=lambda a: a["N"] * a["T"])            # los cortos primero
    return out


def analizar(res):
    prev = crudos_previos()
    idx_prev = {clave(x): x for x in prev}
    L = ["=== E48 · 32 réplicas pareadas en las celdas de cabecera ==="]
    P = L.append
    out = {"config": {"replicas_nuevas": [8, 31], "semilla": "7 + 1000 * i_phi + réplica"},
           "ancla": {}, "celdas": {}, "pred": {}}

    # ---- ancla A0
    dm, n_cmp = 0.0, 0
    for x in res:
        if x["celda"] != "ancla":
            continue
        y = idx_prev.get(clave({**x, "celda": "base"}))
        if y is None:
            continue
        for w in x["r"]:
            if w in y["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w]))
                n_cmp += 1
    out["ancla"] = {"delta_max": dm, "comparaciones": n_cmp, "cumple": bool(n_cmp > 0 and dm < 1e-12)}
    P(f"ancla A0: réplica 0 vuelta a correr frente a los crudos guardados: |Δ| máx = {dm:.2e} "
      f"en {n_cmp} comparaciones")

    todo = prev + [x for x in res if x["celda"] == "base"]
    orig, nuevas = set(range(8)), set(REPS_NUEVAS)
    dentro, solapa, n_cifras = 0, 0, 0
    filas_p2 = {"a": [], "b": {}, "c": []}
    for N in (250, 1000):
        for etq, pulso, tau, ref in CELDAS:
            for titulo, win in (("corrida entera", T_N[N]), ("primeros 4 s", 4.0)):
                k = f"{etq}|{N}|{tau:g}|{_w(win)}"
                fila = {}
                for nom in ("adaptado", "amplitud"):
                    q8 = pareado(todo, "base", nom, ref, N, pulso, win, tau, orig)
                    q24 = pareado(todo, "base", nom, ref, N, pulso, win, tau, nuevas)
                    q32 = pareado(todo, "base", nom, ref, N, pulso, win, tau)
                    d = {"8": limpio(q8), "24": limpio(q24), "32": limpio(q32)}
                    if q24["n_rep"]:
                        d["dentro_ic8"] = bool(q8["ic95"][0] <= q32["rec"] <= q8["ic95"][1])
                        d["solapan_8_24"] = bool(q8["ic95"][0] <= q24["ic95"][1]
                                                 and q24["ic95"][0] <= q8["ic95"][1])
                        sa = 0.5 * (q8["ic95"][1] - q8["ic95"][0])
                        d["distancia_en_semianchuras"] = float(
                            abs(q32["rec"] - q8["rec"]) / sa) if sa > 0 else None
                    fila[nom] = d
                    P(f'{etq:9s} N = {N:4d} tau_W = {tau:g} s, {titulo:14s} {nom:9s} | '
                      f'8: {fmt(q8)} | 24 nuevas: {fmt(q24)} | 32: {fmt(q32)}'
                      + (f' | dentro del IC de 8: {"sí" if d.get("dentro_ic8") else "NO"}'
                         if q24["n_rep"] else ""))
                    if win == T_N[N] and q24["n_rep"]:
                        n_cifras += 1
                        dentro += d["dentro_ic8"]
                        solapa += d["solapan_8_24"]
                        if nom == "amplitud":
                            filas_p2["a"].append((k, q32["rec"], q32["ic95"][1]))
                        elif tau == 4.0:
                            filas_p2["b"][(etq, N)] = q32["rec"]
                        else:
                            filas_p2["c"].append((k, q32["rec"], q32["ic95"]))
                out["celdas"][k] = fila

    pr = out["pred"]
    pr["A0"] = out["ancla"]
    pr["P1"] = {"dentro": int(dentro), "de": int(n_cifras), "cumple": bool(n_cifras and dentro == n_cifras)}
    a = filas_p2["a"]
    b = filas_p2["b"]
    c = filas_p2["c"]
    pr["P2a"] = {"sin_memoria": [[k, r, s] for k, r, s in a],
                 "cumple": bool(a and all(abs(r) < 0.10 and s < 0.10 for _, r, s in a))}
    if len(b) == 4:
        pr["P2b"] = {"adaptado_tauW4": {f"{e}|{n}": v for (e, n), v in b.items()},
                     "cumple": bool(all(0.35 <= v <= 0.76 for v in b.values())
                                    and b[("forzado", 1000)] < b[("forzado", 250)]
                                    and b[("maniobra", 1000)] < b[("maniobra", 250)])}
    pr["P2c"] = {"adaptado_tauW1": [[k, r, ic] for k, r, ic in c],
                 "cumple": bool(c and all(r >= 0.95 and (ic[0] <= 1.0 <= ic[1] or ic[0] >= 0.95)
                                          for _, r, ic in c))}
    pr["P3"] = {"solapan": int(solapa), "de": int(n_cifras),
                "cumple": bool(n_cifras and solapa == n_cifras)}
    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        P(f'{k}: {"CUMPLE" if v.get("cumple") else "FALLA"}  {json.dumps(v, ensure_ascii=False)[:420]}')
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if SOLO_ANALISIS:
        f = F_CRUDO if os.path.exists(F_CRUDO) else F_PARCIAL
        with open(f) as fh:
            d = json.load(fh)
        res, seg = (d["resp"] if isinstance(d, dict) else d), 0.0
    else:
        res, seg = correr(trabajos(NS), F_PARCIAL, "E48")
        if os.path.exists(F_PARCIAL):            # el parcial acumula todas las pasadas
            with open(F_PARCIAL) as fh:
                res = json.load(fh)
        with open(F_CRUDO, "w") as fh:
            json.dump({"resp": res}, fh)
    out, L = analizar(res)
    L.append(f"\n{seg:.0f} s en esta pasada (N = {NS})")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
