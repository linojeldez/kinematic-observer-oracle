"""Plan 14 (objeciones mayores 2 y 6): responsividades absolutas de las celdas de cabecera.

Desde los crudos de E48 (32 réplicas pareadas): R de la referencia (ViSt con líder forzado, CoLd
con maniobra única), del oráculo, del observador de 1 s y del sin memoria, sobre la ventana
completa sqrt(N) s, con IC 95 % por remuestreo de réplicas de las diferencias pareadas.

    PYTHONPATH=codigo python codigo/absolutos_E48.py
"""

import json
import os

import numpy as np

from fase4 import R_rep, clave, crudos_previos, T_N

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
prev = crudos_previos()
nuevas = json.load(open("resultados/E48_replicas_crudo.json"))["resp"]
todo = list({clave(x): x for x in prev + [{**x, "celda": "base"} for x in nuevas]}.values())
CELDAS = [("forzado", False, "ViSt", 250, 4.0), ("maniobra", True, "CoLd", 250, 4.0), ("maniobra", True, "CoLd", 250, 1.0),
          ("forzado", False, "ViSt", 1000, 4.0), ("maniobra", True, "CoLd", 1000, 4.0), ("maniobra", True, "CoLd", 1000, 1.0)]
rng = np.random.default_rng(48)
out, L = {}, ["=== Responsividades absolutas (ventana completa, réplicas pareadas) ==="]
for etq, pulso, ref, N, tw in CELDAS:
    win = T_N[N]
    S = {nom: dict(zip(*R_rep(todo, "base", nom, N, pulso, win, tw))) for nom in (ref, "oraculo", "adaptado", "amplitud")}
    com = sorted(set.intersection(*[set(v) for v in S.values()]))
    A = {nom: np.array([S[nom][k] for k in com]) for nom in S}
    B = rng.integers(0, len(com), (4000, len(com)))
    ic = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    c = {"n_rep": len(com), "R_ref": float(A[ref].mean()), "R_oraculo": float(A["oraculo"].mean()),
         "R_observador": float(A["adaptado"].mean()), "R_sin_memoria": float(A["amplitud"].mean())}
    for nom, k in (("oraculo", "ventaja_oraculo"), ("adaptado", "ganancia_observador"), ("amplitud", "ganancia_sin_memoria")):
        d = A[nom] - A[ref]
        c[k] = float(d.mean()); c[k + "_ic95"] = ic(d[B].mean(axis=1))
    d = A["adaptado"] - A["oraculo"]
    c["observador_menos_oraculo"] = float(d.mean()); c["observador_menos_oraculo_ic95"] = ic(d[B].mean(axis=1))
    c["rec"] = c["ganancia_observador"] / c["ventaja_oraculo"]
    out[f"{etq}|{N}|{tw:g}"] = c
    L.append(f"{etq:9s} N={N:4d} tau_W={tw:g}: R ref {c['R_ref']:.3f} | oráculo {c['R_oraculo']:.3f} | observador {c['R_observador']:.3f} | "
             f"sin memoria {c['R_sin_memoria']:.3f} | ventaja {c['ventaja_oraculo']:.3f} {c['ventaja_oraculo_ic95']} | "
             f"obs−ref {c['ganancia_observador']:.3f} | obs−oráculo {c['observador_menos_oraculo']:+.3f} {c['observador_menos_oraculo_ic95']} | "
             f"rec {c['rec']:.3f} (n = {c['n_rep']})")
json.dump(out, open("resultados/absolutos_E48.json", "w"), indent=1)
open("resultados/absolutos_E48.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
