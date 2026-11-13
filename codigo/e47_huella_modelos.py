"""Etapa 47: la huella frente a la amplitud del giro y a los parámetros, en CoLd y en WiSt.

Por qué. La serie de saturación con la amplitud y la tabla de invariancia del paper
corto eran de CoLd, con 8, 8, 6 y 6 réplicas. Aquí se miden en los dos modelos con
informados, con más réplicas y con intervalo, y se deja la comparación con la forma
cerrada de la derivación.

Diseño. `e15_helpers.huella` sin cambios; N = 200, k = 12, sigma_xi = sqrt(2), semilla
11 + réplica; intención a 115° salvo en el barrido de amplitud.
  - amplitud: 57°, 115°, 153°, 172°; 16 réplicas (las de la tabla de la huella);
  - barridos de tau_c, k y sigma_xi; 32 réplicas; cambio = mediana mayor / mediana menor,
    IC 95 % por remuestreo de réplicas (2000);
  - forma cerrada t = (2/w_max) ln[sec(d/2) + tan(d/2)], d = min(amplitud, 129°).

Anclas H0a-H0c y H1 en docs/bitacora/resultados_E36-E45.md.

    PYTHONPATH=codigo python codigo/e47_huella_modelos.py      (unos 3 min, 8 procesos)
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from flocking import Params
from e15_helpers import huella

CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
BASE = dict(N=200, k=12, sigma_xi=np.sqrt(2.0), seed=11)
MODELOS = ("CoLd", "WiSt")
ANGULOS = (1.0, 2.0, 2.67, 3.0)                        # rad: 57°, 115°, 153°, 172°
N_AMP, N_BARR, N_BOOT = 16, 32, 2000
BARR = {"tau_c": (0.05, 0.1, 0.2, 0.4, 0.8, 1.6), "k": (5, 10, 20, 40),
        "sigma_xi": (0.5, 1.0, np.sqrt(2.0), 2.0)}
D_VIS = 129.0
F_JSON = "resultados/E47_huella_modelos.json"
F_TXT = "resultados/E47_huella_modelos.txt"


def uno(a):
    m, par, val, rep = a
    kw = {**BASE, **CFG, "seed": BASE["seed"] + rep}
    ang = 2.0
    if par == "ang":
        ang = val
    else:
        kw[par] = val
    return m, par, float(val), rep, float(huella(Params(**kw), m, ang=ang, n_rep=1)["dur"])


def cerrada(grados, w_max=10.0 / 8.0):
    d = np.deg2rad(min(grados, D_VIS)) / 2.0
    return float((2.0 / w_max) * np.log(1.0 / np.cos(d) + np.tan(d)))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t0 = time.time()
    tareas = [(m, "ang", a, r) for m in MODELOS for a in ANGULOS for r in range(N_AMP)]
    tareas += [(m, par, v, r) for m in MODELOS for par, vs in BARR.items() for v in vs
               for r in range(N_BARR)]
    with Pool(8) as pool:
        res = pool.map(uno, tareas, chunksize=4)

    def serie(m, par, v):
        return np.array([x[4] for x in res if x[0] == m and x[1] == par and abs(x[2] - float(v)) < 1e-9])

    out = {"config": {**{k: (float(v) if not isinstance(v, int) else v) for k, v in BASE.items()},
                      **CFG, "replicas_amplitud": N_AMP, "replicas_barridos": N_BARR,
                      "remuestreos": N_BOOT, "d_vis_grados": D_VIS},
           "amplitud": {}, "barridos": {}, "forma_cerrada": {}, "anclas": {}}
    L = ["=== E47 · la huella frente a la amplitud y a los parámetros, en CoLd y en WiSt ==="]
    P = L.append

    P(f"\n--- amplitud ({N_AMP} réplicas): mediana ± desviación (s) ---")
    for m in MODELOS:
        out["amplitud"][m] = {}
        for a in ANGULOS:
            d = serie(m, "ang", a)
            g = float(np.rad2deg(a))
            out["amplitud"][m][f"{g:.0f}"] = {"mediana": float(np.median(d)), "sd": float(d.std()),
                                               "mediana_8": float(np.median(d[:8])), "n": int(len(d))}
        P(f"  {m}: " + "  ".join(f'{g}°: {v["mediana"]:.3f} ± {v["sd"]:.3f}' for g, v in out["amplitud"][m].items()))

    P(f"\n--- forma cerrada frente a CoLd ({N_AMP} réplicas) ---")
    for a in ANGULOS:
        g = float(np.rad2deg(a))
        med = out["amplitud"]["CoLd"][f"{g:.0f}"]["mediana"]
        out["forma_cerrada"][f"{g:.0f}"] = {"cerrada": cerrada(g), "medido": med, "cociente": cerrada(g) / med}
        P(f"  {g:5.0f}°: cerrada {cerrada(g):.2f} s; medido {med:.3f} s; cociente {cerrada(g) / med:.2f}")

    P(f"\n--- barridos ({N_BARR} réplicas): medianas (s) y cambio máx/mín con IC 95 % ---")
    rng = np.random.default_rng(1)
    for m in MODELOS:
        out["barridos"][m] = {}
        for par, vs in BARR.items():
            D = {float(v): serie(m, par, v) for v in vs}
            med = {v: float(np.median(d)) for v, d in D.items()}
            cambio = max(med.values()) / min(med.values())
            bs = []
            for _ in range(N_BOOT):
                mb = [np.median(d[rng.integers(0, len(d), len(d))]) for d in D.values()]
                bs.append(max(mb) / max(min(mb), 1e-9))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out["barridos"][m][par] = {"valores": [float(v) for v in vs], "medianas": list(med.values()),
                                       "medianas_8": [float(np.median(d[:8])) for d in D.values()],
                                       "rango": float(max(vs) / min(vs)), "cambio": float(cambio),
                                       "ic95": [float(lo), float(hi)]}
            P(f"  {m} {par:<9s} {[round(x, 3) for x in med.values()]}  x{cambio:.2f} [{lo:.2f}; {hi:.2f}]")

    # ---- anclas
    e16 = json.load(open("resultados/E16_formula.json"))["comparacion"]
    e15 = json.load(open("resultados/E15_clase.json"))["tau_c"]
    a = out["anclas"]
    a["H0a"] = {"delta_max": float(max(abs(v["mediana_8"] - c["med"]) for v, c in
                                       zip(out["amplitud"]["CoLd"].values(), e16)))}
    a["H0b"] = {"delta_max": float(max(abs(x - c["dur"]) for x, c in
                                       zip(out["barridos"]["CoLd"]["tau_c"]["medianas_8"], e15)))}
    a["H0c"] = {"WiSt_115": out["amplitud"]["WiSt"]["115"]["mediana"],
                "CoLd_115": out["amplitud"]["CoLd"]["115"]["mediana"]}
    for k in ("H0a", "H0b"):
        a[k]["cumple"] = bool(a[k]["delta_max"] == 0.0)
    a["H0c"]["cumple"] = bool(a["H0c"]["WiSt_115"] == 0.875 and a["H0c"]["CoLd_115"] == 1.375)
    esp_amp = {"CoLd": (0.775, 1.375, 1.475, 1.450), "WiSt": (0.500, 0.875, 1.175, 1.600)}
    esp_cam = {"CoLd": (1.30, 1.06, 1.18), "WiSt": (1.85, 1.14, 1.80)}
    a["H1"] = {"cumple": bool(
        all(abs(v["mediana"] - e) < 1e-9 for m in MODELOS
            for v, e in zip(out["amplitud"][m].values(), esp_amp[m]))
        and all(abs(out["barridos"][m][par]["cambio"] - e) < 0.005 for m in MODELOS
                for par, e in zip(BARR, esp_cam[m])))}
    P("\n=== Anclas ===")
    for k, v in a.items():
        P(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v)}')
    out["segundos"] = time.time() - t0
    P(f"\n{out['segundos']:.0f} s")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
