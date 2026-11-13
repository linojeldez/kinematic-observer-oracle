"""Etapa 57b: potencia del contraste de E57 por inyección de trazas del modelo.

Diseño y predicción en docs/bitacora/resultados_E56-E65.md.

    python -I codigo/e57b_potencia.py
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np                                                   # noqa: E402

import watts_palomas as w                                            # noqa: E402
from e57_palomas import cinematica, residuos, ventana, auc_dentro, CONTROL, DT   # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASAS = (1.0, 3.0, 10.0)


def trazas():
    from e51b_inyeccion import traza
    with ProcessPoolExecutor(max_workers=8) as ex:
        tr = list(ex.map(traza, [("CoLd", r) for r in range(16)]))
    out = []
    for _, _, t in tr:
        t = np.array(t)
        out.append(np.interp(np.arange(0, 2.0, DT), np.arange(len(t)) * 0.01, t))
    return out


def una(args):
    f, r, lider, trz, semilla = args
    rng = np.random.default_rng(semilla)
    s = w.suelta(f, r, w.archivos())
    if s is None or lider not in s["aves"]:
        return None
    X, Y, th, om = cinematica(s)
    j = s["aves"].index(lider)
    eps0, _ = residuos(X, Y, th, om)
    validos = np.flatnonzero(np.isfinite(eps0[:, j]))
    if len(validos) < 100:
        return None
    n1 = int(round(1.0 / DT))
    p0, _ = auc_dentro(ventana(np.abs(eps0), n1), j)
    out = {"f": f, "r": r, "P0": p0, "minutos": len(validos) * DT / 60.0}
    for tasa in TASAS:
        n_ev = max(1, int(round(tasa * out["minutos"])))
        om2 = om.copy()
        for t0 in rng.choice(validos, size=n_ev, replace=False if n_ev <= len(validos) else True):
            tr = trz[int(rng.integers(len(trz)))] * rng.choice([-1.0, 1.0])
            a, b = t0, min(t0 + len(tr), len(om2))
            om2[a:b, j] += tr[:b - a]
        eps, _ = residuos(X, Y, th, om2)
        p, _ = auc_dentro(ventana(np.abs(eps), n1), j)
        out[f"P_{tasa:g}"] = p
    return out


if __name__ == "__main__":
    os.chdir(RAIZ)
    lideres = {int(k): v["lider"] for k, v in json.load(open("resultados/E57_palomas.json"))["lideres"].items()}
    trz = trazas()
    jobs = [(f, r, lideres[f], trz, 1000 * f + r) for f in sorted(lideres) for r in CONTROL]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [x for x in ex.map(una, jobs) if x]
    rng = np.random.default_rng(571)
    L = ["=== E57b · potencia del contraste de palomas (inyección en sueltas de control) ==="]
    out = {"tasas": {}}
    for tasa in TASAS:
        D = []
        for f in sorted(lideres):
            v = [x[f"P_{tasa:g}"] - x["P0"] for x in res if x["f"] == f]
            if v:
                D.append(float(np.mean(v)))
        D = np.array(D)
        b = D[rng.integers(0, len(D), (10000, len(D)))].mean(1)
        ic = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out["tasas"][f"{tasa:g}"] = {"D": float(D.mean()), "ic95": ic, "por_bandada": D.tolist()}
        L.append(f"{tasa:4.0f} trazas por minuto: D_inj = {D.mean():+.3f} [{ic[0]:+.3f}; {ic[1]:+.3f}] (8 bandadas)")
    L.append(f"minutos de vuelo con ≥ 3 aves por suelta de control: mediana {np.median([x['minutos'] for x in res]):.1f}")
    json.dump(out, open("resultados/E57b_potencia.json", "w"), indent=1)
    open("resultados/E57b_potencia.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
