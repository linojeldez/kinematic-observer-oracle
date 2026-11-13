"""Etapa 45: control de ruido de la huella.

Por qué. En CaSp el motor no lee el will, así que el agente «informado» es un agente
cualquiera y su pico es el máximo de un registro de ruido de 5 s. Aquí se calcula el
mismo estadístico (pico y duración de `e15_helpers.huella`, sin cambios) para los 200
agentes de las 48 corridas de la figura de la huella, y se compara el etiquetado con
los que no llevan etiqueta.

Predicciones C0 a C4 en docs/bitacora/resultados_E36-E45.md (declaradas antes de correr).

    PYTHONPATH=codigo python codigo/e45_control_huella.py      (unos 20 s, 6 procesos)
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from flocking import Flock, Params
from e15_helpers import huella

CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
BASE = dict(N=200, k=12, sigma_xi=np.sqrt(2.0), seed=11)
MODELOS = ("WiSt", "CoLd", "CaSp")
N_REP = 16
AGENTE, ANG, T_BURN, T_OBS, UMBRAL = 7, 2.0, 6.0, 5.0, 1.5
F_JSON = "resultados/E45_control_huella.json"
F_TXT = "resultados/E45_control_huella.txt"


def _estadisticos(t, y, sig):
    """Pico y duración de una serie, con el algoritmo de e15_helpers.huella."""
    thr = UMBRAL * max(sig, 1e-9)
    w = max(3, len(y) // 25)
    sm = np.convolve(y, np.ones(w) / w, mode="same")
    i_pico = int(np.argmax(sm))
    d = float(t[-1])
    for j in range(i_pico, len(sm)):
        if sm[j] < thr and np.all(sm[j:j + w] < thr):
            d = float(t[j])
            break
    return float(sm.max() / max(sig, 1e-9)), d


def uno(arg):
    m, rep = arg
    q = Params(**{**BASE, **CFG, "seed": BASE["seed"] + rep})
    ref = huella(q, m, n_rep=1)

    f = Flock(q, m)                                   # misma secuencia que huella()
    for _ in range(int(round(T_BURN / q.dt))):
        f.step()
    base = []
    for _ in range(int(round(2.0 / q.dt))):
        f.step()
        i, _ = f._neighbors()
        base.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    sig = float(np.median(np.concatenate(base)) * 1.4826)
    f.will[AGENTE] = 1.0
    f.d_des[AGENTE] = np.array([np.cos(ANG), np.sin(ANG)])
    t, E = [], []
    for s in range(int(round(T_OBS / q.dt))):
        f.step()
        f.will[AGENTE] = 1.0
        if s % 5 == 0:
            i, _ = f._neighbors()
            t.append(s * q.dt)
            E.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    t, E = np.array(t), np.array(E)                   # (T,), (T, N)
    est = [_estadisticos(t, E[:, a], sig) for a in range(q.N)]
    pico = np.array([e[0] for e in est])
    dur = np.array([e[1] for e in est])
    return {"model": m, "rep": rep, "pico": pico.tolist(), "dur": dur.tolist(),
            "ancla": [abs(pico[AGENTE] - ref["pico_rel"]), abs(dur[AGENTE] - ref["dur"])]}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t0 = time.time()
    with Pool(6) as pool:
        res = pool.map(uno, [(m, r) for m in MODELOS for r in range(N_REP)])
    otros = np.arange(BASE["N"]) != AGENTE
    out = {"config": {**{k: (float(v) if not isinstance(v, int) else v) for k, v in BASE.items()},
                      **CFG, "replicas": N_REP, "agente": AGENTE, "ang": ANG},
           "modelos": {}, "pred": {}}
    L = ["=== E45 · control de ruido de la huella ==="]
    P = L.append
    ancla = max(max(r["ancla"]) for r in res)
    P(f"ancla (etiquetado frente a e15_helpers.huella, 48 corridas): |Δ| máx = {ancla:.2e}")
    for m in MODELOS:
        rs = sorted([r for r in res if r["model"] == m], key=lambda r: r["rep"])
        pico = np.array([r["pico"] for r in rs])        # (rep, N)
        dur = np.array([r["dur"] for r in rs])
        pe, de = pico[:, AGENTE], dur[:, AGENTE]
        po, do = pico[:, otros], dur[:, otros]
        rango = (po < pe[:, None]).mean(axis=1)         # rango percentil del etiquetado
        p95 = np.percentile(po, 95, axis=1)
        fila = {
            "etiquetado": {"pico_mediana": float(np.median(pe)), "dur_mediana": float(np.median(de)),
                           "dur_sd": float(de.std()), "pico_min": float(pe.min()),
                           "pico_max": float(pe.max()), "sobre_umbral": int((pe > UMBRAL).sum()),
                           "primeras8": {"pico": float(np.median(pe[:8])),
                                         "dur": float(np.median(de[:8])),
                                         "dur_sd": float(de[:8].std())}},
            "sin_etiqueta": {"pico_mediana": float(np.median(po)),
                             "pico_p05": float(np.percentile(po, 5)),
                             "pico_p95": float(np.percentile(po, 95)),
                             "sobre_umbral_frac": float((po > UMBRAL).mean()),
                             "dur_mediana": float(np.median(do)), "dur_sd": float(do.std()),
                             "pico_mediana_por_replica": np.median(po, axis=1).tolist()},
            "rango_etiquetado": {"media": float(rango.mean()), "por_replica": rango.tolist()},
            "sobre_p95": int((pe > p95).sum()),
        }
        out["modelos"][m] = fila
        e, o = fila["etiquetado"], fila["sin_etiqueta"]
        P(f"\n{m}")
        P(f"  etiquetado   : pico {e['pico_mediana']:.2f} σ [{e['pico_min']:.2f}; {e['pico_max']:.2f}], "
          f"sobre 1,5 σ en {e['sobre_umbral']}/16; duración {e['dur_mediana']:.3f} ± {e['dur_sd']:.3f} s")
        P(f"    primeras 8 réplicas: pico {e['primeras8']['pico']:.4f} σ, duración "
          f"{e['primeras8']['dur']:.3f} ± {e['primeras8']['dur_sd']:.3f} s")
        P(f"  sin etiqueta : pico {o['pico_mediana']:.2f} σ (p5 {o['pico_p05']:.2f}; p95 {o['pico_p95']:.2f}), "
          f"sobre 1,5 σ el {100 * o['sobre_umbral_frac']:.0f} %; duración {o['dur_mediana']:.2f} ± {o['dur_sd']:.2f} s")
        P(f"  rango percentil del etiquetado: media {rango.mean():.3f}; sobre el p95 en "
          f"{fila['sobre_p95']}/16 réplicas")

    M = out["modelos"]
    pr = out["pred"]
    pr["C0"] = {"delta_max": ancla, "cumple": bool(ancla == 0.0)}
    c1 = M["CaSp"]["sin_etiqueta"]["pico_mediana"]
    pr["C1"] = {"sin_etiqueta": c1, "etiquetado": M["CaSp"]["etiquetado"]["pico_mediana"],
                "cumple": bool(1.45 <= c1 <= 1.75
                               and abs(c1 - M["CaSp"]["etiquetado"]["pico_mediana"]) < 0.15)}
    c2 = M["CaSp"]["rango_etiquetado"]["media"]
    pr["C2"] = {"rango_medio": c2, "cumple": bool(0.35 <= c2 <= 0.65)}
    pr["C3"] = {"WiSt": M["WiSt"]["sobre_p95"], "CoLd": M["CoLd"]["sobre_p95"],
                "cumple": bool(M["WiSt"]["sobre_p95"] >= 14 and M["CoLd"]["sobre_p95"] >= 14)}
    c4 = M["CaSp"]["sin_etiqueta"]
    pr["C4"] = {"dur_mediana": c4["dur_mediana"], "dur_sd": c4["dur_sd"],
                "cumple": bool(0.5 <= c4["dur_mediana"] <= 2.5 and c4["dur_sd"] > 1.0)}
    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        P(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v, ensure_ascii=False)}')
    out["segundos"] = time.time() - t0
    P(f"\n{out['segundos']:.0f} s")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
