"""Etapa 52b (plan 14, objeción mayor 6): grajillas con la grabación como unidad de remuestreo.

Mismos eventos, k y puntuaciones que E52 (k = 3, eventos con 1 s antes del inicio colectivo).
Unidad = grabación: los eventos de mobbing que son tramos de una misma bandada del archivo
continuo (mapa de E53) se remuestrean juntos; los de tránsito, sin mapa, como unidades
distintas. Diseño en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e52b_aves_por_grabacion.py
"""

import json
import os
import re

import numpy as np

import turns
from e52_auc_ventana_aves import por_evento, agrupada, auc

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_BOOT = 10000
rng = np.random.default_rng(521)
mapa = {}
for flock, tramos in json.load(open("resultados/E53_agilidad_calibracion.json"))["intervalos_curados"].items():
    for _, _, ev in tramos:
        mapa[ev] = flock
evs = [x for x in (por_evento(c, e, 3) for c, e in turns.eventos()) if x]
sel = [x for x in evs if np.isfinite(x["st"][1.0]).any()]


def unidad(x):
    num = re.findall(r"\d+", x["ev"])[-1]
    if x["ctx"] == "mobbing":
        return mapa.get(f"mobbing {int(num):02d}", x["ev"])
    return x["ev"]


uni = sorted({unidad(x) for x in sel})
grupos = [[i for i, x in enumerate(sel) if unidad(x) == u] for u in uni]
e52 = json.load(open("resultados/E52_auc_ventana_aves.json"))["k3|1|todos"]
a1, a0 = agrupada(sel, 1.0), agrupada(sel, "cuadro")
B = rng.integers(0, len(grupos), (N_BOOT, len(grupos)))
idxs = [[i for g in b for i in grupos[g]] for b in B]
b1 = np.array([agrupada(sel, 1.0, ix) for ix in idxs])
b0 = np.array([agrupada(sel, "cuadro", ix) for ix in idxs])
ic = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
loo = []
for j, u in enumerate(uni):
    ix = [i for g, gr in enumerate(grupos) if g != j for i in gr]
    loo.append({"fuera": u, "auc1": agrupada(sel, 1.0, ix), "ganancia": agrupada(sel, 1.0, ix) - agrupada(sel, "cuadro", ix)})
from e52_auc_ventana_aves import VENTANAS
curva = {}
for w in [w for w in VENTANAS if w == "cuadro" or w <= 1.0]:
    bw = np.array([agrupada(sel, w, ix) for ix in idxs])
    curva[str(w)] = {"auc": agrupada(sel, w), "ic95": ic(bw)}
pe1 = [auc(x["st"][1.0][x["et"] == 1], x["st"][1.0][x["et"] == 0]) for x in sel]
pe0 = [auc(x["st"]["cuadro"][x["et"] == 1], x["st"]["cuadro"][x["et"] == 0]) for x in sel]
out = {"eventos": [x["ev"] for x in sel], "unidades": {u: [sel[i]["ev"] for i in g] for u, g in zip(uni, grupos)},
       "ancla": {"auc1": [a1, e52["curva"]["1.0"]["auc"]], "cumple": bool(abs(a1 - e52["curva"]["1.0"]["auc"]) < 1e-12)},
       "auc1": a1, "auc1_ic95": ic(b1), "auc_cuadro": a0, "auc_cuadro_ic95": ic(b0),
       "ganancia": a1 - a0, "ganancia_ic95": ic(b1 - b0), "frac_boot_positiva": float(np.nanmean(b1 - b0 > 0)),
       "dejando_una_fuera": loo, "curva": curva,
       "por_evento_igual_peso": {"auc1": float(np.nanmean(pe1)), "auc_cuadro": float(np.nanmean(pe0)),
                                 "ganancia": float(np.nanmean(np.array(pe1) - np.array(pe0))),
                                 "auc1_por_evento": [float(v) for v in pe1], "auc_cuadro_por_evento": [float(v) for v in pe0]}}
L = ["=== E52b · grajillas, remuestreo por grabación (k = 3, ventana de 1 s) ===",
     f"ancla frente a E52: AUC(1 s) {a1:.6f} / {e52['curva']['1.0']['auc']:.6f}",
     f"{len(sel)} eventos en {len(uni)} unidades: " + "; ".join(f"{u}: {', '.join(v)}" for u, v in out["unidades"].items()),
     f"AUC(1 s) {a1:.3f} {[round(v, 3) for v in out['auc1_ic95']]} | AUC(cuadro) {a0:.3f} {[round(v, 3) for v in out['auc_cuadro_ic95']]} | "
     f"ganancia {a1 - a0:+.3f} {[round(v, 3) for v in out['ganancia_ic95']]} (positiva en {100 * out['frac_boot_positiva']:.1f} %)",
     "dejando una unidad fuera: " + "; ".join(f"{d['fuera']}: AUC {d['auc1']:.3f}, ganancia {d['ganancia']:+.3f}" for d in loo),
     f"promedio por evento con igual peso: AUC(1 s) {out['por_evento_igual_peso']['auc1']:.3f}, AUC(cuadro) "
     f"{out['por_evento_igual_peso']['auc_cuadro']:.3f}, ganancia {out['por_evento_igual_peso']['ganancia']:+.3f}"]
json.dump(out, open("resultados/E52b_aves_por_grabacion.json", "w"), indent=1, default=float)
open("resultados/E52b_aves_por_grabacion.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
