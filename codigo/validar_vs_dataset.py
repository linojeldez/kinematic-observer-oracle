"""Valida el motor contra las respuestas R(k) publicadas con el modelo de referencia.

Es la tabla S2 del Paper A corto. Protocolo de la referencia: N = 250, sigma^2 = 1 m^2 s^-3,
un líder forzado, t_R = 15,8 s, rejilla de 18 ángulos y 3 réplicas por ángulo (54 corridas
por modelo y k; semilla 1000 i_phi + réplica), con k = 5, 15, 30 y 70.

Los valores publicados son los del fichero `v-w-resp-kscan-250A-sqrttime.p` del conjunto de
datos de la referencia (Mendeley Data, doi:10.17632/3brbybg4nk.1), copiados aquí con toda su
precisión. Si ese fichero está en `datos/`, se comprueban contra él.

Escribe `resultados/validacion_vs_dataset.json` y `.txt` con las cifras sin redondear.

    PYTHONPATH=codigo python codigo/validar_vs_dataset.py
"""
import json
import os
import pickle
import time

import numpy as np

from flocking import Params, sweep_responsivity

N, SIGMA_XI, T_R, D_PHI, N_REP, SEMILLA = 250, 1.0, 15.8, 10.0, 3, 0
MODELOS = ("ViSt", "WiSt")
REF = {  # k -> (ViSt, WiSt), del conjunto de datos publicado
    5: (0.09353565263157891, 0.1767614315789474),
    15: (0.10416592631578946, 0.4433074210526315),
    30: (0.09990604210526313, 0.7199129578947367),
    70: (0.09865192631578946, 0.898946894736842),
}
F_REF = os.path.join(
    "datos", "persistence-responsivity trade off",
    "Data for Adaptive leadership overcomes persistence-responsivity tradeoff in flocking",
    "Figure 2 & 3 - Responsivity vs k", "v-w-resp-kscan-250A-sqrttime.p")
F_JSON, F_TXT = "resultados/validacion_vs_dataset.json", "resultados/validacion_vs_dataset.txt"
W = int(os.environ.get("FASE4_W", "8"))


class _SinClases(pickle.Unpickler):
    """El fichero publicado sólo contiene listas, diccionarios, cadenas y números."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"clase no permitida: {module}.{name}")


def referencia_del_fichero(ruta=F_REF):
    """{(modelo, k): R} leído del fichero publicado, o None si no está."""
    if not os.path.exists(ruta):
        return None
    with open(ruta, "rb") as fh:
        filas = _SinClases(fh, encoding="latin1").load()
    nombre = {"V": "ViSt", "W": "WiSt"}
    return {(nombre[f["modelname"]], int(f["neigh_k"])): float(f["data"]) for f in filas}


if __name__ == "__main__":
    t0 = time.time()
    ks = sorted(REF)
    base = Params(N=N, sigma_xi=SIGMA_XI, seed=SEMILLA)
    R = {}
    for modelo in MODELOS:
        filas = sweep_responsivity(base, [modelo], ks, n_rep=N_REP, d_phi_deg=D_PHI,
                                   t_R=T_R, workers=W)
        for k in ks:
            por_phi = {}
            for x in filas:
                if x["k"] == k:
                    por_phi.setdefault(round(x["phi"], 6), []).append(float(x["r"]))
            medias = [float(np.mean(v)) for _, v in sorted(por_phi.items())]
            R[f"{modelo}|{k}"] = {"R": float(np.mean(medias)),
                                  "corridas": sum(len(v) for v in por_phi.values()),
                                  "por_angulo": medias}

    ref = {f"{m}|{k}": REF[k][i] for k in ks for i, m in enumerate(MODELOS)}
    desvio = {c: R[c]["R"] / ref[c] - 1.0 for c in ref}
    leida = referencia_del_fichero()
    if leida is None:
        comprobada = None
    else:
        comprobada = max(abs(leida[(m, k)] - REF[k][i]) for k in ks for i, m in enumerate(MODELOS))

    sal = {"protocolo": {"N": N, "sigma_xi": SIGMA_XI, "t_R": T_R, "d_phi_grados": D_PHI,
                         "replicas": N_REP, "semilla": SEMILLA, "k": ks, "modelos": list(MODELOS)},
           "referencia": ref, "referencia_dif_max_con_el_fichero": comprobada,
           "R": R, "desvio": desvio, "segundos": round(time.time() - t0)}
    with open(F_JSON, "w") as fh:
        json.dump(sal, fh, indent=1)

    lineas = [f"=== Validación del motor contra el conjunto de datos publicado (N = {N}, "
              f"sigma^2 = {SIGMA_XI:g}, t_R = {T_R} s, {N_REP} réplicas por ángulo, "
              f"{R['ViSt|5']['corridas']} corridas por modelo y k) ===",
              f"{'k':>4s} | {'ViSt motor':>10s} {'ViSt publ.':>10s} {'desvío':>9s} | "
              f"{'WiSt motor':>10s} {'WiSt publ.':>10s} {'desvío':>9s}"]
    for k in ks:
        v, w = f"ViSt|{k}", f"WiSt|{k}"
        lineas.append(f"{k:4d} | {R[v]['R']:10.6f} {ref[v]:10.6f} {100 * desvio[v]:+8.2f} % | "
                      f"{R[w]['R']:10.6f} {ref[w]:10.6f} {100 * desvio[w]:+8.2f} %")
    lineas.append("")
    if comprobada is None:
        lineas.append("valores publicados: los de este script (el fichero del conjunto de datos "
                      "no está en datos/)")
    else:
        lineas.append(f"valores publicados comprobados contra el fichero: |Δ| máx = {comprobada:.2e}")
    lineas.append(f"\n{sal['segundos']} s")
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))
