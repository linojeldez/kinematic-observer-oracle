"""Etapa 71 (plan 16, R1 y R4 del revisor): integrador frente a sin memoria con el mismo presupuesto
de ajuste y semillas nuevas; barrido de la ventana en el régimen de la cabecera.

N = 250 con 1, 3, 12 y 25 informados (prior = fracción), tres celdas. Entrenamiento: réplicas 0-7
(las de E68); prueba: réplicas 8-15 (semillas e informados nuevos). Con el entrenamiento se elige la
ganancia del sin memoria (x1, x6, x12; en la celda central x1, x3, x6, x12, x24) y la ventana del
integrador (0,5, 1, 2 s; en la central 0,25, 0,5, 1, 2, 4 s). Con la prueba se evalúan el integrador
de 1 s (sin ajuste), el integrador con la ventana elegida y el sin memoria con la ganancia elegida.
N = 1000 con 48 informados: sin memoria x6 y x12 en las réplicas 0-7; la ganancia se elige en la
celda central de N = 250. R4: celda central con ventanas 0,1-8 s y 16 réplicas.

Diseño y predicciones J1-J4, W1-W2 en docs/bitacora/resultados_E66-E75.md («Plan 16»).

    PYTHONPATH=codigo python codigo/e71_comparacion_justa.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from fase4 import D_PHI, T_N, ARM
from e68_fraccion_informados import job, clave, informados, CELDAS
from plan16_comun import correr, series, rec_boot, guardar

GRUPOS = ((250, 1), (250, 3), (250, 12), (250, 25))
CENTRAL = (250, 12)
ENT, PRU = list(range(8)), list(range(8, 16))
VENT_R4 = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
F_CRUDO, F_JSON, F_TXT = ("resultados/E71_comparacion_justa_crudo.json", "resultados/E71_comparacion_justa.json",
                          "resultados/E71_comparacion_justa.txt")


def nom_v(v):
    return "integrador" if v == 1.0 else f"integrador_{v:g}s"


def nom_g(g):
    return "sin_memoria" if g == 1.0 else f"sin_memoria_x{g:g}"


def opciones(N, n):
    """(ventanas para elegir, ganancias para elegir, ventanas que se corren)."""
    if (N, n) == CENTRAL:
        return (0.25, 0.5, 1.0, 2.0, 4.0), (1.0, 3.0, 6.0, 12.0, 24.0), VENT_R4
    return (0.5, 1.0, 2.0), (1.0, 6.0, 12.0), (0.5, 1.0, 2.0)


def trabajos():
    phis = phi_grid(D_PHI)
    out = []

    def add(N, n, etq, pulso, tw, nom, base, kw, fac, reps):
        model, kw0 = ARM[base]
        for i, phi in enumerate(phis):
            for rep in reps:
                out.append({"celda": f"n{n}", "nom": nom, "model": model, "kw": {**kw0, **kw}, "factor": fac,
                            "N": N, "tau_W": tw, "pulso": pulso, "i_phi": i, "phi": float(phi), "rep": rep,
                            "T": T_N[N], "inf": informados(N, n, rep)})

    todas = ENT + PRU
    for N, n in GRUPOS:
        pi = n / N
        _, gan, vent = opciones(N, n)
        for etq, pulso, tw in CELDAS:
            add(N, n, etq, pulso, tw, "CoLd", "CoLd", {}, 1.0, todas)
            add(N, n, etq, pulso, tw, "oraculo", "oraculo", {}, 1.0, todas)
            for v in vent:
                add(N, n, etq, pulso, tw, nom_v(v), "adaptado", {"pi0": pi, "tau_det": v}, 1.0, todas)
            for g in gan:
                add(N, n, etq, pulso, tw, nom_g(g), "amplitud", {"pi0": pi}, g, todas)
    for etq, pulso, tw in CELDAS:                       # N = 1000: las ganancias que faltaban
        for g in (6.0, 12.0):
            add(1000, 48, etq, pulso, tw, nom_g(g), "amplitud", {"pi0": 48 / 1000}, g, ENT)
    out.sort(key=lambda a: -a["N"])
    return out


def previos():
    return json.load(open("resultados/E68_fraccion_informados_crudo.json"))["resp"]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = trabajos()
    # ancla: re-correr 2 ángulos de la réplica 0 de cada grupo y celda (integrador y sin memoria x6)
    prev = previos()
    propios = json.load(open(F_CRUDO))["resp"] if os.path.exists(F_CRUDO) else []   # E71 ya corrida
    ancla = [dict(a, celda="ancla_" + a["celda"]) for a in jobs
             if a["rep"] == 0 and a["i_phi"] in (4, 13) and a["N"] == 250 and a["nom"] in ("integrador", "sin_memoria_x6")]
    res, nuevos, seg = correr(jobs + ancla, F_CRUDO + ".parcial", "E71", job, clave, previos=prev + propios)
    union = {clave(x): x for x in propios + nuevos}
    json.dump({"resp": list(union.values())}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    if propios and not nuevos:
        seg = 10134.0                                   # duración de la corrida original (análisis rehecho)
    idx = {clave(x): x for x in prev}
    dm, nc = 0.0, 0
    for x in res:
        if x["celda"].startswith("ancla_"):
            y = idx[clave({**x, "celda": x["celda"][6:]})]
            for w in x["r"]:
                dm = max(dm, abs(x["r"][w] - y["r"][w])); nc += 1
    # E68 entra completo: las condiciones de N = 1000 que E71 no repite (CoLd, oráculo, integrador, x1) vienen de ahí
    todo = list({clave(x): x for x in prev + [y for y in res if not y["celda"].startswith("ancla_")]}.values())
    w = T_N[250]
    out = {"ancla": {"delta_max": dm, "comparaciones": nc, "cumple": bool(nc > 0 and dm < 1e-12)},
           "eleccion": {}, "prueba": {}, "dif": {}, "R4": {}, "N1000": {}}
    L = ["=== E71 · comparación con el mismo presupuesto de ajuste (elección con réplicas 0-7, prueba 8-15) ===",
         f"ancla (frente a E68): |Δ| máx = {dm:.2e} en {nc} comparaciones"]
    for N, n in GRUPOS:
        ven, gan, vent = opciones(N, n)
        for etq, pulso, tw in CELDAS:
            c = f"N{N}|n{n}|{etq}|{tw:g}"
            ref = "CoLd"
            obs_v, obs_g = [nom_v(v) for v in ven], [nom_g(g) for g in gan]
            _, S = series(todo, f"n{n}", ["CoLd", "oraculo"] + obs_v + obs_g, N, pulso, tw, w, ENT)
            qe, _ = rec_boot(S, ref, obs_v + obs_g)
            v_star = max(obs_v, key=lambda k: qe[k]["rec"])
            g_star = max(obs_g, key=lambda k: qe[k]["rec"])
            out["eleccion"][c] = {"ventana": v_star, "ganancia": g_star,
                                  "rec_ent": {k: qe[k]["rec"] for k in obs_v + obs_g}}
            com, S = series(todo, f"n{n}", ["CoLd", "oraculo"] + obs_v + obs_g, N, pulso, tw, w, PRU)
            qp, dif = rec_boot(S, ref, obs_v + obs_g)
            out["prueba"][c] = qp
            g_max_p = max(obs_g, key=lambda k: qp[k]["rec"])
            out["dif"][c] = {"int1_vs_gstar": dif("integrador", g_star), "vstar_vs_gstar": dif(v_star, g_star),
                             "vstar_vs_int1": dif(v_star, "integrador"), "int1_vs_gmax_prueba": dif("integrador", g_max_p),
                             "gmax_prueba": g_max_p}
            d1 = out["dif"][c]["int1_vs_gstar"]
            L.append(f"{c:24s} n = {len(com)} | elegidos: {v_star}, {g_star} | prueba: integrador 1 s "
                     f"{qp['integrador']['rec']:.3f} [{qp['integrador']['ic95'][0]:.3f}; {qp['integrador']['ic95'][1]:.3f}], "
                     f"{v_star} {qp[v_star]['rec']:.3f}, {g_star} {qp[g_star]['rec']:.3f} "
                     f"[{qp[g_star]['ic95'][0]:.3f}; {qp[g_star]['ic95'][1]:.3f}] | int1 − sin memoria elegido "
                     f"{d1['dif']:+.3f} [{d1['ic95'][0]:+.3f}; {d1['ic95'][1]:+.3f}]")
    # R4: celda central, 16 réplicas, ventanas 0,1-8 s
    L.append("\n=== R4 · ventana en la celda central (12 de 250, prior 12/250), 16 réplicas ===")
    for etq, pulso, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        noms = [nom_v(v) for v in VENT_R4]
        com, S = series(todo, "n12", ["CoLd", "oraculo"] + noms, 250, pulso, tw, w, ENT + PRU)
        q, _ = rec_boot(S, "CoLd", noms)
        # frecuencia con que cada ventana es la mejor al remuestrear réplicas
        rng = np.random.default_rng(61)
        n = len(com)
        cuenta = {k: 0 for k in noms}
        for _ in range(2000):
            b = rng.integers(0, n, n)
            den = S["oraculo"][b].mean() - S["CoLd"][b].mean()
            r = {k: (S[k][b].mean() - S["CoLd"][b].mean()) / den for k in noms}
            cuenta[max(r, key=r.get)] += 1
        mejor = max(noms, key=lambda k: q[k]["rec"])
        out["R4"][c] = {"rec": {k: q[k] for k in noms}, "mejor": mejor, "frec_mejor": {k: v / 2000 for k, v in cuenta.items()},
                        "n_rep": n, "ventaja": q["_abs"]}
        L.append(f"{c:12s} n = {n} | " + "  ".join(f"{v:g} s: {q[nom_v(v)]['rec']:.3f}" for v in VENT_R4)
                 + f" | mejor {mejor} (frecuencia {cuenta[mejor] / 2000:.2f})")
    # N = 1000: ganancia elegida con la celda central de N = 250
    L.append("\n=== N = 1000, 48 informados (réplicas 0-7): ganancia elegida en N = 250 ===")
    for etq, pulso, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        g_star = out["eleccion"][f"N250|n12|{c}"]["ganancia"]
        g_star = g_star if g_star in ("sin_memoria", "sin_memoria_x6", "sin_memoria_x12") else None
        obs = ["integrador", "sin_memoria", "sin_memoria_x6", "sin_memoria_x12"]
        com, S = series(todo, "n48", ["CoLd", "oraculo"] + obs, 1000, pulso, tw, T_N[1000], ENT)
        q, dif = rec_boot(S, "CoLd", obs)
        g_max = max(obs[1:], key=lambda k: q[k]["rec"])
        out["N1000"][c] = {"rec": {k: q[k] for k in obs}, "g_star_N250": g_star, "g_max": g_max,
                           "int_vs_gmax": dif("integrador", g_max),
                           "int_vs_gstar": dif("integrador", g_star) if g_star else None}
        L.append(f"{c:12s} n = {len(com)} | " + "  ".join(f"{k}: {q[k]['rec']:.3f}" for k in obs)
                 + f" | elegida en N = 250: {g_star} | integrador − {g_max}: {out['N1000'][c]['int_vs_gmax']['dif']:+.3f} "
                 f"[{out['N1000'][c]['int_vs_gmax']['ic95'][0]:+.3f}; {out['N1000'][c]['int_vs_gmax']['ic95'][1]:+.3f}]")
    D = out["dif"]
    c3 = [c for c in D if not c.startswith("N250|n1|")]
    c1 = [c for c in D if c.startswith("N250|n1|") and not c.endswith("|1")]
    paso = {0.25: (0.5,), 0.5: (0.25, 1.0), 1.0: (0.5, 2.0), 2.0: (1.0, 4.0), 4.0: (2.0,)}
    v_de = lambda nom: 1.0 if nom == "integrador" else float(nom.split("_")[1][:-1])
    j3 = [c for c in out["eleccion"] if v_de(out["eleccion"][c]["ventana"]) in (1.0,) + paso[1.0]]
    out["pred"] = {
        "J1": {"falla_en": [c for c in c3 if D[c]["int1_vs_gstar"]["ic95"][0] <= 0],
               "cumple": bool(all(D[c]["int1_vs_gstar"]["ic95"][0] > 0 for c in c3))},
        "J2": {"difs": {c: D[c]["int1_vs_gstar"] for c in c1},
               "cumple": bool(all(D[c]["int1_vs_gstar"]["ic95"][0] <= 0 for c in c1))},
        "J3": {"a_un_paso": len(j3), "de": len(out["eleccion"]),
               "dif_max": max(abs(D[c]["vstar_vs_int1"]["dif"]) for c in D),
               "cumple": bool(len(j3) >= 10 and all(abs(D[c]["vstar_vs_int1"]["dif"]) <= 0.05 for c in D))},
        "J4": {"cumple": bool(all(v["int_vs_gmax"]["dif"] > 0 and (v["int_vs_gstar"] is None or v["int_vs_gstar"]["dif"] > 0)
                                  for v in out["N1000"].values()))},
        "W1": {"mejores": {c: v["mejor"] for c, v in out["R4"].items()},
               "cumple": bool(all(v["mejor"] in ("integrador_0.5s", "integrador", "integrador_2s") for v in out["R4"].values()))},
        "W2": {"cociente": {c: v["rec"]["integrador"]["rec"] / v["rec"][v["mejor"]]["rec"] for c, v in out["R4"].items()},
               "cumple": bool(all(v["rec"]["integrador"]["rec"] >= 0.9 * v["rec"][v["mejor"]]["rec"] for v in out["R4"].values()))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v, default=float)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    guardar(out, L, F_JSON, F_TXT)
