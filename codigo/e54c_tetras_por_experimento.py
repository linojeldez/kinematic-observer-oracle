"""Etapa 54c (plan 14, objeción mayor 6 y menor 3): tetras por experimento.

Configuración basal de E54 (N = 8 y 10). AUC(1 s), AUC(cuadro) y ganancia por experimento;
dejar un experimento fuera; promedio con igual peso por experimento; ventana del 95 % definida
sobre el máximo bruto (como en el paper) y sobre el exceso sobre 0,5. Diseño y predicciones
T1-T3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e54c_tetras_por_experimento.py
"""

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import e27_tetras as e27
from e54_peces import experimento, pares, curva, tau95, auc, TAUS, BASE

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs = [(fn, g) for g in ("8", "10") for fn in sorted(glob.glob(os.path.join(e27.DATA, g, "*.csv")))]
    with ProcessPoolExecutor(max_workers=8) as ex:
        exps = list(ex.map(experimento, jobs))
    cfg = f"{BASE[0]:g}|{BASE[1]:g}"
    exps = [e for e in exps if e["cfg"].get(cfg, {}).get("ev")]
    pp = pares(exps, cfg)
    c = curva(pp)
    e54 = json.load(open("resultados/E54_peces.json"))
    nev = [len(e["cfg"][cfg]["ev"]) for e in exps]

    def tau95_exceso(cv):
        fin = {t: a for t, a in cv.items() if t != "cuadro" and np.isfinite(a)}
        tope = max(fin.values())
        return min(t for t, a in fin.items() if a - 0.5 >= 0.95 * (tope - 0.5))

    por = []
    for e, p, n in zip(exps, pp, nev):
        a1, a0 = auc(*p["1.0"]), auc(*p["cuadro"])
        por.append({"exp": e["fn"], "grupo": e["grupo"], "giros": n, "auc1": a1, "auc_cuadro": a0, "ganancia": a1 - a0})
    loo = []
    for j in range(len(exps)):
        cj = curva(pp, [i for i in range(len(exps)) if i != j])
        loo.append({"fuera": exps[j]["fn"], "giros": nev[j], "auc1": cj[1.0], "ganancia": cj[1.0] - cj["cuadro"],
                    "tau95": tau95(cj)})
    con5 = [x for x in por if x["giros"] >= 5]
    ip = lambda k: float(np.mean([x[k] for x in por]))
    out = {"n_exp": len(exps), "n_giros": int(sum(nev)),
           "ancla": {"auc1": [c[1.0], e54["N8_10"]["curva"]["1.0"] if "N8_10" in e54 else None]},
           "agrupada": {"auc1": c[1.0], "auc_cuadro": c["cuadro"], "ganancia": c[1.0] - c["cuadro"],
                        "tau95_bruto": tau95(c), "tau95_exceso": tau95_exceso(c)},
           "por_experimento": por,
           "igual_peso": {"auc1": ip("auc1"), "auc_cuadro": ip("auc_cuadro"), "ganancia": ip("ganancia")},
           "dejando_uno_fuera": loo,
           "loo_rango": {"auc1": [min(x["auc1"] for x in loo), max(x["auc1"] for x in loo)],
                         "ganancia": [min(x["ganancia"] for x in loo), max(x["ganancia"] for x in loo)],
                         "tau95": [min(x["tau95"] for x in loo), max(x["tau95"] for x in loo)]},
           "positivos_con5": [int(sum(x["ganancia"] > 0 for x in con5)), len(con5)]}
    mayor = max(loo, key=lambda x: x["giros"])
    out["sin_el_mayor"] = mayor
    out["pred"] = {
        "T1": {"cumple": bool(0.15 <= out["loo_rango"]["ganancia"][0] and out["loo_rango"]["ganancia"][1] <= 0.35
                              and out["loo_rango"]["auc1"][0] > 0.70)},
        "T2": {"positivos": out["positivos_con5"], "cumple": bool(out["positivos_con5"][0] >= 15)},
        "T3": {"dif": out["igual_peso"]["auc1"] - c[1.0], "cumple": bool(abs(out["igual_peso"]["auc1"] - c[1.0]) < 0.05)}}
    L = [f"=== E54c · tetras por experimento ({len(exps)} experimentos, {sum(nev)} giros) ===",
         f"agrupada: AUC(1 s) {c[1.0]:.4f}, AUC(cuadro) {c['cuadro']:.4f}, ganancia {c[1.0] - c['cuadro']:+.4f}; "
         f"ventana 95 % del máximo bruto {tau95(c):g} s, del exceso sobre 0,5 {tau95_exceso(c):g} s",
         f"igual peso por experimento: AUC(1 s) {out['igual_peso']['auc1']:.3f}, AUC(cuadro) {out['igual_peso']['auc_cuadro']:.3f}, "
         f"ganancia {out['igual_peso']['ganancia']:+.3f}",
         f"dejando uno fuera: AUC(1 s) {out['loo_rango']['auc1'][0]:.3f}–{out['loo_rango']['auc1'][1]:.3f}, ganancia "
         f"{out['loo_rango']['ganancia'][0]:+.3f}–{out['loo_rango']['ganancia'][1]:+.3f}, ventana 95 % "
         f"{out['loo_rango']['tau95'][0]:g}–{out['loo_rango']['tau95'][1]:g} s",
         f"sin el experimento de más giros ({mayor['fuera']}, {mayor['giros']} giros): AUC(1 s) {mayor['auc1']:.3f}, "
         f"ganancia {mayor['ganancia']:+.3f}, ventana {mayor['tau95']:g} s",
         f"ganancia positiva en {out['positivos_con5'][0]} de {out['positivos_con5'][1]} experimentos con 5 o más giros",
         "por experimento (giros, AUC 1 s, ganancia): " + "; ".join(f"{x['giros']}, {x['auc1']:.2f}, {x['ganancia']:+.2f}" for x in por),
         "\n=== Predicciones ==="] + [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    json.dump(out, open("resultados/E54c_tetras_por_experimento.json", "w"), indent=1, default=float)
    open("resultados/E54c_tetras_por_experimento.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
