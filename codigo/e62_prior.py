"""Etapa 62 (plan 14, revisor independiente): sensibilidad de la recuperación al prior pi0.

Observador de 1 s del motor con pi0 en {1/250, 0.05, 0.5}; referencia pi0 = 0.173 (corridas
guardadas). Diseño y predicciones Q1-Q3 en docs/bitacora/resultados_E56-E65.md.

    PYTHONPATH=codigo python codigo/e62_prior.py
"""
import json
import os

from flocking import phi_grid
from fase4 import D_PHI, trabajo, correr, crudos_previos, pareado, limpio, clave

PIS = (0.004, 0.05, 0.5)
CELDAS = (("forzado", False, "ViSt", 4.0), ("maniobra", True, "CoLd", 4.0), ("maniobra", True, "CoLd", 1.0))
N_REP = 8
F_CRUDO, F_JSON, F_TXT = "resultados/E62_prior_crudo.json", "resultados/E62_prior.json", "resultados/E62_prior.txt"

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phis = phi_grid(D_PHI)
    jobs = [trabajo("base", f"pi{p:g}", 250, tw, pulso, i, phi, rep, kw={"pi0": p}, base="adaptado")
            for p in PIS for etq, pulso, ref, tw in CELDAS for i, phi in enumerate(phis) for rep in range(N_REP)]
    res, seg = correr(jobs, F_CRUDO + ".parcial", "E62")
    json.dump({"resp": res}, open(F_CRUDO, "w"))
    if os.path.exists(F_CRUDO + ".parcial"):
        os.remove(F_CRUDO + ".parcial")
    prev = crudos_previos() + [{**x, "celda": "base"} for x in json.load(open("resultados/E48_replicas_crudo.json"))["resp"]]
    todo = list({clave(x): x for x in prev}.values()) + res
    out, L = {"rec": {}}, ["=== E62 · sensibilidad al prior pi0 (N = 250, ciclo de 1 s, 8 réplicas) ==="]
    for etq, pulso, ref, tw in CELDAS:
        c = f"{etq}|{tw:g}"
        fila = {"0.173": limpio(pareado(todo, "base", "adaptado", ref, 250, pulso, 15.8, tw, reps=list(range(N_REP))))}
        for p in PIS:
            fila[f"{p:g}"] = limpio(pareado(todo, "base", f"pi{p:g}", ref, 250, pulso, 15.8, tw, reps=list(range(N_REP))))
        out["rec"][c] = fila
        L.append(f"{c:12s}: " + " | ".join(f"pi0 {k}: {fila[k]['rec']:.3f} [{fila[k]['ic95'][0]:.3f}; {fila[k]['ic95'][1]:.3f}]"
                                         for k in ("0.004", "0.05", "0.173", "0.5")))
    R = out["rec"]
    cs = list(R)
    out["pred"] = {
        "Q1": {"cumple": bool(all(R[c]["0.004"]["rec"] <= R[c]["0.05"]["rec"] <= R[c]["0.173"]["rec"] for c in cs))},
        "Q2": {"cocientes": [R[c]["0.004"]["rec"] / R[c]["0.173"]["rec"] for c in cs],
               "cumple": bool(all(R[c]["0.004"]["rec"] >= 0.5 * R[c]["0.173"]["rec"] for c in cs))},
        "Q3": {"cumple": bool(all(R[c]["0.5"]["rec"] >= R[c]["0.173"]["rec"] for c in cs))}}
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    L.append(f"\n{seg:.0f} s")
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
