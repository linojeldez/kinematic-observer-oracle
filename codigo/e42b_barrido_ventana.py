"""Etapa 42b: completar el barrido de la ventana del detector por el lado largo.

Por qué. E42 falló N5: con maniobra única la ventana de 2 s recupera más que la de 1 s
(0,663 frente a 0,600 con N = 250; 0,530 frente a 0,455 con N = 1000). El barrido de E42
llegaba sólo a 2 s, y «integrar tau_W = 4 s no recupera casi nada» se midió únicamente
con el líder forzado (E18). Si con maniobra única la ventana de 4 s rindiera todavía más,
«la ventana es la de la huella» no valdría para ese protocolo.

Diseño: mismas corridas que el barrido de E42 (4 réplicas, 18 ángulos, mismas semillas,
ventana de medida sqrt(N)), agregando:
  - N = 250: ciclo de 0,25 s y de 4 s, los dos protocolos;
  - N = 1000: ciclo de 4 s, los dos protocolos.
Las referencias (ViSt, oráculo) y los ciclos de 0,5, 1 y 2 s se leen del crudo de E42.

    PYTHONPATH=codigo python codigo/e42b_barrido_ventana.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from e42_escalera_N import (D_PHI, TAU_W, T_N, N_REP_BARRIDO, _w, correr, R_rep, rec_ic)

NUEVOS = {
    250: {"ciclo 0.25 s": 0.25, "ciclo 4 s": 4.0},
    1000: {"ciclo 4 s": 4.0},
}
F_CRUDO = "resultados/E42b_barrido_ventana_crudo.json"
F_JSON = "resultados/E42b_barrido_ventana.json"
F_TXT = "resultados/E42b_barrido_ventana.txt"
F_PARCIAL = F_CRUDO + ".parcial"
NOMBRE = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s",
          4.0: "ciclo 4 s"}


def trabajos():
    phis = phi_grid(D_PHI)
    out = []
    for N, d in NUEVOS.items():
        for nom, tau in d.items():
            kw = dict(will_mode="bayes", detector="ciclo", tau_det=tau)
            for pulso in (False, True):
                for i, phi in enumerate(phis):
                    for rep in range(N_REP_BARRIDO):
                        out.append((nom, "WiSt", kw, N, TAU_W, pulso, i, float(phi), rep, T_N[N]))
    out.sort(key=lambda a: -a[3])
    return out


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nuevos, seg = correr(trabajos(), F_PARCIAL, "E42b")
    with open(F_CRUDO, "w") as fh:
        json.dump({"resp": nuevos}, fh)
    if os.path.exists(F_PARCIAL):
        os.remove(F_PARCIAL)
    with open("resultados/E42_escalera_N_crudo.json") as fh:
        res = json.load(fh)["resp"] + nuevos

    out = {"config": {"replicas": N_REP_BARRIDO, "ventana_medida": "sqrt(N)"}, "rec": {}}
    L = ["=== E42b · barrido de la ventana del detector, lado largo ===",
         f"{N_REP_BARRIDO} réplicas, mismas semillas que el barrido de E42; ventana de medida sqrt(N)",
         f'{"N":>5s} {"protocolo":>9s} | ' + " | ".join(f"{t:>20g} s" for t in (0.25, 0.5, 1.0, 2.0, 4.0))
         + " | mejor"]
    for N in (250, 1000):
        for pulso, etq in ((False, "tractor"), (True, "pulso")):
            win = T_N[N]
            Rv, Ro = R_rep(res, "ViSt", N, pulso, win), R_rep(res, "oraculo", N, pulso, win)
            fila = {}
            for tau, nom in NOMBRE.items():
                Rm = R_rep(res, nom, N, pulso, win)
                if not len(Rm):
                    continue
                n = min(len(Rm), N_REP_BARRIDO)
                q = rec_ic(Rm[:n], Rv[:n], Ro[:n])
                q.pop("_boot", None)
                fila[_w(tau)] = q
            Ra = R_rep(res, "amplitud", N, pulso, win)
            qa = rec_ic(Ra[:N_REP_BARRIDO], Rv[:N_REP_BARRIDO], Ro[:N_REP_BARRIDO])
            qa.pop("_boot", None)
            mejor = max(fila, key=lambda k: fila[k]["rec"])
            out["rec"][f"{etq}|{N}"] = {**fila, "amplitud": qa, "mejor": mejor}
            celdas = []
            for t in (0.25, 0.5, 1.0, 2.0, 4.0):
                q = fila.get(_w(t))
                celdas.append(f'{q["rec"]:6.3f} [{q["ic95"][0]:5.3f}; {q["ic95"][1]:5.3f}]'
                              if q else f'{"—":>22s}')
            L.append(f"{N:5d} {etq:>9s} | " + " | ".join(celdas) + f" | {mejor} s"
                     + f'   (sin memoria {qa["rec"]:.3f})')

    def g(etq, N, tau):
        return out["rec"][f"{etq}|{N}"][_w(tau)]["rec"]

    pr = {}
    pr["V1"] = {"4s_menor_que_2s": {f"{e}|{N}": [g(e, N, 4.0), g(e, N, 2.0)]
                                    for e in ("tractor", "pulso") for N in (250, 1000)}}
    pr["V1"]["cumple"] = bool(all(a < b for a, b in pr["V1"]["4s_menor_que_2s"].values()))
    pr["V2"] = {"0.25s_menor_que_0.5s": {f"{e}|250": [g(e, 250, 0.25), g(e, 250, 0.5)]
                                         for e in ("tractor", "pulso")}}
    pr["V2"]["cumple"] = bool(all(a < b for a, b in pr["V2"]["0.25s_menor_que_0.5s"].values()))
    out["pred"] = pr
    L.append("\n=== Expectativas declaradas ===")
    for k, v in pr.items():
        L.append(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v)[:300]}')
    L.append(f"\n{seg:.0f} s")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
