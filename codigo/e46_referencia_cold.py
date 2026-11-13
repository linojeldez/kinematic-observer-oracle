"""Etapa 46: CoLd como referencia única de la maniobra única.

Por qué. Con maniobra única, entre 9 y 14 % de la ventaja del oráculo sobre ViSt la
obtiene CoLd: el líder insiste en su rumbo mientras le dura el estado y nadie lo lee
(E44b). La recuperación de un observador pasa a contarse desde CoLd,

    recuperación = (R - R_CoLd) / (R_oráculo - R_CoLd),

en los dos protocolos. Con líder forzado CoLd y ViSt son la misma corrida (E44b), así
que ahí se usan las corridas de ViSt y ninguna cifra cambia.

Qué hace. Reanaliza los crudos de E42, E42b, E43, E43b, E44 y E44b (mismas semillas) y
corre de nuevo el observador secuencial (N = 250, tau_W = 4 s, los dos protocolos,
288 corridas) con el arnés de E42, porque E22 sólo guardó el agregado. Deja en un solo
archivo todas las cifras de recuperación que usa paper_a_corto.

Predicciones A0 a A6 en docs/bitacora/resultados_E36-E45.md (declaradas antes de correr).

    PYTHONPATH=codigo python codigo/e46_referencia_cold.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from flocking import phi_grid
from e42_escalera_N import D_PHI, N_REP, T_N, _w, correr, R_rep, rec_ic
from e44_lectura import cargar_base, t_equivalente, COMPUERTAS

F_CRUDO = "resultados/E46_referencia_cold_crudo.json"
F_JSON = "resultados/E46_referencia_cold.json"
F_TXT = "resultados/E46_referencia_cold.txt"
F_PARCIAL = F_CRUDO + ".parcial"

SEC = ("secuencial", "WiSt", dict(will_mode="bayes", detector="secuencial", tau_det=1.0))
VENT = {0.25: "ciclo 0.25 s", 0.5: "ciclo 0.5 s", 1.0: "adaptado", 2.0: "ciclo 2 s",
        4.0: "ciclo 4 s"}
NS = (250, 500, 1000)
PULSO = ((250, 4.0), (500, 4.0), (1000, 4.0), (250, 2.0), (250, 1.0), (1000, 1.0))
PROT = ((False, "tractor"), (True, "pulso"))
N_BARRIDO = 4


def trabajos():
    phis = phi_grid(D_PHI)
    return [(SEC[0], SEC[1], SEC[2], 250, 4.0, pulso, i, float(phi), rep, T_N[250])
            for pulso in (False, True) for i, phi in enumerate(phis) for rep in range(N_REP)]


def _crudo(nombre):
    with open(f"resultados/{nombre}_crudo.json") as fh:
        return json.load(fh)["resp"]


def _json(nombre):
    with open(f"resultados/{nombre}.json") as fh:
        return json.load(fh)


def cargar(sec):
    base, e42 = cargar_base()
    res = list(base)
    res += [x for x in _crudo("E43_escalera_tauW")
            if x["N"] == 250 and x["pulso"] and abs(x["tau_W"] - 2.0) < 1e-9]
    res += [x for x in e42 if x["nom"].startswith("ciclo")]
    res += _crudo("E42b_barrido_ventana")
    res += [x for x in _crudo("E44_lectura") if not x["nom"].startswith("ancla")]
    res += [x for x in _crudo("E44b_cold") if x["pulso"]]
    res += sec
    return res


def _limpio(q):
    q = dict(q)
    q.pop("_boot", None)
    return q


def _f(q):
    return f'{q["rec"]:6.3f} [{q["ic95"][0]:6.3f}; {q["ic95"][1]:6.3f}]'


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.exists(F_CRUDO):
        with open(F_CRUDO) as fh:
            c = json.load(fh)
        sec, seg = c["resp"], c.get("segundos", 0.0)
    else:
        sec, seg = correr(trabajos(), F_PARCIAL, "E46")
        with open(F_CRUDO, "w") as fh:
            json.dump({"resp": sec, "segundos": seg}, fh)
        if os.path.exists(F_PARCIAL):
            os.remove(F_PARCIAL)
    res = cargar(sec)

    def R(nom, N, pulso, win, tau=4.0):
        return R_rep(res, nom, N, pulso, win, tau)

    def rec(nom, N, pulso, win, tau=4.0, ref=None, n=None, seed=42, techo="oraculo"):
        """Recuperación de `nom`. ref por defecto: CoLd con maniobra única, ViSt con líder
        forzado (allí son la misma corrida)."""
        ref = ref or ("CoLd" if pulso else "ViSt")
        Rm, Rr, Ro = R(nom, N, pulso, win, tau), R(ref, N, pulso, win, tau), R(techo, N, pulso, win, tau)
        if n:
            Rm, Rr, Ro = Rm[:n], Rr[:n], Ro[:n]
        return rec_ic(Rm, Rr, Ro, seed=seed)

    out = {"config": {"definicion": "rec = (R - R_CoLd)/(R_oraculo - R_CoLd); con líder forzado "
                                    "CoLd = ViSt (E44b) y se usan las corridas de ViSt",
                      "replicas": N_REP, "replicas_barrido": N_BARRIDO,
                      "compuertas_s": list(COMPUERTAS)},
           "escalera": {}, "barrido": {}, "secuencial": {}, "lectura": {}, "lectura_tauW1": {},
           "caidas": {}, "anclas": {}, "pred": {}}
    L = ["=== E46 · CoLd como referencia única de la maniobra única ==="]
    P = L.append

    # ------------------------------------------------------------------ anclas (A0)
    anc = {}

    def cmp(etq, mio, arch):
        d = abs(mio - arch)
        a = anc.setdefault(etq, {"delta_max": 0.0, "n": 0})
        a["delta_max"] = max(a["delta_max"], d)
        a["n"] += 1

    e42j, e42bj = _json("E42_escalera_N"), _json("E42b_barrido_ventana")
    e43j, e43bj = _json("E43_escalera_tauW"), _json("E43b_tauW_N1000")
    e44j, e44bj, e22j = _json("E44_lectura"), _json("E44b_cold"), _json("E22_completar")
    for pulso, etq in PROT:
        for N in NS:
            for win in (4.0, T_N[N]):
                for nom in ("adaptado", "amplitud"):
                    cmp("E42", rec(nom, N, pulso, win, ref="ViSt")["rec"],
                        e42j["rec"][f"{etq}|{N}|{_w(win)}"][nom]["rec"])
        for N in (250, 1000):
            for t, v in e42bj["rec"][f"{etq}|{N}"].items():
                if t == "mejor":
                    continue
                nom = "amplitud" if t == "amplitud" else VENT[float(t)]
                cmp("E42b", rec(nom, N, pulso, T_N[N], ref="ViSt", n=N_BARRIDO)["rec"], v["rec"])
            for win in (4.0, T_N[N]):
                for nom, v in e44j["rec"][f"{etq}|{N}|{_w(win)}"].items():
                    if nom.startswith("R_"):
                        continue
                    cmp("E44", rec(nom, N, pulso, win, ref="ViSt")["rec"], v["rec"])
    for tau in (1.0, 2.0, 4.0):
        for win in (4.0, 15.8):
            for nom in ("adaptado", "amplitud"):
                cmp("E43", rec(nom, 250, True, win, tau, ref="ViSt")["rec"],
                    e43j["rec"][f"{tau:g}|{_w(win)}"][nom]["rec"])
    for win in (4.0, 31.6):
        for nom in ("adaptado", "amplitud"):
            cmp("E43b", rec(nom, 1000, True, win, 1.0, ref="ViSt")["rec"],
                e43bj["rec"][_w(win)][nom]["rec"])
    for N in (250, 1000):
        for nom, v in e44j["rec"][f"pulso|{N}|{_w(T_N[N])}|tauW1"].items():
            if not nom.startswith("R_"):
                cmp("E44 tauW1", rec(nom, N, True, T_N[N], 1.0, ref="ViSt")["rec"], v["rec"])
    for N, tau in PULSO:
        for win in (4.0, T_N[N]):
            fila = e44bj["rec"][f"{N}|{tau:g}|{_w(win)}"]
            cmp("E44b", rec("CoLd", N, True, win, tau, ref="ViSt")["rec"], fila["cold_ref_vist"]["rec"])
            for nom in ("adaptado", "amplitud", "D"):
                if f"{nom}_ref_cold" in fila:
                    cmp("E44b", rec(nom, N, True, win, tau)["rec"], fila[f"{nom}_ref_cold"]["rec"])
    for pulso, et in ((False, "sostenido"), (True, "pulso")):
        cmp("E22 secuencial", rec("secuencial", 250, pulso, T_N[250], ref="ViSt")["rec"],
            e22j["secuencial"][et]["rec"] / 100.0)
    out["anclas"] = anc
    P("\n--- anclas (referencia ViSt frente a lo archivado; E44b con referencia CoLd) ---")
    for k, v in anc.items():
        P(f"  {k:<16s} |Δ| máx = {v['delta_max']:.2e} en {v['n']} comparaciones")

    # ------------------------------------------------------------------ identidad (A1)
    dmax, n_id = 0.0, 0

    def identidad(nom, N, win, tau, n=None):
        global dmax, n_id
        if len(R(nom, N, True, win, tau)) == 0:
            return
        rv = rec(nom, N, True, win, tau, ref="ViSt", n=n)["rec"]
        rc = rec(nom, N, True, win, tau, n=n)["rec"]
        c = rec("CoLd", N, True, win, tau, ref="ViSt", n=n)["rec"]
        dmax = max(dmax, abs(rc - (rv - c) / (1.0 - c)))
        n_id += 1

    for N, tau in PULSO:
        for win in (4.0, T_N[N]):
            for nom in ("adaptado", "amplitud", "D", "R", "G1", "G2", "G4", "G8", "secuencial"):
                identidad(nom, N, win, tau)
    for N in (250, 1000):
        for t, nom in VENT.items():
            identidad(nom, N, T_N[N], 4.0, n=N_BARRIDO)
        identidad("amplitud", N, T_N[N], 4.0, n=N_BARRIDO)
    out["identidad"] = {"delta_max": dmax, "n": n_id}
    P(f"\n--- identidad rec_CoLd = (rec_ViSt - c)/(1 - c): |Δ| máx = {dmax:.2e} en {n_id} comparaciones")

    # ------------------------------------------------------------------ escalera
    P("\n--- escalera (recuperación; maniobra única contada desde CoLd) ---")
    P(f'{"protocolo":<8s} {"N":>5s} {"tau_W":>5s} | {"ventaja":>8s} {"CoLd/ViSt":>9s} | '
      f'{"adaptado, entero":>24s} {"adaptado, 4 s":>24s} | {"sin memoria, entero":>24s} {"sin memoria, 4 s":>24s}')
    celdas = [(False, N, 4.0) for N in NS] + [(True, N, tau) for N, tau in PULSO]
    for pulso, N, tau in celdas:
        etq = "pulso" if pulso else "tractor"
        ref = "CoLd" if pulso else "ViSt"
        T = T_N[N]
        fila = {"ventana": T, "referencia": ref,
                "ventaja": float(R("oraculo", N, pulso, T, tau).mean() - R(ref, N, pulso, T, tau).mean())}
        if pulso:
            fila["cold_sobre_vist"] = _limpio(rec("CoLd", N, True, T, tau, ref="ViSt"))
        for nom in ("adaptado", "amplitud"):
            fila[nom] = {"entero": _limpio(rec(nom, N, pulso, T, tau)),
                         "primeros4": _limpio(rec(nom, N, pulso, 4.0, tau))}
        out["escalera"][f"{etq}|{N}|{tau:g}"] = fila
        cv = f'{fila["cold_sobre_vist"]["rec"]:9.3f}' if pulso else " " * 9
        P(f'{etq:<8s} {N:5d} {tau:5.1f} | {fila["ventaja"]:8.4f} {cv} | '
          f'{_f(fila["adaptado"]["entero"])} {_f(fila["adaptado"]["primeros4"])} | '
          f'{_f(fila["amplitud"]["entero"])} {_f(fila["amplitud"]["primeros4"])}')

    # ------------------------------------------------------------------ barrido de ventana
    P(f"\n--- barrido de la ventana del observador ({N_BARRIDO} réplicas, ventana de medida sqrt(N)) ---")
    P(f'{"protocolo":<8s} {"N":>5s} | {"sin memoria":>11s} ' + " ".join(f"{t:>6g} s" for t in VENT))
    for pulso, etq in PROT:
        for N in (250, 1000):
            d = {"amplitud": _limpio(rec("amplitud", N, pulso, T_N[N], n=N_BARRIDO))}
            for t, nom in VENT.items():
                if len(R(nom, N, pulso, T_N[N])):
                    d[f"{t:g}"] = _limpio(rec(nom, N, pulso, T_N[N], n=N_BARRIDO))
            vs = {t: d[t]["rec"] for t in d if t != "amplitud"}
            d["mejor"] = max(vs, key=vs.get)
            out["barrido"][f"{etq}|{N}"] = d
            P(f'{etq:<8s} {N:5d} | {d["amplitud"]["rec"]:11.3f} ' + " ".join(
                f'{d[f"{t:g}"]["rec"]:8.3f}' if f"{t:g}" in d else f'{"---":>8s}' for t in VENT)
              + f'   mejor: {d["mejor"]} s')

    # ------------------------------------------------------------------ secuencial
    P("\n--- observador secuencial, N = 250, tau_W = 4 s ---")
    for pulso, etq in PROT:
        s, a = rec("secuencial", 250, pulso, T_N[250]), rec("adaptado", 250, pulso, T_N[250])
        out["secuencial"][etq] = {"secuencial": _limpio(s), "adaptado": _limpio(a),
                                  "disjuntos": bool(s["ic95"][0] > a["ic95"][1] or s["ic95"][1] < a["ic95"][0])}
        P(f"  {etq:<8s} secuencial {_f(s)}   ciclo de 1 s {_f(a)}")

    # ------------------------------------------------------------------ intervenciones (E44)
    P("\n--- intervenciones sobre la lectura, tau_W = 4 s, ventana sqrt(N) ---")
    P(f'{"celda":<14s} | {"G(1)":>6s} {"G(2)":>6s} {"G(4)":>6s} {"G(8)":>6s} | {"adaptado":>24s} | '
      f'{"D, sólo el líder":>24s} | {"R, sin el líder":>24s} | {"T eq. adapt.":>12s} {"T eq. D":>8s} | {"D / adaptado":>24s}')
    for pulso, etq in PROT:
        ref = "CoLd" if pulso else "ViSt"
        for N in (250, 1000):
            T = T_N[N]
            d = {nom: _limpio(rec(nom, N, pulso, T)) for nom in ("G1", "G2", "G4", "G8", "adaptado", "D", "R")}
            gs = [d[f"G{t:g}"]["rec"] for t in COMPUERTAS]
            d["t_equiv_adaptado"] = t_equivalente(d["adaptado"]["rec"], gs)
            d["t_equiv_D"] = t_equivalente(d["D"]["rec"], gs)
            d["D_sobre_adaptado"] = _limpio(rec("D", N, pulso, T, techo="adaptado"))
            d["R_sobre_adaptado"] = _limpio(rec("R", N, pulso, T, techo="adaptado"))
            d["primeros4"] = {nom: _limpio(rec(nom, N, pulso, 4.0)) for nom in ("G2", "adaptado", "D")}
            out["lectura"][f"{etq}|{N}"] = d
            P(f'{etq + " N=" + str(N):<14s} | ' + " ".join(f"{g:6.3f}" for g in gs)
              + f' | {_f(d["adaptado"])} | {_f(d["D"])} | {_f(d["R"])} | '
              f'{d["t_equiv_adaptado"]:>10.2f} s {d["t_equiv_D"]:>6.2f} s | {_f(d["D_sobre_adaptado"])}')
    P("  primeros 4 s: " + "; ".join(
        f'{k}: G(2) {v["primeros4"]["G2"]["rec"]:.3f}, adaptado {v["primeros4"]["adaptado"]["rec"]:.3f}, '
        f'D {v["primeros4"]["D"]["rec"]:.3f}' for k, v in out["lectura"].items()))
    for N in (250, 1000):
        d = {nom: _limpio(rec(nom, N, True, T_N[N], 1.0)) for nom in ("D", "R", "adaptado")}
        out["lectura_tauW1"][str(N)] = d
        P(f'  tau_W = 1 s, maniobra única, N = {N}: D {_f(d["D"])}   R {_f(d["R"])}   adaptado {_f(d["adaptado"])}')
    for pulso, etq in PROT:
        for nom in ("G4", "adaptado"):
            b = [rec(nom, N, pulso, T_N[N], seed=sem) for N, sem in ((250, 42), (1000, 43))]
            lo, hi = np.percentile(b[0]["_boot"] - b[1]["_boot"], [2.5, 97.5])
            out["caidas"][f"{etq}|{nom}"] = {"N250": b[0]["rec"], "N1000": b[1]["rec"],
                                            "caida": b[0]["rec"] - b[1]["rec"], "ic95": [float(lo), float(hi)]}
            P(f'  caída de {nom} de N = 250 a 1000, {etq}: {b[0]["rec"]:.3f} -> {b[1]["rec"]:.3f} = '
              f'{b[0]["rec"] - b[1]["rec"]:.3f} [{lo:.3f}; {hi:.3f}]')

    # ------------------------------------------------------------------ predicciones
    E, B, S, Lc = out["escalera"], out["barrido"], out["secuencial"], out["lectura"]
    pr = out["pred"]
    pr["A0"] = {"anclas": anc, "cumple": bool(all(
        v["delta_max"] < (1e-9 if k.startswith("E22") else 1e-12) and v["n"] > 0 for k, v in anc.items()))}
    pr["A1"] = {**out["identidad"], "cumple": bool(dmax < 1e-12 and n_id > 0)}
    b250 = {t: B["pulso|250"][t]["rec"] for t in ("0.25", "0.5", "1", "2", "4")}
    esp = {"0.25": 0.07, "0.5": 0.27, "1": 0.53, "2": 0.60, "4": 0.32}
    pr["A2"] = {"N250": b250, "mejor": {k: B[k]["mejor"] for k in ("pulso|250", "pulso|1000")},
                "cumple": bool(all(abs(b250[t] - esp[t]) <= 0.02 for t in esp)
                               and B["pulso|250"]["mejor"] == "2" and B["pulso|1000"]["mejor"] == "2")}
    sp = S["pulso"]
    pr["A3"] = {"secuencial": sp["secuencial"]["rec"], "ic95": sp["secuencial"]["ic95"],
                "ciclo": sp["adaptado"]["rec"], "disjuntos": sp["disjuntos"],
                "cumple": bool(abs(sp["secuencial"]["rec"] - 0.58) <= 0.01 and sp["disjuntos"]
                               and sp["secuencial"]["rec"] > sp["adaptado"]["rec"])}
    te = {str(N): Lc[f"pulso|{N}"]["t_equiv_adaptado"] for N in (250, 1000)}
    te_arch = {str(N): e44j["razon"][f"pulso|{N}"].get("t_equiv_adaptado") for N in (250, 1000)}
    pr["A4"] = {"t_equiv": te, "archivado": te_arch,
                "cumple": bool(abs(te["250"] - 2.7) < 0.05 and abs(te["1000"] - 2.5) < 0.05)}
    cg, ca = out["caidas"]["pulso|G4"], out["caidas"]["pulso|adaptado"]
    pr["A5"] = {"G4": [cg["N250"], cg["N1000"]], "caida_G4": cg["caida"], "caida_adaptado": ca["caida"],
                "cumple": bool(abs(cg["N250"] - 0.75) <= 0.01 and abs(cg["N1000"] - 0.58) <= 0.01
                               and abs(cg["caida"] - 0.165) <= 0.02)}

    def ad(N, tau=4.0):
        return E[f"pulso|{N}|{tau:g}"]["adaptado"]["entero"]["rec"]

    def am(etq, N, tau=4.0):
        return E[f"{etq}|{N}|{tau:g}"]["amplitud"]["entero"]["rec"]

    viejas = {
        "N2": bool(0.47 <= ad(500) <= 0.60 and 0.40 <= ad(1000) <= 0.51),
        "N3_sin_memoria_bajo_0.10": bool(all(am(e, N) < 0.10 for e in ("tractor", "pulso") for N in NS)),
        "N3_cociente_tractor": bool(all(
            E[f"tractor|{N}|4"]["adaptado"]["entero"]["rec"] / am("tractor", N) >= 5 for N in NS)),
        "N3_cociente_pulso_definido": bool(all(am("pulso", N) > 0 for N in NS)),
        "N5": bool(all(max(("0.5", "1", "2"), key=lambda t: B[f"{e}|1000"][t]["rec"]) == "1"
                       for e in ("tractor", "pulso"))),
        "T3": bool(ad(250, 4.0) < ad(250, 2.0) < ad(250, 1.0) and ad(250, 1.0) >= 0.75),
        "T4": bool(all(am("pulso", 250, t) <= 0.15 for t in (1.0, 2.0, 4.0))),
        "U1": bool(ad(1000, 1.0) >= 0.85),
        "U2": bool(am("pulso", 1000, 1.0) <= 0.20),
        "V1": bool(all(B[f"{e}|{N}"]["4"]["rec"] < B[f"{e}|{N}"]["2"]["rec"]
                       for e in ("tractor", "pulso") for N in (250, 1000))),
        "V2": bool(all(B[f"{e}|250"]["0.25"]["rec"] < B[f"{e}|250"]["0.5"]["rec"]
                       for e in ("tractor", "pulso"))),
        "E20_mas_de_0.25": bool(all(ad(N) > 0.25 for N in NS)),
    }
    esperado = {"N2": False, "N3_sin_memoria_bajo_0.10": True, "N3_cociente_tractor": True,
                "N3_cociente_pulso_definido": False, "N5": False, "T3": True, "T4": True,
                "U1": True, "U2": True, "V1": True, "V2": True, "E20_mas_de_0.25": True}
    pr["A6"] = {"reevaluadas": viejas, "esperado": esperado, "cumple": bool(viejas == esperado)}
    P("\n=== Predicciones declaradas ===")
    for k, v in pr.items():
        P(f'{k}: {"CUMPLE" if v["cumple"] else "FALLA"}  {json.dumps(v, ensure_ascii=False)[:600]}')
    out["segundos"] = seg
    P(f"\n{seg:.0f} s de corridas nuevas (secuencial)")
    with open(F_JSON, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    with open(F_TXT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
