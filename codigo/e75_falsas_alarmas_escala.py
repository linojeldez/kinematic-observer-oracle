"""Etapa 75 (revisión del 14-nov, objeción 5): persistencia con falsas alarmas según el intervalo
de medición, en exceso de difusión, contra CoLd con las mismas alarmas, y cohesión.

Sólo lectura de las series de rumbo guardadas de E64 (sin falsas alarmas) y E72 (con lambda y
10 lambda): rumbo cada 0,5 s durante 600 s, semillas 11 + réplica comunes a todas las condiciones.
No corre el motor.

  D(L) = <dPhi^2>/L, todos los orígenes;  r = P/P_ViSt = D_ViSt(sin alarmas)/D_x
  c_P = (1 - r_x)/(1 - r_or)   (fracción de la pérdida de P del oráculo; la de E72)
  c_D = (D_x - D_ViSt)/(D_or - D_ViSt)   (fracción del exceso de difusión del oráculo)
  delta_rms = sqrt((D_x - D_ViSt) T / n_FA)   (giro cuadrático medio del rumbo por falsa alarma)

Diseño y predicciones F1-F6 en docs/bitacora/resultados_E66-E75.md («E75»).

    PYTHONPATH=codigo python codigo/e75_falsas_alarmas_escala.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

DT, T = 0.5, 600.0
LAGS = (1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0)
L_MAIN, L_LARGO = 10.0, 60.0
N_BOOT = 2000
RAIZ = os.environ.get("E75_RAIZ") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.environ.get("E75_SALIDA", "resultados")
F_JSON, F_TXT = os.path.join(SALIDA, "E75_falsas_alarmas_escala.json"), os.path.join(SALIDA, "E75_falsas_alarmas_escala.txt")
SIN = ("adaptado_pi0.173", "adaptado_pi0.05", "adaptado_pi0.3", "amplitud_pi0.173", "inst_m6_pi0.173", "inst_m12_pi0.173")


def arr(xs):
    return np.array([np.asarray(x["phi"], float) for x in sorted(xs, key=lambda y: y["rep"])])


def D(A, lag):
    j = int(round(lag / DT))
    return float(np.mean((A[:, j:] - A[:, :-j]) ** 2) / lag)


def ic(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def k(lag):
    return f"{lag:g}"


if __name__ == "__main__":
    os.chdir(RAIZ)
    e64 = json.load(open("resultados/E64_persistencia_crudo.json"))["P"]
    e72 = json.load(open("resultados/E72_persistencia_fa_crudo.json"))["P"]
    V = arr([x for x in e64 if x["nom"] == "ViSt"])                    # ViSt sin falsas alarmas
    out = {"lags": list(LAGS), "n_boot": N_BOOT, "x1": {}, "x10": {}, "sin_alarmas": {}}
    L = ["=== E75 · falsas alarmas: intervalo, exceso de difusión, contraste con CoLd y cohesión ===",
         "series de E64 y E72 (600 s, rumbo cada 0,5 s); bootstrap pareado de réplicas, 2000 remuestreos"]

    # ---------------- con falsas alarmas
    for mult in (1, 10):
        S = {}
        for x in e72:
            if x["mult"] == mult:
                S.setdefault(x["nom"], []).append(x)
        n = len(S["oraculo"])
        A = {nom: arr(xs) for nom, xs in S.items()}
        Vn = V[:n]
        rng = np.random.default_rng(750 + mult)
        B = rng.integers(0, n, (N_BOOT, n))
        Dv = {k(l): D(Vn, l) for l in LAGS}
        Dvb = {k(l): np.array([D(Vn[b], l) for b in B]) for l in LAGS}
        Db = {nom: {k(l): np.array([D(A[nom][b], l) for b in B]) for l in LAGS} for nom in A}
        Dx = {nom: {k(l): D(A[nom], l) for l in LAGS} for nom in A}
        L.append(f"\n--- lambda x{mult} ({n} réplicas; falsas alarmas por corrida: {np.mean([x['n_fp'] for x in S['oraculo']]):.1f}) ---")
        L.append("D_ViSt sin alarmas por intervalo: " + "  ".join(f"{l:g} s {Dv[k(l)]:.3e}" for l in LAGS))
        for nom in A:
            q = {"n": n, "D": Dx[nom], "r": {}, "r_or": {}, "r_cold": {}, "cP": {}, "cD": {}}
            for l in LAGS:
                kk = k(l)
                d, db = Dx[nom][kk], Db[nom][kk]
                do, dob = Dx["oraculo"][kk], Db["oraculo"][kk]
                dc, dcb = Dx["CoLd"][kk], Db["CoLd"][kk]
                q["r"][kk] = {"v": Dv[kk] / d, "ic": ic(Dvb[kk] / db)}
                q["r_or"][kk] = {"v": do / d, "ic": ic(dob / db)}
                q["r_cold"][kk] = {"v": dc / d, "ic": ic(dcb / db)}
                if nom != "oraculo":
                    q["cP"][kk] = {"v": (1 - Dv[kk] / d) / (1 - Dv[kk] / do), "ic": ic((1 - Dvb[kk] / db) / (1 - Dvb[kk] / dob))}
                    q["cD"][kk] = {"v": (d - Dv[kk]) / (do - Dv[kk]), "ic": ic((db - Dvb[kk]) / (dob - Dvb[kk]))}
            cm = np.array([x["coh_min"] for x in sorted(S[nom], key=lambda y: y["rep"])])
            nfp = np.array([x["n_fp"] for x in sorted(S[nom], key=lambda y: y["rep"])], float)
            ok = cm >= 0.5
            q.update({"parten": int((~ok).sum()), "coh_min_mediana": float(np.median(cm)), "coh_min_min": float(cm.min()),
                      "pol": float(np.mean([x["pol"] for x in S[nom]])), "n_fp": float(nfp.mean())})
            if ok.sum() >= 3:
                q["r_enteras"] = {"v": D(Vn, L_MAIN) / D(A[nom][ok], L_MAIN), "n": int(ok.sum())}
            ex, exb = Dx[nom][k(L_LARGO)] - Dv[k(L_LARGO)], Db[nom][k(L_LARGO)] - Dvb[k(L_LARGO)]
            if nfp.mean() > 0 and nom != "ViSt":
                q["delta_rms"] = {"v": float(np.sqrt(max(ex, 0) * T / nfp.mean())),
                                  "ic": ic(np.sqrt(np.clip(exb, 0, None) * T / nfp.mean()))}
            out[f"x{mult}"][nom] = q
            L.append(f"{nom:18s} " + "  ".join(f"{l:g}s {q['r'][k(l)]['v']:.3f}" for l in LAGS)
                     + f" | 10 s: P/P_ViSt {q['r']['10']['v']:.3f} [{q['r']['10']['ic'][0]:.3f}; {q['r']['10']['ic'][1]:.3f}]"
                     + f"  P/P_CoLd {q['r_cold']['10']['v']:.3f} [{q['r_cold']['10']['ic'][0]:.3f}; {q['r_cold']['10']['ic'][1]:.3f}]"
                     + (f"  c_P {q['cP']['10']['v']:.3f}  c_D {q['cD']['10']['v']:.3f} [{q['cD']['10']['ic'][0]:.3f}; {q['cD']['10']['ic'][1]:.3f}]" if q["cP"] else "")
                     + f" | parten {q['parten']}/{n}  coh_min med {q['coh_min_mediana']:.2f}  pol {q['pol']:.3f}"
                     + (f"  sin las partidas {q['r_enteras']['v']:.3f} (n = {q['r_enteras']['n']})" if "r_enteras" in q else "")
                     + (f"  delta_rms {q['delta_rms']['v']:.2f} rad" if "delta_rms" in q else ""))
        for nom in ("oraculo", "adaptado_pi0.173", "adaptado_pi0.05"):
            q = out[f"x{mult}"][nom]
            L.append(f"  {nom}: P/P_or por intervalo " + "  ".join(
                f"{l:g}s {q['r_or'][k(l)]['v']:.2f} [{q['r_or'][k(l)]['ic'][0]:.2f}; {q['r_or'][k(l)]['ic'][1]:.2f}]" for l in LAGS)
                + (" | c_D " + "  ".join(f"{l:g}s {q['cD'][k(l)]['v']:.2f}" for l in LAGS) if q["cD"] else ""))
        if mult == 1:
            mes = {nom: {"v": Dx[nom]["120"] / Dx[nom]["20"], "ic": ic(Db[nom]["120"] / Db[nom]["20"]),
                         "sube": Dx[nom]["20"] / Dx[nom]["1"]} for nom in ("oraculo", "adaptado_pi0.173", "adaptado_pi0.05")}
            mes["ViSt_sin"] = {"v": Dv["120"] / Dv["10"], "ic": ic(Dvb["120"] / Dvb["10"])}
            out["meseta"] = mes
            xs = sorted(S["oraculo"], key=lambda y: y["rep"])
            rho, p = spearmanr([x["n_fp"] for x in xs], [x["coh_min"] for x in xs])
            out["spearman_alarmas_cohesion"] = {"rho": float(rho), "p": float(p), "n": len(xs)}

    # ---------------- sin falsas alarmas (E64): ¿depende la cifra de H2 del intervalo?
    rng = np.random.default_rng(752)
    n = len(V)
    B = rng.integers(0, n, (N_BOOT, n))
    L.append("\n--- sin falsas alarmas (E64): P/P_ViSt por intervalo ---")
    for nom in ("ViSt",) + SIN:
        xs = [x for x in e64 if x["nom"] == nom]
        A0 = arr(xs)
        m = min(len(A0), n)
        Bm = B[:, :m] % m if m < n else B
        q = {"n": m, "r": {}, "coh_media": float(np.mean([x["coh"] for x in xs])), "coh_min": float(np.min([x["coh"] for x in xs])),
             "pol": float(np.mean([x["pol"] for x in xs])),
             "finales_partidas": int(sum(x["coh"] < 0.99 for x in xs)), "n_corridas": len(xs)}
        for l in LAGS:
            q["r"][k(l)] = {"v": D(V[:m], l) / D(A0, l), "ic": ic(np.array([D(V[:m][b], l) / D(A0[b], l) for b in Bm]))}
        out["sin_alarmas"][nom] = q
        L.append(f"{nom:18s} n {m:2d}  " + "  ".join(f"{l:g}s {q['r'][k(l)]['v']:.3f} [{q['r'][k(l)]['ic'][0]:.3f}; {q['r'][k(l)]['ic'][1]:.3f}]"
                                                    for l in (2.0, 10.0, 60.0)) + f"  cohesión final media {q['coh_media']:.3f} (mín {q['coh_min']:.2f}; partidas al final {q['finales_partidas']}/{q['n_corridas']})")

    # ---------------- predicciones
    X = out["x1"]
    mes = out["meseta"]
    i173, i05, orc = X["adaptado_pi0.173"], X["adaptado_pi0.05"], X["oraculo"]
    s173 = out["sin_alarmas"]["adaptado_pi0.173"]
    out["pred"] = {
        "F1": {"D120_sobre_D20": {nom: mes[nom]["v"] for nom in ("oraculo", "adaptado_pi0.173", "adaptado_pi0.05")},
               "D20_sobre_D1": {nom: mes[nom]["sube"] for nom in ("oraculo", "adaptado_pi0.173", "adaptado_pi0.05")},
               "ViSt_D120_sobre_D10": mes["ViSt_sin"]["v"],
               "a": bool(all(0.75 <= mes[nom]["v"] <= 1.25 and mes[nom]["sube"] > 1 for nom in ("oraculo", "adaptado_pi0.173", "adaptado_pi0.05"))),
               "b": bool(0.7 <= mes["ViSt_sin"]["v"] <= 1.2)},
        "F2": {"P_int_sobre_P_or": {kk: i173["r_or"][kk]["v"] for kk in i173["r_or"]},
               "a": bool(all(v["v"] >= 2 for v in i173["r_or"].values())),
               "b": bool(all(i173["r_or"][k(l)]["ic"][0] > 1 for l in (2.0, 5.0, 10.0, 20.0, 60.0)))},
        "F3": {"oraculo": orc["r_cold"]["10"], "integrador": i173["r_cold"]["10"],
               "a": bool(orc["r_cold"]["10"]["ic"][1] < 0.08),
               "b": bool(0.07 <= i173["r_cold"]["10"]["ic"][0] and i173["r_cold"]["10"]["ic"][1] <= 0.25)},
        "F4": {"cD_0.173": {kk: i173["cD"][kk]["v"] for kk in ("5", "20", "120")}, "cD_0.05": {kk: i05["cD"][kk]["v"] for kk in ("5", "20", "120")},
               "a": bool(all(0.20 <= i173["cD"][kk]["v"] <= 0.45 for kk in ("5", "20", "120"))),
               "b": bool(all(0.15 <= i05["cD"][kk]["v"] <= 0.50 for kk in ("5", "20", "120")))},
        "F5": {**out["spearman_alarmas_cohesion"], "a": bool(out["spearman_alarmas_cohesion"]["rho"] <= -0.2)},
        "F6": {"r": {kk: s173["r"][kk] for kk in ("2", "10", "60")},
               "a": bool(all(s173["r"][kk]["v"] >= 0.9 and s173["r"][kk]["ic"][0] > 0.8 for kk in ("2", "10", "60")))}}
    for kk, v in out["pred"].items():
        v["cumple"] = bool(all(v[c] for c in ("a", "b") if c in v))
    L.append("\n=== Predicciones ===")
    L += [f"{kk}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v, default=float)}" for kk, v in out["pred"].items()]
    json.dump(out, open(F_JSON, "w"), indent=1, default=float)
    open(F_TXT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
