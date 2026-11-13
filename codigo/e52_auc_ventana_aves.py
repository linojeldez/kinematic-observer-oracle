"""Etapa 52 (A2 del plan de fortalecimiento): ¿integrar ayuda también en el ave?

Curva AUC iniciador/seguidor frente a la ventana previa al inicio colectivo, con un solo
estimador (amplitud media de |eps|/sigma en la ventana, `amp_pre` de E23) y un solo k (3; el
7 como sensibilidad). IC por remuestreo de eventos, nulo por permutación dentro de evento,
diferencia pareada AUC(1 s) - AUC(un cuadro), por tipo de vuelo y dejando un evento fuera.

Diseño y predicción en docs/bitacora/resultados_E46-E55.md, antes de mirar los datos.

    PYTHONPATH=codigo python codigo/e52_auc_ventana_aves.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import turns
from e16_helpers import campos, omega, residuos, inicio_colectivo, inicio_individual

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q, N_BOOT, N_PERM = 0.25, 10000, 4000
VENTANAS = ("cuadro", 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0)


def por_evento(ctx, ev, k):
    """Para cada ave etiquetada (ini / seg) con la selección de E23, la amplitud media en
    cada ventana previa al inicio colectivo."""
    e = turns.cargar(ctx, ev)
    ts, P, V = campos(e)
    dt = float(np.median(np.diff(ts)))
    eps = residuos(P, omega(V, dt), k)
    s_col = inicio_colectivo(V)
    base = eps[:s_col] if s_col and s_col > 20 else eps
    fb = np.isfinite(base)
    if fb.sum() < 100 or s_col is None:
        return None
    sig = float(np.median(np.abs(base[fb] - np.median(base[fb]))) * 1.4826)
    d = e["delay"]
    ok = np.isfinite(d)
    if ok.sum() < 8:
        return None
    lim_i, lim_s = np.quantile(d[ok], Q), np.quantile(d[ok], 1 - Q)
    n1 = int(round(1.0 / dt))
    et, st = [], {w: [] for w in VENTANAS}
    for a in np.flatnonzero(ok):
        g = 1 if d[a] <= lim_i else (0 if d[a] >= lim_s else None)
        if g is None:
            continue
        s0 = inicio_individual(V, a)                      # la selección de aves de E23
        if s0 is None or s0 + n1 >= len(ts):
            continue
        tramo = np.abs(eps[s0:s0 + n1, a])
        if np.isfinite(tramo).mean() < 0.7:
            continue
        et.append(g)
        for w in VENTANAS:
            n = 1 if w == "cuadro" else int(round(w / dt))
            if s_col - n < 0:
                st[w].append(np.nan)
                continue
            pre = np.abs(eps[s_col - n:s_col, a])
            st[w].append(float(np.nanmean(pre) / sig) if np.isfinite(pre).mean() >= 0.7 else np.nan)
    if len(et) < 4:
        return None
    return {"ev": f"{ctx} {ev}", "ctx": ctx, "t_col": float(s_col * dt), "et": np.array(et),
            "st": {w: np.array(v) for w, v in st.items()}, "N": int(len(e["ids"]))}


def auc(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    return float(((x1[:, None] > x0[None, :]).sum()
                  + 0.5 * (x1[:, None] == x0[None, :]).sum()) / (len(x1) * len(x0)))


def agrupada(evs, w, idx=None, et=None):
    """AUC agrupada sobre eventos (índices con repetición `idx`; etiquetas alternativas)."""
    x1, x0 = [], []
    for j, i in enumerate(idx if idx is not None else range(len(evs))):
        v = evs[i]["st"][w]
        lab = et[i] if et is not None else evs[i]["et"]
        x1.append(v[lab == 1]); x0.append(v[lab == 0])
    return auc(np.concatenate(x1), np.concatenate(x0))


def analizar(evs, w_larga, w_corta, rng):
    """Curva, IC, nulo y diferencia pareada sobre los eventos donde existe w_larga."""
    n_ev = len(evs)
    out = {"n_eventos": n_ev, "eventos": [x["ev"] for x in evs],
           "n_ini": int(sum((x["et"] == 1).sum() for x in evs)),
           "n_seg": int(sum((x["et"] == 0).sum() for x in evs)), "curva": {}}
    ws = [w for w in VENTANAS if w == "cuadro" or w <= w_larga]
    B = rng.integers(0, n_ev, (N_BOOT, n_ev))
    boot = {w: np.array([agrupada(evs, w, b) for b in B]) for w in ws}
    perm_et = [[rng.permutation(x["et"]) for x in evs] for _ in range(N_PERM)]
    for w in ws:
        a = agrupada(evs, w)
        nulo = np.array([agrupada(evs, w, et=pe) for pe in perm_et])
        out["curva"][str(w)] = {"auc": a, "ic95": [float(np.nanpercentile(boot[w], 2.5)),
                                                    float(np.nanpercentile(boot[w], 97.5))],
                                "p_perm": float((1 + (nulo >= a).sum()) / (1 + N_PERM))}
    d = boot[w_larga] - boot[w_corta]
    delta = agrupada(evs, w_larga) - agrupada(evs, w_corta)
    loo = [agrupada(evs, w_larga, [i for i in range(n_ev) if i != j])
           - agrupada(evs, w_corta, [i for i in range(n_ev) if i != j]) for j in range(n_ev)]
    loo_a = [agrupada(evs, w_larga, [i for i in range(n_ev) if i != j]) for j in range(n_ev)]
    out["delta"] = {"larga": str(w_larga), "corta": str(w_corta), "delta": float(delta),
                    "ic95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                    "frac_boot_positiva": float(np.nanmean(d > 0)),
                    "dejando_uno_fuera": [float(min(loo)), float(max(loo))],
                    "auc_larga_dejando_uno_fuera": [float(min(loo_a)), float(max(loo_a))]}
    # ventana más corta dentro del 95 % del máximo de la curva, con IC
    def tau95(curva):
        fin = {w: a for w, a in curva.items() if w != "cuadro" and np.isfinite(a)}
        tope = max(fin.values())
        return min(w for w, a in fin.items() if a >= 0.95 * tope)
    t_obs = tau95({w: out["curva"][str(w)]["auc"] for w in ws})
    t_b = [tau95({w: boot[w][i] for w in ws}) for i in range(0, N_BOOT, 5)]
    out["tau95"] = {"valor": float(t_obs), "ic95": [float(np.percentile(t_b, 2.5)),
                                                    float(np.percentile(t_b, 97.5))]}
    return out


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(52)
    L = ["=== E52 · AUC iniciador/seguidor frente a la ventana previa al inicio colectivo ==="]
    P = L.append
    out = {}
    for k in (3, 7):
        evs = [por_evento(c, e, k) for c, e in turns.eventos()]
        evs = [x for x in evs if x]
        if k == 3:
            P("inicio colectivo desde el comienzo del evento (s): "
              + ", ".join(f'{x["ev"]} {x["t_col"]:.2f}' for x in evs))
        for w_larga in (1.0, 1.5, 2.0):
            sel = [x for x in evs if np.isfinite(x["st"][w_larga]).any()]
            if len(sel) < 4:
                P(f"\nk = {k}, ventana hasta {w_larga} s: sólo {len(sel)} eventos; no se analiza")
                continue
            for nombre, sub in (("todos", sel), ("transito", [x for x in sel if x["ctx"] == "transito"]),
                                ("mobbing", [x for x in sel if x["ctx"] == "mobbing"])):
                if len(sub) < 3 or (nombre != "todos" and w_larga != 1.0):
                    continue
                r = analizar(sub, w_larga, "cuadro", rng)
                out[f"k{k}|{w_larga:g}|{nombre}"] = r
                P(f'\nk = {k}; eventos con al menos {w_larga:g} s antes del inicio colectivo; {nombre}: '
                  f'{r["n_eventos"]} eventos, {r["n_ini"]} iniciadoras y {r["n_seg"]} seguidoras')
                P("  " + " | ".join(f'{w}: {v["auc"]:.3f} [{v["ic95"][0]:.3f}; {v["ic95"][1]:.3f}]'
                                    for w, v in r["curva"].items()))
                P(f'  p de permutación: ' + ", ".join(f'{w}: {v["p_perm"]:.4f}' for w, v in r["curva"].items()))
                dl = r["delta"]
                P(f'  Δ = AUC({w_larga:g} s) − AUC(un cuadro) = {dl["delta"]:+.3f} '
                  f'[{dl["ic95"][0]:+.3f}; {dl["ic95"][1]:+.3f}]; positiva en '
                  f'{100 * dl["frac_boot_positiva"]:.1f} % de los remuestreos; dejando un evento fuera: '
                  f'{dl["dejando_uno_fuera"][0]:+.3f} a {dl["dejando_uno_fuera"][1]:+.3f}; AUC larga dejando '
                  f'uno fuera: {dl["auc_larga_dejando_uno_fuera"][0]:.3f} a {dl["auc_larga_dejando_uno_fuera"][1]:.3f}')
                P(f'  ventana más corta dentro del 95 % del máximo: {r["tau95"]["valor"]:g} s '
                  f'{r["tau95"]["ic95"]}')
    # sensibilidad no declarada (añadida después de ver el resultado): todos los eventos con
    # al menos 0,7 s antes del inicio colectivo, ventanas hasta 0,7 s
    for k in (3, 7):
        evs = [x for x in (por_evento(c, e, k) for c, e in turns.eventos()) if x]
        sel = [x for x in evs if np.isfinite(x["st"][0.7]).any()]
        r = analizar(sel, 0.7, "cuadro", rng)
        out[f"k{k}|0.7|todos"] = r
        dl = r["delta"]
        P(f'\n[sensibilidad no declarada] k = {k}; eventos con al menos 0,7 s antes del inicio '
          f'colectivo: {r["n_eventos"]} eventos, {r["n_ini"]} + {r["n_seg"]} aves')
        P("  " + " | ".join(f'{w}: {v["auc"]:.3f} [{v["ic95"][0]:.3f}; {v["ic95"][1]:.3f}]'
                            for w, v in r["curva"].items()))
        P(f'  Δ = AUC(0,7 s) − AUC(un cuadro) = {dl["delta"]:+.3f} [{dl["ic95"][0]:+.3f}; '
          f'{dl["ic95"][1]:+.3f}]; dejando un evento fuera: {dl["dejando_uno_fuera"][0]:+.3f} a '
          f'{dl["dejando_uno_fuera"][1]:+.3f}')
    r = out.get("k3|1|todos")
    if r:
        lo, hi = r["delta"]["ic95"]
        out["decision"] = {"delta": r["delta"]["delta"], "ic95": [lo, hi],
                           "ic_contiene_cero": bool(lo <= 0 <= hi)}
        P(f'\n=== regla del plan === el intervalo de Δ (k = 3, 1 s frente a un cuadro) '
          f'{"contiene" if lo <= 0 <= hi else "excluye"} el cero')
    with open("resultados/E52_auc_ventana_aves.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    with open("resultados/E52_auc_ventana_aves.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
