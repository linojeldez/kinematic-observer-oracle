"""Etapa 51 (A1 del plan de fortalecimiento): la huella en grajillas con una definición
que no dependa de un umbral fijo que la curva no cruza.

Estimadores (declarados en docs/bitacora/resultados_E46-E55.md antes de mirar los datos):
  principal   T95: tiempo, en los 2 s que siguen al inicio propio de un iniciador, en que
              su |eps|/sigma suavizado (0,2 s) supera el percentil 95 de las aves no
              iniciadoras del mismo evento en esos mismos cuadros. Nulo: 0,10 s.
  grupo       la curva mediana de los iniciadores frente al percentil 95 de curvas medianas
              de seudogrupos de no iniciadoras (sensibilidad).
  secundario  anchura a media altura del exceso de la curva mediana sobre su línea base.
  terciario   el operador anterior (`e16_helpers.duracion`) con umbrales de 1,0 a 1,5 sigma.
Lo mismo (T95 y anchura) en los modelos: el agente con intención frente a los agentes sin
etiqueta de la misma corrida, con ruido gaussiano y con ruido de cola pesada.

    PYTHONPATH=codigo python codigo/e51_huella_aves.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

import turns
from e16_helpers import (campos, omega, residuos, inicio_colectivo, inicio_individual,
                         huella_evento, duracion, cinematica)

T_SUAV, T_POST, T_PRE, Q_INI = 0.2, 2.0, 0.3, 0.25
N_BOOT, N_SEUDO = 10000, 200
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def suavizar(x, n):
    """Media móvil centrada de n muestras por columnas, tolerante a NaN."""
    k = np.ones(n)
    f = np.isfinite(x)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, np.where(f, x, 0.0))
    den = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, f.astype(float))
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def fwhm(t, curva, n_suav):
    """Anchura a media altura del exceso sobre la línea base previa (t < 0).

    Devuelve (anchura, censurada_derecha, pico_exceso, t_pico)."""
    sm = suavizar(curva[:, None], n_suav)[:, 0]
    pre = sm[t < 0]
    base = float(np.nanmedian(pre)) if np.isfinite(pre).any() else float(np.nanmedian(sm))
    exc = sm - base
    pos = np.flatnonzero(t >= 0)
    ip = pos[int(np.nanargmax(exc[pos]))]
    if not exc[ip] > 0:
        return float("nan"), False, float(exc[ip]), float(t[ip])
    medio = 0.5 * exc[ip]
    i, j = ip, ip
    while i > 0 and exc[i - 1] >= medio:
        i -= 1
    while j < len(exc) - 1 and exc[j + 1] >= medio:
        j += 1
    return float(t[j] - t[i]), bool(j == len(exc) - 1), float(exc[ip]), float(t[ip])


def evento(ctx, ev, k=7, rng=None):
    e = turns.cargar(ctx, ev)
    ts, P, V = campos(e)
    dt = float(np.median(np.diff(ts)))
    eps = residuos(P, omega(V, dt), k)
    if np.isfinite(eps).sum() < 200:
        return None
    s_col = inicio_colectivo(V)
    base = eps[:s_col] if s_col and s_col > 20 else eps
    fb = np.isfinite(base)
    sig = float(np.median(np.abs(base[fb] - np.median(base[fb]))) * 1.4826)
    n_suav = max(3, int(round(T_SUAV / dt)))
    A = suavizar(np.abs(eps) / sig, n_suav)                # (cuadros, aves)
    d = e["delay"]
    ok = np.isfinite(d)
    lim = np.quantile(d[ok], Q_INI)
    ini = [a for a in np.flatnonzero(ok) if d[a] <= lim]
    no_ini = np.array([a for a in np.flatnonzero(ok) if d[a] > lim])
    n_pre, n_post = int(round(T_PRE / dt)), int(round(T_POST / dt))

    vent, t95, curvas = [], [], []
    for a in ini:
        s0 = inicio_individual(V, a)
        if s0 is None or s0 - n_pre < 0 or s0 + n_post >= len(ts):
            continue
        ref = A[s0:s0 + n_post][:, no_ini]
        thr = np.nanpercentile(ref, 95)
        t95.append(float(np.nansum(A[s0:s0 + n_post, a] > thr) * dt))
        vent.append(s0)
        curvas.append(A[s0:s0 + n_post, a])
    out = {"ev": f"{ctx} {ev}", "ctx": ctx, "N": int(len(e["ids"])), "dt": dt, "sigma": sig,
           "n_ini": len(ini), "n_ini_validos": len(t95), "n_no_ini": int(len(no_ini))}
    if len(t95) < 3:
        out["evaluable"] = False
        return out
    out["evaluable"] = True
    out["T95"] = float(np.median(t95))
    out["T95_por_ave"] = t95
    # nulo: cada no iniciadora como foco, en las ventanas de los iniciadores
    nulo = []
    for m in no_ini:
        resto = no_ini[no_ini != m]
        for s0 in vent:
            thr = np.nanpercentile(A[s0:s0 + n_post][:, resto], 95)
            nulo.append(float(np.nansum(A[s0:s0 + n_post, m] > thr) * dt))
    out["T95_nulo_media"] = float(np.mean(nulo))
    out["T95_nulo_mediana"] = float(np.median(nulo))
    out["T95_nulo_p95"] = float(np.percentile(nulo, 95))
    # variante de grupo
    cur = np.nanmedian(np.vstack(curvas), axis=0)
    seudo = []
    for _ in range(N_SEUDO):
        ms = rng.choice(no_ini, size=len(vent), replace=len(no_ini) < len(vent))
        seudo.append(np.nanmedian(np.vstack([A[s0:s0 + n_post, m] for s0, m in zip(vent, ms)]),
                                  axis=0))
    thr_g = np.nanpercentile(np.vstack(seudo), 95)
    out["T95_grupo"] = float(np.nansum(cur > thr_g) * dt)
    out["umbral_grupo"] = float(thr_g)
    # secundario y terciario, sobre la curva de huella_evento (sin cambios respecto de E16)
    h = huella_evento(e, k=k)
    if h and "ini" in h:
        w, cens, pico, tp = fwhm(h["t"], h["ini"], n_suav)
        out.update({"fwhm": w, "fwhm_censurada": cens, "exceso_pico": pico, "t_pico": tp,
                    "pico_sigma": float(np.nanmax(h["ini"])),
                    "dur": {f"{u:.1f}": duracion(h["t"], h["ini"], umbral=u)
                            for u in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)},
                    "_t": h["t"], "_ini": h["ini"]})
    cin = cinematica(e)
    if cin:
        out["amplitud_deg"] = float(np.rad2deg(np.median([c["dphi"] for c in cin])))
    return out


def ic_mediana(v, rng):
    v = np.asarray(v, float)
    b = rng.integers(0, len(v), (N_BOOT, len(v)))
    m = np.median(v[b], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ----------------------------------------------------------------------------
# Los modelos, con el mismo estimador
# ----------------------------------------------------------------------------

def modelo(arg):
    from flocking import Flock, Params
    m, rep, ruido = arg
    CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
    q = Params(N=200, k=12, sigma_xi=float(np.sqrt(2.0)), seed=11 + rep, **CFG,
               **(dict(noise_dist="student", noise_nu=3.0) if ruido == "student" else {}))
    f = Flock(q, m)
    for _ in range(int(round(6.0 / q.dt))):
        f.step()
    base = []
    for _ in range(int(round(2.0 / q.dt))):
        f.step()
        i, _ = f._neighbors()
        base.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    base = np.array(base)
    sig = float(np.median(base) * 1.4826)
    ag, ang = 7, 2.0
    f.will[ag] = 1.0
    f.d_des[ag] = np.array([np.cos(ang), np.sin(ang)])
    serie = []
    for s in range(int(round(5.0 / q.dt))):
        f.step()
        f.will[ag] = 1.0
        if s % 5 == 0:
            i, _ = f._neighbors()
            serie.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
    S = np.array(serie) / sig                               # (100, N), cada 0,05 s
    t = np.arange(len(S)) * 0.05
    w = max(3, len(S) // 25)                                # 4 muestras = 0,2 s (e15_helpers)
    sm = np.apply_along_axis(lambda a: np.convolve(a, np.ones(w) / w, mode="same"), 0, S)
    otros = np.array([j for j in range(q.N) if j != ag])
    out = {"modelo": m, "rep": rep, "ruido": ruido, "sigma": sig}
    for nom, n in (("2s", 40), ("5s", 100)):
        thr = np.percentile(sm[:n][:, otros], 95)
        out[f"T95_{nom}"] = float((sm[:n, ag] > thr).sum() * 0.05)
        out[f"umbral_{nom}"] = float(thr)
        nul = [(sm[:n, j] > np.percentile(sm[:n][:, otros[otros != j]], 95)).sum() * 0.05
               for j in otros[:40]]
        out[f"T95_nulo_{nom}"] = float(np.mean(nul))
    # el operador anterior, para anclar contra E47
    thr15 = 1.5
    ip = int(np.argmax(sm[:, ag]))
    d = float(t[-1])
    for j in range(ip, len(sm)):
        if sm[j, ag] < thr15 and np.all(sm[j:j + w, ag] < thr15):
            d = float(t[j])
            break
    out["dur_15"] = d
    out["pico"] = float(sm[:, ag].max())
    # anchura a media altura: línea base = mediana del propio agente antes de la activación
    b0 = float(np.median(base[:, ag]) / sig)
    exc = sm[:, ag] - b0
    medio = 0.5 * exc[ip]
    i, j = ip, ip
    while i > 0 and exc[i - 1] >= medio:
        i -= 1
    while j < len(exc) - 1 and exc[j + 1] >= medio:
        j += 1
    out["fwhm"] = float(t[j] - t[i])
    out["fwhm_censurada_2s"] = bool(t[j] >= 1.95)
    return out


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(51)
    L = ["=== E51 · la huella en grajillas, familia de estimadores ==="]
    P = L.append
    out = {"config": {"t_suav": T_SUAV, "t_post": T_POST, "q_ini": Q_INI, "n_boot": N_BOOT}}

    for k in (7, 3, 12):
        evs = [evento(ctx, ev, k=k, rng=rng) for ctx, ev in turns.eventos()]
        evs = [x for x in evs if x is not None]
        for min_aves in ((12, 8) if k == 7 else (12,)):
            sel = [x for x in evs if x["N"] >= min_aves]
            val = [x for x in sel if x.get("evaluable")]
            clave = f"k{k}_min{min_aves}"
            P(f"\n=== k = {k}; eventos con al menos {min_aves} aves: {len(sel)}; evaluables "
              f"(≥ 3 iniciadores con ventana válida): {len(val)} ===")
            if k == 7:
                P(f'{"evento":<12s} {"aves":>4s} {"ini":>3s} {"ampl.":>6s} {"T95":>6s} {"nulo":>6s} '
                  f'{"grupo":>6s} {"anch.":>6s} {"cens":>4s} {"pico":>5s} {"dur1,5":>6s}')
                for x in sel:
                    if not x.get("evaluable"):
                        P(f'{x["ev"]:<12s} {x["N"]:4d} {x["n_ini_validos"]:3d}  no evaluable')
                        continue
                    P(f'{x["ev"]:<12s} {x["N"]:4d} {x["n_ini_validos"]:3d} '
                      f'{x.get("amplitud_deg", float("nan")):6.0f} {x["T95"]:6.2f} '
                      f'{x["T95_nulo_media"]:6.2f} {x["T95_grupo"]:6.2f} '
                      f'{x.get("fwhm", float("nan")):6.2f} {"sí" if x.get("fwhm_censurada") else "no":>4s} '
                      f'{x.get("pico_sigma", float("nan")):5.2f} {x.get("dur", {}).get("1.5", float("nan")):6.2f}')
            if len(val) < 3:
                continue
            r = {"n_eventos": len(sel), "n_evaluables": len(val)}
            for nom in ("T95", "T95_nulo_media", "T95_grupo", "fwhm"):
                v = [x[nom] for x in val if nom in x and np.isfinite(x[nom])]
                r[nom] = {"mediana": float(np.median(v)), "ic95": ic_mediana(v, rng), "n": len(v),
                          "por_evento": [float(a) for a in v]}
            r["fwhm_censuradas"] = int(sum(bool(x.get("fwhm_censurada")) for x in val))
            aves = np.concatenate([x["T95_por_ave"] for x in val])
            r["T95_por_ave"] = {"mediana": float(np.median(aves)), "media": float(aves.mean()),
                                "n_aves": int(len(aves)),
                                "frac_sobre_nulo_p95": float(np.mean(
                                    [np.mean(np.array(x["T95_por_ave"]) > x["T95_nulo_p95"])
                                     for x in val]))}
            # curva agregada (como E16) y sus estimadores
            cur = [x["_ini"] for x in val if "_ini" in x]
            Lm = min(len(c) for c in cur)
            I = np.nanmedian(np.vstack([c[:Lm] for c in cur]), axis=0)
            T = val[0]["_t"][:Lm]
            n_s = max(3, int(round(T_SUAV / val[0]["dt"])))
            w, cens, pico, tp = fwhm(T, I, n_s)
            r["agregada"] = {"fwhm": w, "fwhm_censurada": cens, "exceso_pico": pico, "t_pico": tp,
                             "pico_sigma": float(I.max()),
                             "dur": {f"{u:.1f}": duracion(T, I, umbral=u)
                                     for u in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)}}
            out[clave] = r
            P(f'T95: mediana {r["T95"]["mediana"]:.2f} s, IC 95 % {r["T95"]["ic95"]} '
              f'(nulo medido {r["T95_nulo_media"]["mediana"]:.2f} s); por ave: mediana '
              f'{r["T95_por_ave"]["mediana"]:.2f} s, media {r["T95_por_ave"]["media"]:.2f} s, '
              f'{r["T95_por_ave"]["n_aves"]} iniciadoras; fracción de iniciadoras sobre el p95 del '
              f'nulo de su evento: {r["T95_por_ave"]["frac_sobre_nulo_p95"]:.2f}')
            P(f'T95 de grupo: mediana {r["T95_grupo"]["mediana"]:.2f} s, IC {r["T95_grupo"]["ic95"]}')
            P(f'anchura a media altura: mediana {r["fwhm"]["mediana"]:.2f} s, IC {r["fwhm"]["ic95"]}; '
              f'censuradas por la ventana: {r["fwhm_censuradas"]} de {r["fwhm"]["n"]}')
            P(f'curva agregada: pico {I.max():.2f} sigma; anchura {w:.2f} s '
              f'({"censurada" if cens else "no censurada"}); operador anterior por umbral: '
              + ", ".join(f"{u} σ: {d:.2f} s" for u, d in r["agregada"]["dur"].items()))
            if k == 7 and min_aves == 12:
                # exploratorio: los dos grupos del operador anterior
                d15 = np.array([x["dur"]["1.5"] for x in val])
                corto = d15 < 1.0
                ex = {"n_cortos": int(corto.sum()), "n_largos": int((~corto).sum())}
                for nom, f in (("N", lambda x: x["N"]),
                               ("amplitud_deg", lambda x: x.get("amplitud_deg", np.nan)),
                               ("T95", lambda x: x["T95"]), ("fwhm", lambda x: x.get("fwhm", np.nan))):
                    v = np.array([f(x) for x in val], float)
                    rho, p = spearmanr(v, d15, nan_policy="omit")
                    ex[nom] = {"cortos_mediana": float(np.nanmedian(v[corto])),
                               "largos_mediana": float(np.nanmedian(v[~corto])),
                               "spearman_con_dur15": float(rho), "p": float(p)}
                tr = np.array([x["ctx"] == "transito" for x in val])
                ex["transito_en_cortos"] = int((tr & corto).sum())
                ex["transito_en_largos"] = int((tr & ~corto).sum())
                for nom in ("T95", "fwhm"):
                    v = np.array([x.get(nom, np.nan) for x in val], float)
                    u, p = mannwhitneyu(v[tr], v[~tr])
                    ex[f"{nom}_transito_vs_mobbing"] = {
                        "transito": float(np.nanmedian(v[tr])), "mobbing": float(np.nanmedian(v[~tr])),
                        "p_mann_whitney": float(p)}
                out["exploratorio"] = ex
                P("exploratorio (n = 12): " + json.dumps(ex, ensure_ascii=False))
        for x in evs:
            x.pop("_t", None); x.pop("_ini", None)
        out[f"eventos_k{k}"] = evs

    # ---- modelos
    P("\n=== modelos, mismo estimador (N = 200, k = 12, 16 réplicas, giro de 115°) ===")
    jobs = [(m, rep, ruido) for ruido in ("gauss", "student") for m in ("WiSt", "CoLd", "CaSp")
            for rep in range(16) if not (ruido == "student" and m == "CaSp")]
    with ProcessPoolExecutor(max_workers=8) as ex_:
        res = list(ex_.map(modelo, jobs))
    out["modelos"] = {}
    for ruido in ("gauss", "student"):
        for m in ("WiSt", "CoLd", "CaSp"):
            v = [x for x in res if x["modelo"] == m and x["ruido"] == ruido]
            if not v:
                continue
            r = {}
            for nom in ("T95_2s", "T95_5s", "T95_nulo_2s", "dur_15", "fwhm", "pico", "umbral_2s", "sigma"):
                a = np.array([x[nom] for x in v])
                r[nom] = {"mediana": float(np.median(a)), "ic95": ic_mediana(a, rng),
                          "sd": float(a.std())}
            r["fwhm_censuradas_2s"] = int(sum(x["fwhm_censurada_2s"] for x in v))
            out["modelos"][f"{m}|{ruido}"] = r
            P(f'{m} ({ruido}): T95 en 2 s {r["T95_2s"]["mediana"]:.2f} s {r["T95_2s"]["ic95"]}; en 5 s '
              f'{r["T95_5s"]["mediana"]:.2f} s; nulo {r["T95_nulo_2s"]["mediana"]:.2f} s; umbral '
              f'{r["umbral_2s"]["mediana"]:.2f} σ; anchura {r["fwhm"]["mediana"]:.2f} s '
              f'{r["fwhm"]["ic95"]}; operador anterior {r["dur_15"]["mediana"]:.3f} s; pico '
              f'{r["pico"]["mediana"]:.2f} σ; σ {r["sigma"]["mediana"]:.3f} rad/s')

    # ---- regla de decisión del plan
    r = out.get("k7_min12")
    if r:
        lo, hi = r["T95"]["ic95"]
        cabe = bool(lo >= 0.5 and hi <= 2.0)
        mayoria = bool(r["n_evaluables"] > 0.5 * r["n_eventos"])
        out["decision"] = {"T95": r["T95"]["mediana"], "ic95": [lo, hi], "ic_cabe_en_0.5_2": cabe,
                           "evaluable_en_mayoria": mayoria,
                           "conserva_duracion_en_aves": bool(cabe and mayoria)}
        P(f"\n=== regla del plan === IC de T95 [{lo:.2f}; {hi:.2f}] s: "
          f"{'cabe' if cabe else 'NO cabe'} entre 0,5 y 2 s; evaluable en la mayoría: "
          f"{'sí' if mayoria else 'no'}")
    with open("resultados/E51_huella_aves.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    with open("resultados/E51_huella_aves.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
