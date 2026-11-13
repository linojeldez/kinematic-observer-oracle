"""Etapa 69 (plan 15, punto 6): ¿la recuperación del observador se explica en dos etapas?

Etapa de la bandada: cuánto vale leer al líder en cada instante. Se mide con oráculos cuya
lectura del líder está intervenida (el estado y la dinámica no cambian):
  compuerta T   los vecinos leen w_líder sólo durante los primeros T s (E44, rejilla más fina)
  atenuado c    los vecinos leen c * w_líder durante toda la corrida
  réplica       los vecinos leen, en vez de w_líder, el perfil medio que un observador dado
                puso sobre el líder en su celda (fase 2)
Etapa del observador: el perfil temporal del posterior sobre el líder, w_L(t), que se registra
en cada celda de observador (fase 1) junto con la fracción de excedencias del líder.

Pruebas (docs/bitacora/resultados_E66-E75.md, E69, declaradas antes de correr):
  L1  exposición lineal (la del plan): rec ~ g(T_ef), T_ef = int (w_L - pi0)/(1 - pi0) dt
  L2  réplica: el oráculo que lee el perfil medio del observador recupera lo mismo que él
  L3  respuesta lineal en el tiempo con peso saturante: rec ~ int g'(t) H(w_L(t)) dt, con g la
      curva de la compuerta y H(c) la recuperación del oráculo atenuado
  L4  forma cerrada: el perfil se calcula del ciclo de trabajo medio del líder, p(t), y de los
      parámetros del observador (tau_det, n, pi0, ganancia), sin mirar la respuesta; luego L3

    PYTHONPATH=codigo python codigo/e69_dos_etapas.py fase1
    PYTHONPATH=codigo python codigo/e69_dos_etapas.py fase2
    PYTHONPATH=codigo python codigo/e69_dos_etapas.py analisis
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
from scipy.special import expit, logit

import flocking
from flocking import Flock, Params, run_response, phi_grid
from parametros import P0
from fase4 import D_PHI, SEMILLA_R, WINS, W, _w, ARM, T_N, R_rep, rec_ic, limpio, crudos_previos, clave as clave4

N, N_REP = 250, 8
DT_REG = 0.1
T_REG = 15.8
CELDAS = {"F4": (False, 4.0, "ViSt"), "M4": (True, 4.0, "CoLd"), "M1": (True, 1.0, "CoLd")}
COMPUERTAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
ATENUADOS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)
TAU_OBS = P0.tau_obs
F1, F2 = "resultados/E69_fase1_crudo.json", "resultados/E69_fase2_crudo.json"
F_JSON, F_TXT = "resultados/E69_dos_etapas.json", "resultados/E69_dos_etapas.txt"


def n_ef(tau):
    return max(tau / TAU_OBS, 1.0)


def observadores():
    """(nombre, base, kw, factor, celdas). Todas con k = 15, N = 250, ruido de referencia."""
    out = []
    for v in (0.25, 0.5, 1.0, 2.0, 4.0):
        out.append((f"ciclo{v:g}", "adaptado", {"tau_det": v}, 1.0, ("F4", "M4") + (("M1",) if v == 1.0 else ())))
    for p in (0.004, 0.01, 0.05, 0.3, 0.5):
        out.append((f"pi{p:g}", "adaptado", {"pi0": p}, 1.0, ("F4", "M4", "M1")))
    for n in (3.0, 6.0, 24.0, 48.0):
        out.append((f"ciclo1_n{n:g}", "adaptado", {"tau_det": 1.0}, n / n_ef(1.0), ("F4", "M4")))
    for v in (0.25, 0.5, 2.0, 4.0):
        out.append((f"ciclo{v:g}_n12", "adaptado", {"tau_det": v}, n_ef(1.0) / n_ef(v), ("F4", "M4")))
    for m in (1.0, 6.0, 12.0):
        out.append((f"inst_x{m:g}", "amplitud", {}, m, ("F4", "M4") + (("M1",) if m == 1.0 else ())))
    out.append(("secuencial", "adaptado", {"detector": "secuencial"}, 1.0, ("F4", "M4")))
    return out


class FlockReg(Flock):
    """Registra el posterior medio sobre el líder y su indicador de excedencia; o interviene
    la lectura del líder (compuerta, atenuado, réplica). La columna propia no se toca."""
    factor = 1.0
    modo = None          # None | 'compuerta' | 'atenuado' | 'replica'
    T_g = np.inf
    c = 1.0
    perfil = None        # lectura del líder por tramo de DT_REG desde t_on (réplica)
    ultimo = None

    def __init__(self, p, model):
        super().__init__(p, model)
        self._t_on = None
        self.wL, self.ex = [], []
        self._acc_w, self._acc_e, self._n = 0.0, 0.0, 0
        self._k = max(int(round(DT_REG / p.dt)), 1)
        FlockReg.ultimo = self

    def _llr_kinematic(self, idx):
        return self.factor * super()._llr_kinematic(idx)

    def _perceived_will(self, idx, dist):
        w = super()._perceived_will(idx, dist)
        if self._t_on is None and self.will.max() > 0.5:
            self._t_on = self.t
        if self._t_on is None:
            return w
        own = idx == np.arange(self.p.N)[:, None]
        es_l = (idx == self.leader) & ~own
        if self.modo is None:
            v = w[es_l]
            eps = self.omega[self.leader] - self.omega[idx[self.leader]].mean()
            self._acc_w += float(v.mean()) if v.size else 0.0
            self._acc_e += float(abs(eps) > self.p.c_det * self.p.sigma_theta)
            self._n += 1
            if self._n == self._k:
                self.wL.append(self._acc_w / self._n)
                self.ex.append(self._acc_e / self._n)
                self._acc_w, self._acc_e, self._n = 0.0, 0.0, 0
            return w
        t = self.t - self._t_on
        if self.modo == "compuerta":
            return np.where(es_l & (t >= self.T_g - 1e-9), 0.0, w)
        if self.modo == "atenuado":
            return np.where(es_l, self.c * w, w)
        if self.modo == "replica":
            i = min(int(t / DT_REG), len(self.perfil) - 1)
            return np.where(es_l, self.perfil[i], w)
        raise ValueError(self.modo)


def job(a):
    p = Params(**{**asdict(P0), **a["kw"], "N": N, "lambda_fp": 0.0, "tau_W": a["tau_W"],
                  "seed": SEMILLA_R + 1000 * a["i_phi"] + a["rep"]})
    attrs = {"factor": float(a.get("factor", 1.0)), "modo": a.get("modo"), "T_g": float(a.get("T_g", np.inf)),
             "c": float(a.get("c", 1.0)), "perfil": a.get("perfil")}
    flocking.Flock = type("FR", (FlockReg,), attrs)
    t0 = time.time()
    try:
        r = run_response(p, a["model"], a["phi"], T_N[N], every_coh=200, pulse=a["pulso"])
    finally:
        flocking.Flock = Flock
    proj = r["proj"]
    out = {"celda": a["celda"], "nom": a["nom"], "N": N, "tau_W": a["tau_W"], "pulso": a["pulso"],
           "i_phi": a["i_phi"], "rep": a["rep"],
           "r": {_w(w): float(proj[:int(round(w / p.dt))].mean()) for w in WINS if w <= T_N[N] + 1e-9},
           "seg": time.time() - t0}
    if a.get("modo") is None and a["model"] == "WiSt" and a["kw"].get("will_mode") == "bayes":
        f = FlockReg.ultimo
        out["wL"] = [round(x, 5) for x in f.wL]
        out["ex"] = [round(x, 4) for x in f.ex]
    return out


def clave(x):
    return f'{x["celda"]}|{x["nom"]}|{x["tau_W"]:g}|{int(x["pulso"])}|{x["i_phi"]}|{x["rep"]}'


def base_job(celda, nom, base, kw, tw, pulso, i, phi, rep, **extra):
    model, kw0 = ARM[base]
    return {"celda": celda, "nom": nom, "model": model, "kw": {**kw0, **kw}, "tau_W": tw, "pulso": pulso,
            "i_phi": i, "phi": float(phi), "rep": rep, **extra}


def trabajos_fase1():
    phis = phi_grid(D_PHI)
    out = []
    for nom, base, kw, fac, cs in observadores():
        for c in cs:
            pulso, tw, _ = CELDAS[c]
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    out.append(base_job(c, nom, base, kw, tw, pulso, i, phi, rep, factor=fac))
    for c, (pulso, tw, _) in CELDAS.items():
        for T in COMPUERTAS:
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    out.append(base_job(c, f"G{T:g}", "oraculo", {}, tw, pulso, i, phi, rep, modo="compuerta", T_g=T))
        for cc in ATENUADOS:
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    out.append(base_job(c, f"A{cc:g}", "oraculo", {}, tw, pulso, i, phi, rep, modo="atenuado", c=cc))
    return out


def perfiles(res1):
    """Perfil medio del posterior sobre el líder por celda de observador (tramos de 0,1 s)."""
    acc = {}
    for x in res1:
        if "wL" in x:
            acc.setdefault((x["celda"], x["nom"]), []).append(x)
    out = {}
    for (c, nom), xs in acc.items():
        L = min(len(x["wL"]) for x in xs)
        out[f"{c}|{nom}"] = {"wL": np.mean([x["wL"][:L] for x in xs], axis=0).tolist(),
                             "ex": np.mean([x["ex"][:L] for x in xs], axis=0).tolist(), "n": len(xs)}
    return out


def trabajos_fase2(perf):
    phis = phi_grid(D_PHI)
    out = []
    for nom, base, kw, fac, cs in observadores():
        for c in cs:
            pulso, tw, _ = CELDAS[c]
            pf = perf[f"{c}|{nom}"]["wL"]
            for i, phi in enumerate(phis):
                for rep in range(N_REP):
                    out.append(base_job(c, f"R_{nom}", "oraculo", {}, tw, pulso, i, phi, rep, modo="replica", perfil=pf))
    return out


def correr(jobs, f_out, etq):
    f_par = f_out + ".parcial"
    hechos = {}
    if os.path.exists(f_par):
        for x in json.load(open(f_par)):
            hechos[clave(x)] = x
    faltan = [a for a in jobs if clave(a) not in hechos]
    print(f"[{etq}] {len(jobs)} trabajos, faltan {len(faltan)}; {W} procesos", flush=True)
    t0, ult = time.time(), time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(job, a) for a in faltan]
        for k, fu in enumerate(as_completed(futs), 1):
            x = fu.result()
            hechos[clave(x)] = x
            if time.time() - ult > 120 or k == len(futs):
                json.dump(list(hechos.values()), open(f_par, "w"))
                ult = time.time()
                print(f"[{etq}] {k}/{len(futs)}  {time.time() - t0:.0f} s", flush=True)
    res = [hechos[clave(a)] for a in jobs]
    json.dump({"resp": res, "seg": time.time() - t0}, open(f_out, "w"))
    if os.path.exists(f_par):
        os.remove(f_par)
    return res


# ------------------------------------------------------------------ análisis

def referencias():
    prev = crudos_previos()
    out = []
    for x in prev:
        if x["N"] != N:
            continue
        for c, (pulso, tw, ref) in CELDAS.items():
            if x["pulso"] == pulso and abs(x["tau_W"] - tw) < 1e-9 and x["nom"] in (ref, "oraculo"):
                out.append({**x, "celda": c})
    return out


def rec_de(todo, c, nom):
    pulso, tw, ref = CELDAS[c]
    reps = list(range(N_REP))
    im, Rm = R_rep(todo, c, nom, N, pulso, T_N[N], tw, reps)
    iv, Rv = R_rep(todo, c, ref, N, pulso, T_N[N], tw, reps)
    io, Ro = R_rep(todo, c, "oraculo", N, pulso, T_N[N], tw, reps)
    com = sorted(set(im) & set(iv) & set(io))
    if not com:
        return None
    g = lambda ids, R: np.array([R[ids.index(k)] for k in com])
    return limpio(rec_ic(g(im, Rm), g(iv, Rv), g(io, Ro)))


def curva_g(todo, c):
    """g(T) en la rejilla de compuertas, con g(0) = 0 y g(inf) = 1 (el oráculo entero)."""
    T = [0.0] + list(COMPUERTAS) + [T_N[N]]
    g = [0.0] + [rec_de(todo, c, f"G{t:g}")["rec"] for t in COMPUERTAS] + [1.0]
    return np.array(T), np.array(g)


def curva_H(todo, c):
    cc = [0.0] + list(ATENUADOS) + [1.0]
    H = [0.0] + [rec_de(todo, c, f"A{x:g}")["rec"] for x in ATENUADOS] + [1.0]
    return np.array(cc), np.array(H)


def pred_L1(perf_w, pi0, T, g):
    w = np.asarray(perf_w)
    T_ef = float(np.sum(np.clip((w - pi0) / (1 - pi0), 0, None)) * DT_REG)
    return float(np.interp(T_ef, T, g)), T_ef


def pred_L3(perf_w, T, g, cc, H):
    """Respuesta lineal en el tiempo: sum_t g'(t) H(w(t)) dt, con g' por tramos de la rejilla."""
    w = np.asarray(perf_w)
    t = (np.arange(len(w)) + 0.5) * DT_REG
    dg = np.diff(g) / np.diff(T)                       # densidad por tramo
    gp = dg[np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(dg) - 1)]
    Hw = np.interp(w, cc, H)
    return float(np.sum(gp * Hw) * DT_REG)


def perfil_cerrado(ex, tau_det, n, pi0, factor, detector):
    """Posterior medio sobre el líder desde su fracción media de excedencias p(t) (sin fluctuaciones)."""
    from scipy.stats import t as tdist
    from flocking import _t_scale_from_mad
    p = P0
    c = _t_scale_from_mad(p.nu)
    s0, s1 = p.sigma_theta, float(np.hypot(p.sigma_theta, p.sigma_W))
    thr = p.c_det * p.sigma_theta
    q0 = float(np.clip(2 * tdist.sf(thr / (s0 * c), p.nu), 1e-6, 1 - 1e-6))
    q1 = float(np.clip(2 * tdist.sf(thr / (s1 * c), p.nu), 1e-6, 1 - 1e-6))
    ex = np.asarray(ex)
    a = np.exp(-DT_REG / max(tau_det, 1e-6))
    d, out = q0, []
    for e in ex:
        d = a * d + (1 - a) * e
        llr = factor * n * (d * np.log(q1 / q0) + (1 - d) * np.log((1 - q1) / (1 - q0)))
        out.append(float(expit(llr + logit(pi0))))
    return np.array(out)


def analizar(res1, res2):
    todo = referencias() + res1 + res2
    perf = perfiles(res1)
    out = {"g": {}, "H": {}, "celdas": {}}
    L = ["=== E69 · dos etapas: cuánto vale leer al líder y cuánto lee el observador (N = 250, 8 réplicas) ==="]
    GH = {}
    for c in CELDAS:
        T, g = curva_g(todo, c)
        cc, H = curva_H(todo, c)
        GH[c] = (T, g, cc, H)
        out["g"][c] = {"T": T.tolist(), "g": g.tolist()}
        out["H"][c] = {"c": cc.tolist(), "H": H.tolist()}
        L.append(f"{c}: g(T) " + " ".join(f"{t:g}:{v:.3f}" for t, v in zip(T[1:-1], g[1:-1]))
                 + " | H(c) " + " ".join(f"{x:g}:{v:.3f}" for x, v in zip(cc[1:-1], H[1:-1])))
    L.append(f"\n{'celda':22s} {'medido':>7s} {'L1':>6s} {'T_ef':>5s} {'L2 rép.':>8s} {'L3':>6s} {'L4':>6s}")
    for nom, base, kw, fac, cs in observadores():
        for c in cs:
            q = rec_de(todo, c, nom)
            qr = rec_de(todo, c, f"R_{nom}")
            pf = perf[f"{c}|{nom}"]
            T, g, cc, H = GH[c]
            pi0 = kw.get("pi0", P0.pi0)
            l1, T_ef = pred_L1(pf["wL"], pi0, T, g)
            l3 = pred_L3(pf["wL"], T, g, cc, H)
            fila = {"medido": q, "replica": qr, "L1": l1, "T_ef": T_ef, "L3": l3}
            if base == "adaptado" and kw.get("detector") != "secuencial":
                tau = kw.get("tau_det", 1.0)
                prof = perfil_cerrado(pf["ex"], tau, n_ef(tau), pi0, fac, "ciclo")
                fila["L4"] = pred_L3(prof, T, g, cc, H)
                fila["L4_perfil_error"] = float(np.mean(np.abs(prof - np.asarray(pf["wL"]))))
            out["celdas"][f"{c}|{nom}"] = fila
            L.append(f"{c + '|' + nom:22s} {q['rec']:7.3f} {l1:6.3f} {T_ef:5.2f} "
                     f"{(qr['rec'] if qr else float('nan')):8.3f} {l3:6.3f} {fila.get('L4', float('nan')):6.3f}")
    C = out["celdas"]
    med = np.array([v["medido"]["rec"] for v in C.values()])

    def crit(clave_, tol, frac):
        pr = np.array([v.get(clave_, np.nan) if not isinstance(v.get(clave_), dict) else v[clave_]["rec"]
                       for v in C.values()], dtype=float)
        ok = np.isfinite(pr)
        e = np.abs(pr[ok] - med[ok])
        from scipy.stats import spearmanr
        rho = float(spearmanr(pr[ok], med[ok]).correlation) if ok.sum() > 2 else float("nan")
        return {"n": int(ok.sum()), "frac_dentro": float((e <= tol).mean()), "error_mediano": float(np.median(e)),
                "spearman": rho, "cumple": bool((e <= tol).mean() >= frac)}

    out["pred"] = {"L1": crit("L1", 0.10, 0.8), "L2": crit("replica", 0.10, 0.8), "L3": crit("L3", 0.10, 0.8),
                   "L4": {**crit("L4", 0.15, 0.7)}}
    out["pred"]["L4"]["cumple"] = bool(out["pred"]["L4"]["cumple"] and out["pred"]["L4"]["spearman"] >= 0.8)
    L.append("\n=== Predicciones ===")
    L += [f"{k}: {'CUMPLE' if v['cumple'] else 'FALLA'}  {json.dumps(v)}" for k, v in out["pred"].items()]
    return out, L


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    etapa = sys.argv[1] if len(sys.argv) > 1 else "fase1"
    if etapa == "fase1":
        correr(trabajos_fase1(), F1, "E69-1")
    elif etapa == "fase2":
        res1 = json.load(open(F1))["resp"]
        correr(trabajos_fase2(perfiles(res1)), F2, "E69-2")
    else:
        res1 = json.load(open(F1))["resp"]
        res2 = json.load(open(F2))["resp"]
        # ancla: las celdas de observador re-corridas reproducen lo guardado (las que existen)
        prev = {clave4({**x, "celda": "base"}): x for x in crudos_previos()}
        dm, nc = 0.0, 0
        mapa = {"ciclo1": "adaptado", "ciclo0.5": "ciclo 0.5 s", "ciclo2": "ciclo 2 s", "inst_x1": "amplitud"}
        for x in res1:
            if x["nom"] in mapa:
                y = prev.get(clave4({**x, "celda": "base", "nom": mapa[x["nom"]]}))
                if y:
                    for w in x["r"]:
                        if w in y["r"]:
                            dm = max(dm, abs(x["r"][w] - y["r"][w])); nc += 1
        out, L = analizar(res1, res2)
        out["ancla"] = {"delta_max": dm, "comparaciones": nc, "cumple": bool(nc > 0 and dm < 1e-12)}
        L.insert(1, f"ancla (observadores re-corridos frente a lo guardado): |Δ| máx = {dm:.2e} en {nc} comparaciones")
        json.dump(out, open(F_JSON, "w"), indent=1, default=float)
        open(F_TXT, "w").write("\n".join(L) + "\n")
        print("\n".join(L))
