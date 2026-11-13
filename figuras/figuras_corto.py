"""Figuras propias del Paper A corto.

Las figuras que no cambian (esquema, derivación) se copian de `paper_a/fig/`. Aquí se
generan, siempre desde los JSON de `resultados/`:

  f_especies.pdf   curva AUC frente a la ventana (tetras, E54; grajillas, E52) y ganancia
                   de integrar con sus intervalos
  f_escalera.pdf   barrido de la ventana, recuperación frente a N y frente a tau_W;
                   todo desde E46 (maniobra única contada desde CoLd)
  f2_footprint.pdf la huella, con la etiqueta del estado corregida (reejecuta la
                   medición con las mismas semillas que `paper_a/figuras.py`)

    PYTHONPATH=codigo python figuras/figuras_corto.py [especies|escalera|huella]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIG = os.path.join(AQUI, "fig")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(RAIZ, "codigo"))

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.handlelength": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
AZUL, NARANJA, VERDE, GRIS, ROJO = "#1f5fa8", "#d1701a", "#2c8c4a", "#8a8a8a", "#b02020"


def cargar(nombre):
    with open(os.path.join(RAIZ, "resultados", nombre)) as f:
        return json.load(f)


def _curva(c):
    ks = sorted(c, key=float)
    return [float(k) for k in ks], [c[k] for k in ks]


def fig_especies():
    """Discriminación iniciador/seguidor frente a la ventana: tetras con ventanas que terminan
    en el giro colectivo (E54) y antes de la primera inversión (E54b), y grajillas (E52, con
    intervalos por grabación de E52b)."""
    e52b, e54, e54b = cargar("E52b_aves_por_grabacion.json"), cargar("E54_peces.json"), cargar("E54b_prospectivo.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.35), width_ratios=[1.45, 1])
    X0 = 0.03

    def serie(curva, ic):
        ks = sorted([k for k in curva if k != "cuadro"], key=float)
        x = np.array([X0] + [float(k) for k in ks])
        y = np.array([curva["cuadro"]] + [curva[k] for k in ks])
        lo = np.array([ic["cuadro"][0]] + [ic[k][0] for k in ks])
        hi = np.array([ic["cuadro"][1]] + [ic[k][1] for k in ks])
        return x, y, lo, hi

    t = e54["N8_10"]
    x, y, lo, hi = serie(t["curva"], t["curva_ic95"])
    a1.fill_between(x, lo, hi, color=AZUL, alpha=0.15, lw=0)
    a1.plot(x, y, "-o", color=AZUL, lw=1.2, ms=2.4, label="tetra, ends at collective reversal")
    for fin, ls, et in (("d0", (0, (4, 2)), "tetra, ends at first reversal"),
                        ("d0.5", (0, (1, 1.5)), "tetra, ends 0.5 s before it")):
        q = e54b["con_primer_pez"][fin]
        ks = ["cuadro", "0.25", "0.5", "1.0", "2.0"]
        xx = [X0, 0.25, 0.5, 1.0, 2.0]
        a1.plot(xx, [q[k]["auc"] for k in ks], ls=ls, marker="o", ms=2.0, color=AZUL, lw=1.0, label=et)
    r = e52b["curva"]
    ks = sorted([k for k in r if k != "cuadro"], key=float)
    xx = np.array([X0] + [float(k) for k in ks])
    yy = np.array([r["cuadro"]["auc"]] + [r[k]["auc"] for k in ks])
    lo = np.array([r["cuadro"]["ic95"][0]] + [r[k]["ic95"][0] for k in ks])
    hi = np.array([r["cuadro"]["ic95"][1]] + [r[k]["ic95"][1] for k in ks])
    a1.fill_between(xx, lo, hi, color=NARANJA, alpha=0.15, lw=0)
    a1.plot(xx, yy, "-s", color=NARANJA, lw=1.1, ms=2.4, label="jackdaw, ends at collective onset")
    a1.axhline(0.5, color="0.6", lw=0.6, ls=(0, (4, 3)))
    a1.text(0.06, 0.505, "chance", ha="left", fontsize=6.0, color="0.45", va="bottom")
    a1.set_xlabel("window length (s)")
    a1.set_ylabel("initiator/follower AUC")
    a1.set_xlim(-0.03, 2.05); a1.set_ylim(0.36, 0.88)
    a1.legend(loc="lower right", fontsize=5.6, borderaxespad=0.2, handlelength=2.4)

    filas = [("tetra, collective reversal", t["delta_1s_cuadro"], t["delta_ic95"], AZUL, "o")]
    for fin, et in (("d0", "tetra, first reversal"), ("d0.25", "tetra, 0.25 s before"), ("d0.5", "tetra, 0.5 s before")):
        g = e54b["con_primer_pez"][fin]["ganancia_1s"]
        filas.append((et, g["valor"], g["ic95"], AZUL, "o"))
    filas.append(("jackdaw (8 events, 6 recordings)", e52b["ganancia"], e52b["ganancia_ic95"], NARANJA, "s"))
    ys = []
    for i, (et, d, (lo, hi), c, mk) in enumerate(filas):
        y = i + (0.4 if c == NARANJA else 0.0)
        a2.errorbar([d], [y], xerr=[[d - lo], [hi - d]], fmt=mk, color=c, ms=3.0, capsize=2, lw=0.9)
        ys.append(y)
    a2.axvline(0.0, color="0.6", lw=0.6, ls=(0, (4, 3)))
    a2.set_yticks(ys); a2.set_yticklabels([f[0] for f in filas], fontsize=6.0)
    a2.invert_yaxis()
    a2.set_xlabel("gain in AUC, 1 s window\nagainst single frame", fontsize=7)
    a2.set_xlim(-0.16, 0.32)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, "f_especies.pdf"))
    plt.close(fig)
    return {"ganancia_tetra": t["delta_1s_cuadro"]}


def _pt(q):
    """(valor, error abajo, error arriba) en porcentaje."""
    return 100 * q["rec"], 100 * (q["rec"] - q["ic95"][0]), 100 * (q["ic95"][1] - q["rec"])


def fig_escalera():
    """Barrido de la ventana; recuperación frente a N; recuperación frente a tau_W.

    Todas las cifras salen de E46: la maniobra única se cuenta desde CoLd y el líder
    forzado desde ViSt, que ahí es la misma corrida.
    """
    e46 = cargar("E46_referencia_cold.json")
    esc, bar = e46["escalera"], e46["barrido"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.1, 2.3), width_ratios=[1.3, 1, 0.85])
    fig.subplots_adjust(wspace=0.30)
    PROT = (("tractor", AZUL, "o", "driven leader"), ("pulso", NARANJA, "s", "single manoeuvre"))
    YLIM = (-10, 104)
    for a in (a1, a2, a3):
        a.axhline(0, color="0.8", lw=0.5, zorder=0)

    # --- (a) barrido de la ventana del detector
    a1.axvspan(0.88, 1.38, color="0.88", lw=0, zorder=0)
    a1.text(1.10, 97, "footprint", ha="center", fontsize=6.0, color="0.35")
    for N, ls, alfa, ms in ((250, "-", 1.0, 3.6), (1000, (0, (3, 2)), 0.6, 2.8)):
        for etq, col, mk, lab in PROT:
            d = bar[f"{etq}|{N}"]
            ts = [t for t in ("0.25", "0.5", "1", "2", "4") if t in d]
            x = [0.1] + [float(t) for t in ts]
            pts = [_pt(d["amplitud"])] + [_pt(d[t]) for t in ts]
            a1.errorbar(x, [p[0] for p in pts], yerr=[[p[1] for p in pts], [p[2] for p in pts]],
                        color=col, marker=mk, ms=ms, ls=ls, lw=1.1, alpha=alfa, capsize=1.6,
                        elinewidth=0.6, zorder=3, label=lab if N == 250 else None)
    a1.set_xscale("log")
    a1.set_xticks([0.1, 0.25, 0.5, 1, 2, 4])
    a1.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
    a1.minorticks_off()
    a1.set_xlabel("observer's integration window (s)")
    a1.set_ylabel("advantage recovered (%)")
    a1.set_ylim(*YLIM)
    a1.legend(loc="upper left", fontsize=5.6, borderaxespad=0.2, labelspacing=0.25)
    a1.text(0.092, 72, "solid: $N=250$\ndashed: $N=1000$", fontsize=5.4, color="0.3",
            va="top")

    # --- (b) frente a N, ventana de 1 s
    Ns = [250, 500, 1000]
    for etq, col, mk, lab in PROT:
        full = [_pt(esc[f"{etq}|{N}|4"]["adaptado"]["entero"]) for N in Ns]
        prim = [_pt(esc[f"{etq}|{N}|4"]["adaptado"]["primeros4"]) for N in Ns]
        sinm = [_pt(esc[f"{etq}|{N}|4"]["amplitud"]["entero"]) for N in Ns]
        a2.errorbar(Ns, [p[0] for p in full], yerr=[[p[1] for p in full], [p[2] for p in full]],
                    color=col, marker=mk, ms=3.6, ls="-", lw=1.1, capsize=1.8, elinewidth=0.6,
                    label=lab)
        a2.errorbar(Ns, [p[0] for p in prim], yerr=[[p[1] for p in prim], [p[2] for p in prim]],
                    color=col, marker=mk, ms=3.6, mfc="white", ls=(0, (1, 1.5)), lw=1.0,
                    capsize=1.8, elinewidth=0.6)
        a2.plot(Ns, [p[0] for p in sinm], color=col, marker="x", ms=3.4, ls=(0, (3, 2)), lw=0.9)
    a2.text(255, 91, "first 4 s", fontsize=5.8, color="0.3")
    a2.text(1000, 56, "whole\nwindow", fontsize=5.8, color="0.3", ha="right")
    a2.text(1000, 12, "memoryless", fontsize=5.8, color="0.3", ha="right")
    a2.set_xscale("log")
    a2.set_xticks(Ns); a2.set_xticklabels([str(n) for n in Ns]); a2.minorticks_off()
    a2.set_xlabel("group size $N$")
    a2.set_ylim(*YLIM)
    a2.set_yticklabels([])

    # --- (c) frente a tau_W, maniobra única
    taus = [1.0, 2.0, 4.0]
    ad = [_pt(esc[f"pulso|250|{t:g}"]["adaptado"]["entero"]) for t in taus]
    am = [_pt(esc[f"pulso|250|{t:g}"]["amplitud"]["entero"]) for t in taus]
    a3.axhline(100, color="0.6", lw=0.6, ls=(0, (4, 3)))
    a3.errorbar(taus, [p[0] for p in ad], yerr=[[p[1] for p in ad], [p[2] for p in ad]],
                color=NARANJA, marker="s", ms=3.6, ls="-", lw=1.1, capsize=1.8, elinewidth=0.6,
                label="$N=250$")
    a3.plot(taus, [p[0] for p in am], color=NARANJA, marker="x", ms=3.4, ls=(0, (3, 2)), lw=0.9)
    g1 = _pt(esc["pulso|1000|1"]["adaptado"]["entero"])
    g4 = _pt(esc["pulso|1000|4"]["adaptado"]["entero"])
    a3.errorbar([1.0, 4.0], [g1[0], g4[0]], yerr=[[g1[1], g4[1]], [g1[2], g4[2]]], color=NARANJA,
                marker="s", ms=2.8, ls=(0, (3, 2)), lw=1.0, alpha=0.6, capsize=1.6,
                elinewidth=0.6, label="$N=1000$")
    a3.set_xticks(taus); a3.set_xticklabels([f"{t:g}" for t in taus])
    a3.set_xlim(0.6, 4.4)
    a3.set_xlabel("state decay time $\\tau_W$ (s)")
    a3.set_ylim(*YLIM)
    a3.set_yticklabels([])
    a3.legend(loc=(0.04, 0.30), fontsize=5.6, borderaxespad=0.2)
    a3.text(4.3, 9, "memoryless", fontsize=5.8, color="0.3", ha="right")
    fig.savefig(os.path.join(FIG, "f_escalera.pdf"))
    plt.close(fig)


def fig_huella():
    from flocking import Params
    from e15_helpers import huella
    CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
    BASE = dict(N=200, k=12, sigma_xi=np.sqrt(2.0), seed=11)
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    datos = {}
    for m, c in (("WiSt", AZUL), ("CoLd", NARANJA), ("CaSp", GRIS)):
        curvas, durs = [], []
        for rep in range(16):
            r = huella(Params(**{**BASE, **CFG, "seed": BASE["seed"] + rep}), m, n_rep=1)
            # en unidades de la anchura de línea base de esa misma corrida: la de CaSp
            # (0,84 rad/s) es 2,3 veces la de WiSt y CoLd (0,36 rad/s)
            curvas.append(r["series"][:, 1] / r["sigma_base"]); durs.append(r["dur"])
        t = r["series"][:, 0]
        med = np.median(np.vstack(curvas), axis=0)
        datos[m] = float(np.median(durs))
        y = np.convolve(med, np.ones(5) / 5, mode="same")
        ax.plot(t, y, color=c, lw=1.1,
                label=f"{m} ({datos[m]:.2f} s)" if m != "CaSp" else f"{m}, no will channel",
                ls="-" if m != "CaSp" else (0, (3, 2)), zorder=3 if m != "CaSp" else 2)
    ax.axhspan(0, 1.5, color="#f0f0f0", zorder=0)
    ax.axhline(1.5, color="k", lw=0.7, ls=(0, (4, 3)))
    ax.text(4.95, 1.62, r"threshold $1.5\,\sigma_\theta$", ha="right", va="bottom",
            fontsize=6.3, color="0.3")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_bounds(0, 5.6)
    ax.set_xlabel("time since the intention was set (s)")
    ax.set_ylabel(r"$|\epsilon_j|\,/\,\sigma_\theta$")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.8)
    fig.savefig(os.path.join(FIG, "f2_footprint.pdf"))
    plt.close(fig)
    return datos


if __name__ == "__main__":
    que = sys.argv[1:] or ["especies", "escalera", "huella"]
    if "especies" in que:
        print("especies", fig_especies())
    if "escalera" in que:
        fig_escalera(); print("escalera ok")
    if "huella" in que:
        print("huella", fig_huella())
