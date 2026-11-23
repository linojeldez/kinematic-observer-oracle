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

# Latin Modern, la letra del texto del paper (la de la guía, figuras_guia.py)
def _letra():
    """Computer Modern (Latin Modern), la misma letra del texto de la guía.

    Las fuentes vienen con TeX Live; si no se encuentran, se usa STIX.
    """
    import glob
    import shutil
    import subprocess
    from matplotlib import font_manager as fm

    carpetas = ["/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm",
                "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm",
                "/usr/share/texmf/fonts/opentype/public/lm",
                "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"]
    kp = shutil.which("kpsewhich")
    if kp:
        try:
            r = subprocess.run([kp, "lmroman10-regular.otf"], capture_output=True, text=True)
            if r.stdout.strip():
                carpetas.insert(0, os.path.dirname(r.stdout.strip()))
        except OSError:
            pass
    for patron in carpetas:
        for carpeta in sorted(glob.glob(patron), reverse=True):
            archivos = [os.path.join(carpeta, f"lmroman10-{v}.otf")
                        for v in ("regular", "italic", "bold", "bolditalic")]
            if all(os.path.exists(a) for a in archivos):
                for a in archivos:
                    fm.fontManager.addfont(a)
                return fm.FontProperties(fname=archivos[0]).get_name(), "cm"
    return "STIXGeneral", "stix"


LETRA, LETRA_MAT = _letra()

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8,
    "font.family": LETRA, "mathtext.fontset": LETRA_MAT,
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
    # plan 15: el barrido de la ventana sale de E66 (16 réplicas); el punto sin memoria, de E46
    c66 = cargar("E66_replicas.json")["celdas"]
    bar = {k: dict(v) for k, v in bar.items()}
    for N in (250, 1000):
        for etq, et66 in (("tractor", "forzado"), ("pulso", "maniobra")):
            for t, q in c66[f"barrido|base|N{N}|{et66}"]["n16"]["rec"].items():
                bar[f"{etq}|{N}"][t] = q
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


def fig_frontera():
    """(a) recuperación frente a la fracción de informados (E68); (b) la cabecera con otro ruido y otra
    agilidad (E70, E70b); (c) persistencia frente a recuperación, sin y con falsas alarmas (E64, E72)."""
    e68, e64 = cargar("E68_fraccion_informados.json"), cargar("E64_persistencia.json")
    e70, e72 = cargar("E70_cabecera_robustez.json"), cargar("E72_persistencia_fa.json")
    f70b = os.path.join(RAIZ, "resultados", "E70b_ventana_agilidad.json")
    e70b = json.load(open(f70b)) if os.path.exists(f70b) else None
    fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(7.1, 2.35), width_ratios=[1.1, 1, 1])
    fig.subplots_adjust(wspace=0.36)
    CEL = (("forzado|4", AZUL, "o", r"driven, $\tau_W=4$ s"), ("maniobra|4", NARANJA, "s", r"manoeuvre, $\tau_W=4$ s"),
           ("maniobra|1", VERDE, "^", r"manoeuvre, $\tau_W=1$ s"))
    ns = (1, 3, 12, 25)
    xs = [100 * n / 250 for n in ns]
    R = e68["rec"]
    for c, col, mk, lab in CEL:
        pts = [_pt(R[f"N250|n{n}|{c}"]["integrador"]) for n in ns]
        a1.errorbar(xs, [p[0] for p in pts], yerr=[[p[1] for p in pts], [p[2] for p in pts]], color=col, marker=mk,
                    ms=3.4, lw=1.1, capsize=1.6, elinewidth=0.6, label=lab)
        inst = [100 * max(v["rec"] for k, v in R[f"N250|n{n}|{c}"].items() if k.startswith("sin_memoria")) for n in ns]
        a1.plot(xs, inst, color=col, marker="x", ms=3.0, ls=(0, (3, 2)), lw=0.9)
        q = _pt(R[f"N1000|n48|{c}"]["integrador"])
        a1.errorbar([4.8 * 1.07], [q[0]], yerr=[[q[1]], [q[2]]], color=col, marker=mk, mfc="white", ms=3.4,
                    capsize=1.6, elinewidth=0.6, ls="none")
    a1.axhline(0, color="0.8", lw=0.5, zorder=0)
    a1.axhline(25, color="0.6", lw=0.6, ls=(0, (1, 2)), zorder=0)
    a1.set_xscale("log")
    a1.set_xticks(xs); a1.set_xticklabels([f"{x:g}" for x in xs]); a1.minorticks_off()
    a1.set_xlabel("informed agents (% of group)")
    a1.set_ylabel("advantage recovered (%)")
    a1.set_ylim(-10, 108)
    a1.legend(loc="upper left", fontsize=5.2, borderaxespad=0.2, labelspacing=0.2)
    a1.text(10.5, 50, "best\nmemoryless", fontsize=5.2, color="0.3", ha="right", va="center")
    a1.text(10.5, 101, "open: $N=1000$", fontsize=5.2, color="0.3", ha="right")

    # (b) robustez de la cabecera
    R7, om = e70["rec"], e70["omega_sobre_sigma"]
    for c, col, mk, lab in CEL:
        for vs, ls in ((("f2", "ref", "f0.5"), "-"), (("a5", "ref", "a20", "a40"), (0, (3, 2)))):
            a3.plot([om[v] for v in vs], [100 * R7[f"{v}|{c}"]["integrador"]["rec"] for v in vs], color=col,
                    marker=mk, ms=3.0, lw=0.9, ls=ls, mfc=("white" if ls != "-" else col))
        a3.plot([om["k70"]], [100 * R7[f"k70|{c}"]["integrador"]["rec"]], color=col, marker="D", ms=2.8, ls="none")
        if e70b:
            for v, w in (("a20", "integrador_0.5s"), ("a40", "integrador_0.25s")):
                q = e70b["rec"].get(f"{v}|{c}")
                if q:
                    a3.plot([om[v] * 1.06], [100 * q["q"][w]["rec"]], color=col, marker="*", ms=4.5, ls="none")
    a3.axhline(25, color="0.6", lw=0.6, ls=(0, (1, 2)), zorder=0)
    a3.set_xscale("log")
    a3.set_xticks([1.72, 3.46, 6.93, 13.87]); a3.set_xticklabels(["1.7", "3.5", "6.9", "13.9"]); a3.minorticks_off()
    a3.set_xlabel(r"$\omega_{\max}/\sigma_\theta$")
    a3.set_ylim(-5, 105)
    a3.text(1.75, 2, "solid: noise\ndashed, open: agility\ndiamond: $k=70$" + ("\nstar: window 0.25-0.5 s" if e70b else ""),
            fontsize=4.8, color="0.3", va="bottom")

    # (c) persistencia sin y con falsas alarmas
    P, Rf, P72 = e64["P"], e64["R"]["forzado|4"], e72["P"]
    PIS = ("0.004", "0.01", "0.05", "0.173", "0.3")
    x = [100 * Rf[f"adaptado_pi{p}"]["rec"] for p in PIS]
    a2.plot(x, [P[f"adaptado_pi{p}"]["ratio_incr"] for p in PIS], color=AZUL, marker="o", ms=3.0, lw=1.0,
            label="integrating (1 s)")
    x = [100 * Rf[f"amplitud_pi{p}"]["rec"] for p in PIS]
    a2.plot(x, [P[f"amplitud_pi{p}"]["ratio_incr"] for p in PIS], color=GRIS, marker="x", ms=3.0, lw=1.0, label="memoryless")
    for nom, mk in (("inst_m6_pi0.173", "+"), ("inst_m12_pi0.173", "1")):
        a2.plot([100 * Rf[nom]["rec"]], [P[nom]["ratio_incr"]], color=ROJO, marker=mk, ms=4.5, ls="none")
    a2.plot([100], [1.0], marker="*", ms=7, color="k", ls="none", label="oracle")
    a2.plot([0], [1.0], marker="D", ms=3.0, color="k", mfc="white", ls="none")
    # con falsas alarmas (abiertos)
    FA = (("adaptado_pi0.173", 100 * Rf["adaptado_pi0.173"]["rec"], AZUL, "o"), ("adaptado_pi0.05", 100 * Rf["adaptado_pi0.05"]["rec"], AZUL, "o"),
          ("amplitud_pi0.173", 100 * Rf["amplitud_pi0.173"]["rec"], GRIS, "x"), ("inst_m6_pi0.173", 100 * Rf["inst_m6_pi0.173"]["rec"], ROJO, "+"),
          ("inst_m12_pi0.173", 100 * Rf["inst_m12_pi0.173"]["rec"], ROJO, "1"), ("oraculo", 100, "k", "*"), ("ViSt", 0, "k", "D"))
    for nom, xx, col, mk in FA:
        q = P72[f"x1|{nom}"]
        a2.errorbar([xx], [q["r"]], yerr=[[q["r"] - q["r_ic"][0]], [q["r_ic"][1] - q["r"]]], color=col, marker=mk,
                    mfc="white", ms=(6 if mk == "*" else 3.6), ls="none", capsize=1.2, elinewidth=0.5)
    a2.set_yscale("log")
    a2.set_ylim(0.01, 1.6)
    a2.axhline(1.0, color="0.75", lw=0.6, zorder=0)
    a2.text(55, 1.15, "no false alarms", fontsize=5.0, color="0.3", ha="center")
    a2.text(40, 0.032, "false alarms, rate of fish (open)", fontsize=5.0, color="0.3", ha="center")
    a2.set_xlabel("recovered, one driven leader (%)", fontsize=7)
    a2.set_ylabel(r"persistence $P/P_{\mathrm{ViSt}}$")
    a2.legend(loc="center left", fontsize=5.0, borderaxespad=0.2, labelspacing=0.2)
    fig.savefig(os.path.join(FIG, "f_frontera.pdf"))
    plt.close(fig)


def fig_dos_etapas():
    """E69: (a) valor de leer al líder, g(T), y de leerlo a medias, H(c); (b) recuperación
    predicha por la ley frente a la medida, celda por celda."""
    e69 = cargar("E69_dos_etapas.json")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.1, 2.25), width_ratios=[1, 1, 1.05])
    fig.subplots_adjust(wspace=0.38)
    CEL = (("F4", AZUL, "o", r"driven, $\tau_W=4$ s"), ("M4", NARANJA, "s", r"manoeuvre, $\tau_W=4$ s"),
           ("M1", VERDE, "^", r"manoeuvre, $\tau_W=1$ s"))
    for c, col, mk, lab in CEL:
        g = e69["g"][c]
        a1.plot(g["T"][:-1], [100 * v for v in g["g"][:-1]], color=col, marker=mk, ms=3, lw=1.0, label=lab)
        h = e69["H"][c]
        a2.plot(h["c"], [100 * v for v in h["H"]], color=col, marker=mk, ms=3, lw=1.0)
    a1.set_xlabel("oracle readable for the first $T$ s")
    a1.set_ylabel("advantage recovered (%)")
    a1.legend(loc="lower right", fontsize=5.4, borderaxespad=0.2)
    a2.plot([0, 1], [0, 100], color="0.75", lw=0.6, ls=(0, (3, 2)))
    a2.set_xlabel(r"oracle read at a fraction $c$ of $w$")
    for c, col, mk, lab in CEL:
        pts = [(v["medido"]["rec"], v["L3"]) for k, v in e69["celdas"].items() if k.startswith(c + "|")]
        a3.plot([100 * p[1] for p in pts], [100 * p[0] for p in pts], color=col, marker=mk, ms=3, ls="none")
    a3.plot([-10, 110], [-10, 110], color="0.6", lw=0.6)
    a3.fill_between([-10, 110], [-20, 100], [0, 120], color="0.92", lw=0, zorder=0)
    a3.set_xlim(-10, 110); a3.set_ylim(-10, 110)
    a3.set_xlabel("predicted from the two stages (%)")
    a3.set_ylabel("measured (%)")
    fig.savefig(os.path.join(FIG, "f_dos_etapas.pdf"))
    plt.close(fig)


def fig_entrenados():
    """E74: peces cebra con cinco entrenados, filtros de sólo pasado. (a) AUC entrenado frente a
    desensibilizado contra el instante en que termina la ventana, con un valor, 0,25 s y 1 s, y el
    control con encendidos ficticios; (b) AUC contra la ventana: en los grupos que quedan fuera de
    la elección del cierre y con ventanas que terminan cuando responden los desensibilizados."""
    e73, e74 = cargar("E73_peces_entrenados.json"), cargar("E74_peces_solo_pasado.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.95), width_ratios=[1.35, 1])
    fig.subplots_adjust(wspace=0.42)
    T = np.array(e74["TG"])
    iw = {w: i for i, w in enumerate(e74["W"])}
    c = e74["curso"]["sp"]
    for w, col, lw, lab in (("valor", GRIS, 0.8, "one value"), ("1.0", VERDE, 0.9, "1 s window"),
                            ("0.25", AZUL, 1.1, "0.25 s window")):
        i = iw[w]
        a1.fill_between(T, c["lo"][i], c["hi"][i], color=col, alpha=0.16, lw=0)
        a1.plot(T, c["auc"][i], color=col, lw=lw, label=lab)
    a1.plot(T, e74["control"]["sp"]["auc"][iw["0.25"]], color="0.45", lw=0.7, ls=(0, (3, 2)), label="0.25 s, no stimulus")
    a1.axvline(0, color="0.75", lw=0.6, zorder=0)
    a1.axhline(0.5, color="0.8", lw=0.5, zorder=0)
    r = e73["resp"]
    a1.axvspan(r["tD_iqr"][0], r["tD_iqr"][1], color=NARANJA, alpha=0.10, lw=0, zorder=0)
    a1.text(r["tD_mediana"] + 0.35, 0.405, "untrained\nrespond", fontsize=4.8, color=NARANJA, ha="center", va="bottom")
    a1.set_xlabel("end of window, from light onset (s)", fontsize=6.6)
    a1.set_ylabel("AUC, trained vs untrained", fontsize=7)
    a1.set_xlim(-1, 3); a1.set_ylim(0.40, 0.77)
    a1.legend(loc="upper left", fontsize=4.6, borderaxespad=0.1, labelspacing=0.12, handlelength=1.4)
    W = e74["W"]
    xs = list(range(len(W)))
    f = e74["fuera"]["sp"]["por_ventana"]
    a2.fill_between(xs, [f[w]["ic95"][0] for w in W], [f[w]["ic95"][1] for w in W], color=AZUL, alpha=0.18, lw=0)
    a2.plot(xs, [f[w]["auc"] for w in W], color=AZUL, marker="o", ms=2.8, lw=1.0, label="left-out groups")
    q = e74["antes"]["en_tD"]["sp"]["0"]
    a2.fill_between(xs, [q[w]["ic95"][0] for w in W], [q[w]["ic95"][1] for w in W], color=NARANJA, alpha=0.13, lw=0)
    a2.plot(xs, [q[w]["auc"] for w in W], color=NARANJA, marker="s", ms=2.6, lw=0.9, ls=(0, (3, 2)),
            label="ending at their\nresponse")
    a2.axhline(0.5, color="0.8", lw=0.5, zorder=0)
    a2.set_xticks(xs); a2.set_xticklabels(["val.", ".25", ".5", "1", "2"], fontsize=7)
    a2.set_xlabel("window (s)", fontsize=7)
    a2.set_ylim(0.46, 0.75)
    a2.legend(loc="upper right", fontsize=4.6, borderaxespad=0.1, labelspacing=0.2, handlelength=1.6)
    fig.savefig(os.path.join(FIG, "f_entrenados.pdf"))
    plt.close(fig)


# esquema de Métodos (antes se copiaba de paper_a/figuras.py; misma función)
def fig_esquema():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.0, 2.05))
    for a in (a1, a2):
        a.set_aspect("equal"); a.axis("off")

    # (a) la maniobra: velocidad, direccion deseada, angulo delta
    a1.annotate("", xy=(1.45, 0.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=AZUL))
    a1.text(1.5, -0.02, r"", color=AZUL, fontsize=9, va="center")
    a1.annotate("", xy=(0.95, 1.15), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=NARANJA,
                                linestyle=(0, (4, 2))))
    a1.text(1.0, 1.2, r"$v_0\hat{\mathbf{d}}$", color=NARANJA, fontsize=9)
    th = np.linspace(0, np.arctan2(1.15, 0.95), 40)
    a1.plot(0.55 * np.cos(th), 0.55 * np.sin(th), color="0.35", lw=0.8)
    a1.text(0.66, 0.28, r"$\delta$", fontsize=9, color="0.25")
    a1.annotate("", xy=(0.62, 0.86), xytext=(0.95, 0.42),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=VERDE))
    a1.text(0.99, 0.62, r"$a_\perp\leq a_{\max}$", color=VERDE, fontsize=8)
    a1.scatter([0], [0], s=42, color="k", zorder=5)
    a1.set_xlim(-0.35, 2.1); a1.set_ylim(-0.5, 1.55)
    a1.set_title("the manoeuvre is bounded", fontsize=7.5)

    # (b) el residuo contra el vecindario
    rng = np.random.default_rng(3)
    pos = rng.normal(0, 0.55, (7, 2))
    for p_ in pos:
        ang = rng.normal(0.12, 0.10)
        a2.annotate("", xy=(p_[0] + 0.42 * np.cos(ang), p_[1] + 0.42 * np.sin(ang)),
                    xytext=tuple(p_),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=GRIS))
        a2.scatter(*p_, s=16, color=GRIS)
    a2.annotate("", xy=(0.65 + 0.5 * np.cos(0.95), -1.15 + 0.5 * np.sin(0.95)),
                xytext=(0.65, -1.15),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=AZUL))
    a2.scatter([0.65], [-1.15], s=34, color=AZUL, zorder=5)
    a2.text(1.22, -1.28, r"$\omega_j$", color=AZUL, fontsize=9)
    a2.text(-1.15, 1.0, r"$\langle\omega\rangle_{\mathcal{N}_j}$",
            color="0.35", fontsize=9)
    a2.text(-0.15, -1.75,
            r"$\epsilon_j=\omega_j-\langle\omega\rangle_{\mathcal{N}_j}$",
            fontsize=9, ha="center")
    a2.set_xlim(-1.9, 1.9); a2.set_ylim(-2.1, 1.6)
    a2.set_title("the observable is differential", fontsize=7.5)
    fig.savefig(os.path.join(FIG, "f_esquema.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    que = sys.argv[1:] or ["especies", "escalera", "huella", "frontera", "entrenados", "esquema"]
    if "especies" in que:
        print("especies", fig_especies())
    if "escalera" in que:
        fig_escalera(); print("escalera ok")
    if "huella" in que:
        print("huella", fig_huella())
    if "frontera" in que:
        fig_frontera(); print("frontera ok")
    if "dos_etapas" in que:
        fig_dos_etapas(); print("dos etapas ok")
    if "entrenados" in que:
        fig_entrenados(); print("entrenados ok")
    if "esquema" in que:
        fig_esquema(); print("esquema ok")
