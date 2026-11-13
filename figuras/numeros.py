"""Cifras generadas del Paper A corto.

Escribe `numeros.tex` (macros) y `gen/*.tex` (cuerpos de tabla) desde los JSON de
`resultados/`. El texto del paper usa esas macros en el resumen, los recuadros de
resultado, la conclusión y las tablas, para que ninguna de esas cifras se escriba a mano.

    python figuras/numeros.py
"""

from __future__ import annotations

import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
GEN = os.path.join(AQUI, "gen")
os.makedirs(GEN, exist_ok=True)


def J(n):
    with open(os.path.join(RAIZ, "resultados", n)) as fh:
        return json.load(fh)


e46, e47, e48, e49 = J("E46_referencia_cold.json"), J("E47_huella_modelos.json"), J("E48_replicas.json"), J("E49_sensibilidad.json")
e51, e51b, e52 = J("E51_huella_aves.json"), J("E51b_inyeccion.json"), J("E52_auc_ventana_aves.json")
e53, e54 = J("E53_agilidad_calibracion.json"), J("E54_peces.json")
e55, e56, e58 = J("E55_agilidad.json"), J("E56_ventana_ruido.json"), J("E58_memoria_ganancia.json")
e54b, e57, e57b = J("E54b_prospectivo.json"), J("E57_palomas.json"), J("E57b_potencia.json")
e60, e62, e63 = J("E60_residuo_receptor.json"), J("E62_prior.json"), J("E63_varios_informados.json")
e52b, e54c, e59, e61 = J("E52b_aves_por_grabacion.json"), J("E54c_tetras_por_experimento.json"), J("E59_componentes_huella.json"), J("E61_ventana_peso_fijo.json")

# ---- plan 15 (E66): las celdas de 4 réplicas de E42/E42b, E55, E56 y E61 pasan a 16. Se sobrescriben
# las entradas que usan las tablas y macros de abajo, para que todas salgan de las 16 réplicas.
e66 = J("E66_replicas.json")
_C66, _A66 = e66["celdas"], e66["amplitud"]
_VK = {"0.25": "0.25", "0.5": "0.5", "1": "1.0", "2": "2.0", "4": "4.0"}
for _etq in ("forzado", "maniobra"):
    for _a in ("5", "20", "40"):
        _q = _C66[f"agilidad|a{_a}|N250|{_etq}"]["n16"]
        e55["mejor"][f"{_etq}|{_a}"] = {"ventana": float(_q["mejor"]), "rec": _q["rec"][_q["mejor"]]["rec"]}
        for _v, _r in _q["rec"].items():
            e55["rec"][f"{_etq}|{_a}"][f"ciclo{_v}"] = _r
        e55["rec"][f"{_etq}|{_a}"]["amplitud"] = _A66[f"agilidad|a{_a}|N250|{_etq}"]
    _q = _C66[f"barrido|base|N250|{_etq}"]["n16"]
    e55["mejor"][f"{_etq}|10"] = {"ventana": float(_q["mejor"]), "rec": _q["rec"][_q["mejor"]]["rec"]}
    for _v, _r in _q["rec"].items():
        e55["rec"][f"{_etq}|10"][f"ciclo{_v}"] = _r
    for _f in ("0.5", "2"):
        _q = _C66[f"ruido|f{_f}|N250|{_etq}"]["n16"]
        e56["mejor"][f"{_etq}|{_f}"] = {"ventana": float(_q["mejor"]), "rec": _q["rec"][_q["mejor"]]["rec"]}
        for _v, _r in _q["rec"].items():
            e56["rec"][f"{_etq}|{_f}"][f"ciclo{_v}"] = _r
        e56["rec"][f"{_etq}|{_f}"]["amplitud"] = _A66[f"ruido|f{_f}|N250|{_etq}"]
    for _c in ("a5", "a20", "a40", "f0.5", "f2"):
        _q = _C66[f"peso_fijo|{_c}|N250|{_etq}"]["n16"]
        e61["mejor"][f"{_etq}|{_c}"] = {"ventana": float(_q["mejor"]), "rec": _q["rec"][_q["mejor"]]["rec"]}
        e61["rec"][f"{_etq}|{_c}"]["serie"] = {_VK[_v]: _r["rec"] for _v, _r in _q["rec"].items()}

M = {}                                   # nombre -> cuerpo de la macro


def num(x, nd=2, signo=False):
    s = f"{x:+.{nd}f}" if signo else f"{x:.{nd}f}"
    if s.startswith("-"):
        s = "-" + s[1:]
    return "\\ensuremath{" + s + "}"


def ic(lo, hi, nd=2, signo=False, k=1.0):
    return f"[{num(k * lo, nd, signo)}, {num(k * hi, nd, signo)}]"


def macro(nombre, cuerpo):
    assert nombre.isalpha(), nombre
    assert nombre not in M, nombre
    M[nombre] = cuerpo


def frac(nombre, q, nd=2, signo=False, v="rec"):
    """Fracción con su intervalo: \\nombre y \\nombreCI."""
    macro(nombre, num(q[v], nd, signo))
    macro(nombre + "CI", ic(*q["ic95"], nd, signo))


def pct(nombre, q, nd=1):
    macro(nombre, num(100 * q["rec"], nd))
    macro(nombre + "CI", ic(*q["ic95"], nd, k=100))


def celda(q, nd=2):
    return f"${q['rec']:.{nd}f}$ [${q['ic95'][0]:.{nd}f}$, ${q['ic95'][1]:.{nd}f}$]"


def escribir(nombre, lineas):
    with open(os.path.join(GEN, nombre), "w") as fh:
        fh.write("% generado por numeros.py; no editar a mano\n" + "\n".join(lineas) + "\n")


# ---------------------------------------------------------------- huella en los modelos
a = e47["amplitud"]
macro("HuWiSt", num(a["WiSt"]["115"]["mediana"]))
macro("HuWiStSD", num(a["WiSt"]["115"]["sd"]))
macro("HuCoLd", num(a["CoLd"]["115"]["mediana"]))
macro("HuCoLdSD", num(a["CoLd"]["115"]["sd"]))
macro("TauWSobreHuMin", num(4.0 / a["CoLd"]["115"]["mediana"], 1))
macro("TauWSobreHuMax", num(4.0 / a["WiSt"]["115"]["mediana"], 1))
mo = e51["modelos"]
for m in ("WiSt", "CoLd", "CaSp"):
    r = mo[f"{m}|gauss"]
    macro("Tn" + m, num(r["T95_2s"]["mediana"]))
    macro("Tn" + m + "CI", ic(*r["T95_2s"]["ic95"]))
    macro("Fw" + m, num(r["fwhm"]["mediana"]))
for m in ("WiSt", "CoLd"):
    macro("Tn" + m + "Cinco", num(mo[f"{m}|gauss"]["T95_5s"]["mediana"]))
    macro("Tn" + m + "St", num(mo[f"{m}|student"]["T95_2s"]["mediana"]))
    macro("Tn" + m + "StCI", ic(*mo[f"{m}|student"]["T95_2s"]["ic95"]))
escribir("tab_huella.tex", [
    f"CoLd~\\cite{{couzin2005}} & $\\HuCoLd\\pm\\HuCoLdSD$ & ${mo['CoLd|gauss']['pico']['mediana']:.1f}$ & $\\TnCoLd$ \\TnCoLdCI \\\\",
    f"WiSt~\\cite{{balazs2020}} & $\\HuWiSt\\pm\\HuWiStSD$ & ${mo['WiSt|gauss']['pico']['mediana']:.1f}$ & $\\TnWiSt$ \\TnWiStCI \\\\",
    "\\midrule",
    f"CaSp~\\cite{{cavagna2015}} (control) & --- & ${mo['CaSp|gauss']['pico']['mediana']:.1f}$ & $\\TnCaSp$ \\TnCaSpCI \\\\"])

# ---------------------------------------------------------------- celdas de cabecera
CEL = {"FzA": ("forzado", 250, 4, "15.8"), "FzK": ("forzado", 1000, 4, "31.6"),
       "McA": ("maniobra", 250, 4, "15.8"), "McK": ("maniobra", 1000, 4, "31.6"),
       "MuA": ("maniobra", 250, 1, "15.8"), "MuK": ("maniobra", 1000, 1, "31.6")}
for cod, (etq, N, tau, win) in CEL.items():
    c = e48["celdas"][f"{etq}|{N}|{tau}|{win}"]
    assert c["adaptado"]["32"]["n_rep"] == 32
    pct("Ad" + cod, c["adaptado"]["32"])
    pct("Sm" + cod, c["amplitud"]["32"])
    pct("AdIni" + cod, e48["celdas"][f"{etq}|{N}|{tau}|4"]["adaptado"]["32"])
t4 = [e48["celdas"][f"{e}|{N}|4|{w}"]["adaptado"]["32"]["rec"]
      for e, N, w in (("forzado", 250, "15.8"), ("forzado", 1000, "31.6"),
                      ("maniobra", 250, "15.8"), ("maniobra", 1000, "31.6"))]
t4 += [e46["escalera"][f"{p}|500|4"]["adaptado"]["entero"]["rec"] for p in ("tractor", "pulso")]
macro("RangoMin", num(100 * min(t4), 0))
macro("RangoMax", num(100 * max(t4), 0))
ini = [e48["celdas"][f"{e}|{N}|4|4"]["adaptado"]["32"]["rec"] for e in ("forzado", "maniobra") for N in (250, 1000)]
macro("AdIniMin", num(100 * min(ini), 0))
macro("AdIniMax", num(100 * max(ini), 0))
pct("AdFzM", e46["escalera"]["tractor|500|4"]["adaptado"]["entero"])
pct("AdMcM", e46["escalera"]["pulso|500|4"]["adaptado"]["entero"])
pct("SmFzM", e46["escalera"]["tractor|500|4"]["amplitud"]["entero"])
fac = [e48["celdas"]["forzado|250|4|15.8"]["adaptado"]["32"]["rec"] / e48["celdas"]["forzado|250|4|15.8"]["amplitud"]["32"]["rec"],
       e46["escalera"]["tractor|500|4"]["adaptado"]["entero"]["rec"] / e46["escalera"]["tractor|500|4"]["amplitud"]["entero"]["rec"],
       e48["celdas"]["forzado|1000|4|31.6"]["adaptado"]["32"]["rec"] / e48["celdas"]["forzado|1000|4|31.6"]["amplitud"]["32"]["rec"]]
macro("FactorMin", num(min(fac), 0))
macro("FactorMax", num(max(fac), 0))
# lo que el oráculo añade con maniobra única: 1 - recuperación
macro("RestoMcA", num(100 * (1 - e48["celdas"]["maniobra|250|4|15.8"]["adaptado"]["32"]["rec"]), 0))
macro("RestoMcK", num(100 * (1 - e48["celdas"]["maniobra|1000|4|31.6"]["adaptado"]["32"]["rec"]), 0))

filas = []
for etq, nom in (("forzado", "driven leader"), ("maniobra", "single manoeuvre")):
    for i, N in enumerate((250, 500, 1000)):
        if N == 500:
            q = e46["escalera"][f"{'tractor' if etq == 'forzado' else 'pulso'}|500|4"]
            a4 = q["adaptado"].get("4s") or q["adaptado"].get("primeros4") or q["adaptado"].get("4")
            fila = [celda(a4) if a4 else "---", celda(q["adaptado"]["entero"]),
                    celda(q["amplitud"]["entero"], 3), "$8$"]
        else:
            w = {250: "15.8", 1000: "31.6"}[N]
            c = e48["celdas"]
            fila = [celda(c[f"{etq}|{N}|4|4"]["adaptado"]["32"]), celda(c[f"{etq}|{N}|4|{w}"]["adaptado"]["32"]),
                    celda(c[f"{etq}|{N}|4|{w}"]["amplitud"]["32"], 3), "$32$"]
        filas.append(f"{nom if i == 0 else ''} & ${N}$ & " + " & ".join(fila) + " \\\\")
escribir("tab_cabecera.tex", filas)

filas = []
for tau, Ns in ((1, (250, 1000)), (2, (250,)), (4, (250, 500, 1000))):
    for i, N in enumerate(Ns):
        q = e46["escalera"][f"pulso|{N}|{tau}"]
        clave = f"maniobra|{N}|{tau}|{ {250: '15.8', 500: '22.4', 1000: '31.6'}[N]}"
        if clave in e48["celdas"] and e48["celdas"][clave]["adaptado"]["32"]["n_rep"] == 32:
            c = e48["celdas"][clave]
            ad, sm, vent, n = c["adaptado"]["32"], c["amplitud"]["32"], c["adaptado"]["32"]["ventaja"], 32
        else:
            ad, sm, vent, n = q["adaptado"]["entero"], q["amplitud"]["entero"], q["ventaja"], 8
        filas.append(f"{'$\\tauW=%d$\\,s' % tau if i == 0 else ''} & ${N}$ & ${vent:.3f}$ & "
                     f"${q['cold_sobre_vist']['rec']:.3f}$ & {celda(ad)} & {celda(sm, 3)} & ${n}$ \\\\")
escribir("tab_tauW.tex", filas)

# ---------------------------------------------------------------- E49: k, ruido, autocalibrado
KS, FS = (7, 15, 30), ((0.5, "$\\times0.5$"), (1, "$\\times1$"), (2, "$\\times2$"))
ma = e49["malla"]
g = lambda k, f, o: ma[f"k{k}|f{f:g}"][o]["rec"]
ref = [(k, 1) for k in KS]
macro("KAdMuMin", num(100 * min(g(k, f, "adaptado_cal") for k, f in ref), 0))
macro("KAdMuMax", num(100 * max(g(k, f, "adaptado_cal") for k, f in ref), 0))
macro("KSmMuMin", num(100 * min(g(k, f, "amplitud_cal") for k, f in ref), 0))
macro("KSmMuMax", num(100 * max(g(k, f, "amplitud_cal") for k, f in ref), 0))
cz = e49["cruz"]
macro("KAdFzMin", num(100 * min(cz[f"k{k}|f1"]["adaptado_cal"]["rec"] for k in KS), 0))
macro("KAdFzMax", num(100 * max(cz[f"k{k}|f1"]["adaptado_cal"]["rec"] for k in KS), 0))
macro("KSmFzMin", num(100 * min(cz[f"k{k}|f1"]["amplitud_cal"]["rec"] for k in KS), 0))
macro("KSmFzMax", num(100 * max(cz[f"k{k}|f1"]["amplitud_cal"]["rec"] for k in KS), 0))
pct("SmFzKsiete", cz["k7|f1"]["amplitud_cal"])
macro("NdAdMin", num(100 * min(g(k, 2, "adaptado_cal") for k in KS), 0))
macro("NdAdMax", num(100 * max(g(k, 2, "adaptado_cal") for k in KS), 0))
pct("NdAdFz", cz["k15|f2"]["adaptado_cal"])
macro("NdSmMin", num(100 * min(g(k, 2, "amplitud_cal") for k in KS), 0))
macro("NdSmMax", num(100 * max(g(k, 2, "amplitud_cal") for k in KS), 0))
macro("NhSmMin", num(100 * min(g(k, 0.5, "amplitud_cal") for k in KS), 0))
macro("NhSmMax", num(100 * max(g(k, 0.5, "amplitud_cal") for k in KS), 0))
macro("NhAdMin", num(100 * min(g(k, 0.5, "adaptado_cal") for k in KS), 0))
macro("NhAdMax", num(100 * max(g(k, 0.5, "adaptado_cal") for k in KS), 0))
pct("NhAdFz", cz["k15|f0.5"]["adaptado_cal"])
pct("NhSmFz", cz["k15|f0.5"]["amplitud_cal"])
macro("FiNdMin", num(100 * min(g(k, 2, "adaptado_fijo") for k in KS), 0))
macro("FiNdMax", num(100 * max(g(k, 2, "adaptado_fijo") for k in KS), 0))
macro("FiNhMin", num(100 * min(g(k, 0.5, "adaptado_fijo") for k in KS), 0))
macro("FiNhMax", num(100 * max(g(k, 0.5, "adaptado_fijo") for k in KS), 0))
an = e49["anchuras"]
macro("AnchoRef", num(an["k15|f1"]["sigma"]))
macro("AnchoMin", num(min(v["sigma"] for v in an.values())))
macro("AnchoMax", num(max(v["sigma"] for v in an.values())))
macro("AnchoNhMin", num(min(an[f"k{k}|f0.5"]["sigma"] for k in KS)))
macro("AnchoNhMax", num(max(an[f"k{k}|f0.5"]["sigma"] for k in KS)))
macro("AnchoNdMin", num(min(an[f"k{k}|f2"]["sigma"] for k in KS)))
macro("AnchoNdMax", num(max(an[f"k{k}|f2"]["sigma"] for k in KS)))
macro("GiroSobreAnchoMin", num(1.25 / max(an[f"k{k}|f2"]["sigma"] for k in KS), 1))
macro("GiroSobreAnchoMax", num(1.25 / min(an[f"k{k}|f2"]["sigma"] for k in KS), 1))
sa = e49["sigma_auto"]
macro("AuSigMin", num(100 * (min(v["sobre_medida"] for v in sa.values()) - 1), 0))
macro("AuSigMax", num(100 * (max(v["sobre_medida"] for v in sa.values()) - 1), 0))
macro("AuSigSube", num(100 * (max(v["maximo_sobre_inicial"] for v in sa.values()) - 1), 0))
cab = e49["cabecera"]
rA = [cab[k]["cociente"]["cociente"] for k in cab if "|250|" in k]
rK = [cab[k]["cociente"]["cociente"] for k in cab if "|1000|" in k]
rG = [ma[c]["cociente_auto_cal"]["cociente"] for c in ma]
for nom, v in (("AuRatA", rA), ("AuRatK", rK), ("AuRatG", rG)):
    macro(nom + "Min", num(min(v)))
    macro(nom + "Max", num(max(v)))
pct("AuFzA", cab["forzado|250|4"]["auto"])
pct("AuMcA", cab["maniobra|250|4"]["auto"])
pct("AuMuA", cab["maniobra|250|1"]["auto"])

filas = []
for k in KS:
    for i, (f, ft) in enumerate(FS):
        c = ma[f"k{k}|f{f:g}"]
        filas.append(f"{'$%d$' % k if i == 0 else ''} & {ft} & ${an[f'k{k}|f{f:g}']['sigma']:.3f}$ & "
                     f"${c['adaptado_cal']['ventaja']:.3f}$ & {celda(c['adaptado_cal'])} & "
                     f"{celda(c['amplitud_cal'])} & ${c['adaptado_fijo']['rec']:.2f}$ & "
                     f"${c['adaptado_auto']['rec']:.2f}$ & ${c['cociente_auto_cal']['cociente']:.2f}$ \\\\")
escribir("tab_malla.tex", filas)
filas = []
for k, f, ft in ((7, 1, "$\\times1$"), (15, 0.5, "$\\times0.5$"), (15, 1, "$\\times1$"), (15, 2, "$\\times2$"), (30, 1, "$\\times1$")):
    c = cz[f"k{k}|f{f:g}"]
    filas.append(f"${k}$ & {ft} & {celda(c['adaptado_cal'])} & {celda(c['adaptado_auto'])} & "
                 f"{celda(c['amplitud_cal'], 3)} \\\\")
escribir("tab_cruz.tex", filas)
filas = []
for N in (250, 1000):
    for i, (clave, nom) in enumerate((("forzado|%d|4", "driven leader"), ("maniobra|%d|4", "single manoeuvre, $\\tauW=4$\\,s"),
                                      ("maniobra|%d|1", "single manoeuvre, $\\tauW=1$\\,s"))):
        c = cab[clave % N]
        co = c["cociente"]
        ex = (f"${c['auto2']['rec']:.2f}$ / ${c['auto10']['rec']:.2f}$" if "auto2" in c else "---")
        filas.append(f"{nom} & ${N}$ & {celda(c['fijo'])} & {celda(c['auto'])} & "
                     f"${co['cociente']:.2f}$ [${co['ic95'][0]:.2f}$, ${co['ic95'][1]:.2f}$] & {ex} \\\\")
escribir("tab_auto.tex", filas)

# ---------------------------------------------------------------- aves
k7 = e51["k7_min12"]
macro("GrT", num(k7["T95"]["mediana"]))
macro("GrTCI", ic(*k7["T95"]["ic95"]))
macro("GrTnulo", num(k7["T95_nulo_media"]["mediana"]))
macro("GrTg", num(k7["T95_grupo"]["mediana"]))
macro("GrTgCI", ic(*k7["T95_grupo"]["ic95"]))
macro("GrFw", num(k7["fwhm"]["mediana"]))
macro("GrFwCI", ic(*k7["fwhm"]["ic95"]))
macro("GrNini", str(k7["T95_por_ave"]["n_aves"]))
macro("GrFracNulo", num(100 * k7["T95_por_ave"]["frac_sobre_nulo_p95"], 0))
k3 = e51["k3_min12"]
macro("GrTtres", num(k3["T95"]["mediana"]))
macro("GrTtresCI", ic(*k3["T95"]["ic95"]))
macro("GrFwtres", num(k3["fwhm"]["mediana"]))
macro("GrFwtresCI", ic(*k3["fwhm"]["ic95"]))
for m in ("WiSt", "CoLd"):
    macro("Iny" + m, num(e51b["agregado"][m]["mediana"]))
    macro("Iny" + m + "CI", ic(*e51b["agregado"][m]["ic95"]))
k12 = e51["k12_min12"]
macro("GrTdoce", num(k12["T95"]["mediana"]))
macro("GrTdoceCI", ic(*k12["T95"]["ic95"]))
macro("GrDurUmbMin", num(min(k7["agregada"]["dur"].values())))
macro("GrDurUmbMax", num(max(k7["agregada"]["dur"].values())))
macro("GrFwAgregada", num(k7["agregada"]["fwhm"]))
ex = e51["exploratorio"]
macro("GrRhoAmp", num(ex["amplitud_deg"]["spearman_con_dur15"]))
macro("GrRhoAmpP", num(ex["amplitud_deg"]["p"], 3))

r = e52["k3|1|todos"]
macro("GrNev", str(r["n_eventos"]))
macro("GrNiniAuc", str(r["n_ini"]))
macro("GrNsegAuc", str(r["n_seg"]))
macro("GrAucUno", num(r["curva"]["1.0"]["auc"]))
macro("GrAucUnoCI", ic(*e52b["auc1_ic95"]))                     # por grabación (plan 14)
macro("GrAucUnoP", num(r["curva"]["1.0"]["p_perm"], 4))
macro("GrAucCuadro", num(r["curva"]["cuadro"]["auc"]))
macro("GrAucCuadroCI", ic(*e52b["auc_cuadro_ic95"]))
macro("GrDelta", num(r["delta"]["delta"], 3, True))
macro("GrDeltaCI", ic(*e52b["ganancia_ic95"], 3, True))
macro("GrAucUnoEvCI", ic(*r["curva"]["1.0"]["ic95"]))           # por eventos (E52), como sensibilidad
macro("GrDeltaEvCI", ic(*r["delta"]["ic95"], 3, True))
macro("GrTau", num(r["tau95"]["valor"]))
macro("GrTauCI", ic(*r["tau95"]["ic95"], 1))
for cod, cl in (("Tr", "k3|1|transito"), ("Mo", "k3|1|mobbing")):
    macro("GrDelta" + cod, num(e52[cl]["delta"]["delta"], 2, True))
    macro("GrDelta" + cod + "CI", ic(*e52[cl]["delta"]["ic95"], 2, True))
r7 = e52["k7|1|todos"]
macro("GrAucUnoSiete", num(r7["curva"]["1.0"]["auc"]))
macro("GrAucUnoSieteCI", ic(*r7["curva"]["1.0"]["ic95"]))
macro("GrDeltaSiete", num(r7["delta"]["delta"], 3, True))
macro("GrDeltaSieteCI", ic(*r7["delta"]["ic95"], 3, True))
s = e52["k3|0.7|todos"]
v = [x["auc"] for x in s["curva"].values()]
macro("GrAucMuestraMin", num(min(v)))
macro("GrAucMuestraMax", num(max(v)))
macro("GrNevMuestra", str(s["n_eventos"]))
ag = e53["agilidad_min12"]
macro("GrOm", num(ag["omega_max"]))
macro("GrOmCI", ic(*ag["ic95"]))
macro("GrNiniOm", str(ag["n_iniciadoras"]))

t17, tin = e53["17 bandadas distintas"], e53["17 bandadas, sin los cuadros de los eventos curados"]
filas = []
for modo, mt in (("horizontal", "horizontal (2D)"), ("3d", "full 3D")):
    for ctx, ct in (("transit", "transit"), ("mobbing", "mobbing")):
        a_, b_ = t17[f"{modo}|{ctx}"], tin[f"{modo}|{ctx}"]
        extra = f"${b_['sigma']:.3f}$ / ${b_['nu']:.1f}$" if ctx == "mobbing" else "---"
        filas.append(f"{mt} & {ct} & ${a_['bandadas']}$ & ${a_['sigma']:.3f}$ & ${a_['nu']:.1f}$ & {extra} \\\\")
escribir("tab_calib.tex", filas)
macro("CalFactor", num(t17["factor_contexto_2d"], 1))
macro("CalExcesoTr", num(100 * t17["exceso_2d"]["transit"], 0))
macro("CalExcesoMo", num(100 * t17["exceso_2d"]["mobbing"], 0))
macro("CalTrDosD", num(t17["horizontal|transit"]["sigma"], 3))
macro("CalMoDosD", num(t17["horizontal|mobbing"]["sigma"], 3))

# ---------------------------------------------------------------- peces
t = e54["N8_10"]
macro("TeNexp", str(t["n_exp"]))
macro("TeNev", str(t["n_eventos"]))
macro("TeAucCuadro", num(t["curva"]["cuadro"]))
macro("TeAucCuadroCI", ic(*t["curva_ic95"]["cuadro"]))
macro("TeAucUno", num(t["curva"]["1.0"]))
macro("TeAucUnoCI", ic(*t["curva_ic95"]["1.0"]))
macro("TeDelta", num(t["delta_1s_cuadro"], 3, True))
macro("TeDeltaCI", ic(*t["delta_ic95"], 3, True))
macro("TeTau", num(t["tau95"], 1))
macro("TeTauCI", ic(*t["tau95_ic95"], 1))
macro("TeOm", num(t["omega_alat"]["mediana"]))
macro("TeOmCI", ic(*t["omega_alat"]["ic95"]))
macro("TeOmRumbo", num(t["omega"]["mediana"]))
macro("TeOmRumboCI", ic(*t["omega"]["ic95"]))
macro("TeT", num(t["t95_2"]["mediana"]))
macro("TeTCI", ic(*t["t95_2"]["ic95"]))
macro("TeTnuloMedia", num(t["t95_nulo"]["media"]))
macro("TeTcinco", num(t["t95_5"]["mediana"]))
macro("TeTcincoCI", ic(*t["t95_5"]["ic95"]))
macro("TeFw", num(t["fwhm_por_experimento"]["mediana"]))
macro("TeFwCI", ic(*t["fwhm_por_experimento"]["ic95"]))
macro("TeHuDos", num(t["hue2"]["mediana"]))
macro("TeHuCinco", num(t["hue5"]["mediana"]))
macro("CocienteAgilidad", num(t["omega_alat"]["mediana"] / ag["omega_max"], 1))
macro("CocienteAgilidadModelo", num(t["omega_alat"]["mediana"] / 1.25, 1))
macro("VentanaEscaladaMin", num(1.0 * 1.25 / t["omega_alat"]["mediana"]))
macro("VentanaEscaladaMax", num(2.0 * 1.25 / t["omega_alat"]["mediana"]))
porN = [e54[f"N{n}"] for n in (4, 5, 8, 10)]
macro("TeTauNMin", num(min(x["tau95"] for x in porN), 1))
macro("TeTauNMax", num(max(x["tau95"] for x in porN), 1))
macro("TeAucNMin", num(min(x["curva"]["1.0"] for x in porN)))
macro("TeAucNMax", num(max(x["curva"]["1.0"] for x in porN)))
d = e54["deteccion"]
macro("TeDetOchoMin", str(min(x["det_N8"] for x in d)))
macro("TeDetOchoMax", str(max(x["det_N8"] for x in d)))
macro("TeDetAucMin", num(min(x["auc_1s"] for x in d)))
macro("TeDetAucMax", num(max(x["auc_1s"] for x in d)))
macro("TeDetTauMin", num(min(x["tau95"] for x in d), 1))
macro("TeDetTauMax", num(max(x["tau95"] for x in d), 1))
fila = lambda et, f: et + " & " + " & ".join(f(x) for x in porN) + " \\\\"
escribir("tab_tetraN.tex", [
    fila("experiments with detected U-turns", lambda x: f"${x['n_exp']}$"),
    fila("U-turns detected", lambda x: f"${x['n_eventos']}$"),
    fila("$\\wmax$ (rad\\,s$^{-1}$)", lambda x: f"${x['omega_alat']['mediana']:.2f}$"),
    fila("AUC, single frame", lambda x: f"${x['curva']['cuadro']:.3f}$"),
    fila("AUC at $1$\\,s", lambda x: f"${x['curva']['1.0']:.3f}$"),
    fila("gain, $1$\\,s against single frame", lambda x: f"${x['delta_1s_cuadro']:+.3f}$"),
    fila("shortest window within $95\\,\\%$ of the maximum (s)",
         lambda x: f"${x['tau95']:.1f}$ [${x['tau95_ic95'][0]:.1f}$, ${x['tau95_ic95'][1]:.1f}$]"),
    fila("$T_{95}$ of the first fish to reverse (s)", lambda x: f"${x['t95_2']['mediana']:.2f}$")])
escribir("tab_especies.tex", [
    "turning agility $\\wmax$ (rad\\,s$^{-1}$) & $\\GrOm$ \\GrOmCI & $\\TeOm$ \\TeOmCI \\\\",
    "AUC, single frame & $\\GrAucCuadro$ \\GrAucCuadroCI & $\\TeAucCuadro$ \\TeAucCuadroCI \\\\",
    "AUC, $1$\\,s window & $\\GrAucUno$ \\GrAucUnoCI & $\\TeAucUno$ \\TeAucUnoCI \\\\",
    "gain from integrating & $\\GrDelta$ \\GrDeltaCI & $\\TeDelta$ \\TeDeltaCI \\\\",
    "shortest window within $95\\,\\%$ of the maximum (s) & $\\GrTau$ \\GrTauCI & $\\TeTau$ \\TeTauCI \\\\",
    "$T_{95}$ of an initiator (s) & $\\GrTtres$ \\GrTtresCI & $\\TeT$ \\TeTCI \\\\"])
escribir("tab_deteccion.tex", [
    f"${x['om_min']:.2f}$ & ${x['sep']:.0f}$ & ${x['det_N8']}$ & ${x['det_N10']}$ & ${x['auc_cuadro']:.3f}$ & "
    f"${x['auc_1s']:.3f}$ & ${x['tau95']:.1f}$ \\\\" for x in d])


# ---------------------------------------------------------------- tetras, control prospectivo (E54b)
pb = e54b["con_primer_pez"]
macro("TeLead", num(e54b["lead_mediana_s"]))
for cod, fin in (("Cero", "d0"), ("Cuarto", "d0.25"), ("Medio", "d0.5")):
    q = pb[fin]
    macro("TePro" + cod, num(q["1.0"]["auc"]))
    macro("TePro" + cod + "CI", ic(*q["1.0"]["ic95"]))
    macro("TeProCuadro" + cod, num(q["cuadro"]["auc"]))
    macro("TeProG" + cod, num(q["ganancia_1s"]["valor"], 3, True))
    macro("TeProG" + cod + "CI", ic(*q["ganancia_1s"]["ic95"], 3, True))
sp = e54b["sin_primer_pez"]
macro("TeProSinMin", num(min(sp[f]["1.0"]["auc"] for f in ("d0.25", "d0.5"))))
macro("TeProSinMax", num(max(sp[f]["1.0"]["auc"] for f in ("d0.25", "d0.5"))))

# ---------------------------------------------------------------- palomas (E57)
t57 = e57["vuelo"]["tabla"]
q = t57["lider|posteriores|1"]
macro("PaD", num(q["media"], 3, True)); macro("PaDCI", ic(*q["ic95"], 3, True)); macro("PaDp", num(q["p_signos"]))
macro("PaPdes", num(t57["lider|posteriores|1"]["P_desplazada"], 3)); macro("PaPctl", num(t57["lider|posteriores|1"]["P_control"], 3))
vs = [t57[f"lider|posteriores|{v}"]["media"] for v in ("0.2", "0.6", "1", "2", "4", "10")]
macro("PaDMin", num(min(vs), 3, True)); macro("PaDMax", num(max(vs), 3, True))
q = t57["lider|posteriores|rumbo"]
macro("PaRumbo", num(q["media"], 1, True)); macro("PaRumboCI", ic(*q["ic95"], 1, True))
q = t57["segundo|posteriores|1"]
macro("PaSeg", num(q["media"], 3, True)); macro("PaSegCI", ic(*q["ic95"], 3, True))
q = e57["2km"]["tabla"]["lider|posteriores|1"]
macro("PaDdos", num(q["media"], 3, True)); macro("PaDdosCI", ic(*q["ic95"], 3, True))
for v, cod in (("0.2", "Cero"), ("1", "Uno"), ("4", "Cuatro")):
    q = t57[f"lider|posteriores|{v}|claros"]
    macro("PaClaro" + cod, num(q["media"], 3, True)); macro("PaClaro" + cod + "p", num(q["p_signos"]))

for tasa, cod in (("1", "Uno"), ("3", "Tres"), ("10", "Diez")):
    q = e57b["tasas"][tasa]
    macro("PaIny" + cod, num(q["D"], 3, True)); macro("PaIny" + cod + "CI", ic(*q["ic95"], 3, True))

# ---------------------------------------------------------------- agilidad (E55)
filas = []
for a, om_ in (("5", "0.63"), ("10", "1.25"), ("20", "2.5"), ("40", "5")):
    hc, hw = e55["huella"][f"CoLd|{a}"], e55["huella"][f"WiSt|{a}"]
    mf, mm = e55["mejor"][f"forzado|{a}"], e55["mejor"][f"maniobra|{a}"]
    filas.append(f"${a}$ & ${om_}$ & ${hc['dur']:.2f}$ & ${hw['dur']:.2f}$ & ${mf['ventana']:g}$ & ${mm['ventana']:g}$ & "
                 f"${mf['rec']:.2f}$ & ${mm['rec']:.2f}$ \\\\")
escribir("tab_agilidad.tex", filas)
macro("AgVentMax", num(e55["mejor"]["forzado|5"]["ventana"], 0))
macro("AgVentMin", num(e55["mejor"]["forzado|40"]["ventana"]))
macro("AgRecCuarenta", num(100 * e55["rec"]["forzado|40"]["ciclo1"]["rec"], 1))
macro("AgRecCinco", num(100 * e55["rec"]["forzado|5"]["ciclo1"]["rec"], 0))
macro("AgMejorCuarentaF", num(100 * e55["mejor"]["forzado|40"]["rec"], 0))
macro("AgMejorCuarentaM", num(100 * e55["mejor"]["maniobra|40"]["rec"], 0))

# ---------------------------------------------------------------- ventana con otro ruido (E56)
for f, cod in (("0.5", "Nh"), ("2", "Nd")):
    mf, mm = e56["mejor"][f"forzado|{f}"], e56["mejor"][f"maniobra|{f}"]
    macro(cod + "VentF", num(mf["ventana"], 1 if mf["ventana"] < 1 else 0))
    macro(cod + "VentM", num(mm["ventana"], 1 if mm["ventana"] < 1 else 0))
    macro(cod + "RecMejorF", num(100 * mf["rec"], 0))
    macro(cod + "RecMejorM", num(100 * mm["rec"], 0))

# ---------------------------------------------------------------- memoria frente a ganancia (E58)
for etq, cod in (("forzado", "F"), ("maniobra", "M")):
    r = e58["rec"][etq]
    macro("MgInst" + cod, num(100 * max(r[f"inst_m{x}"]["rec"] for x in ("3", "6", "12", "24")), 0))
    macro("MgCortaFija" + cod, num(100 * r["ciclo0.25_n12"]["rec"], 0))
    macro("MgUnoFija" + cod, num(100 * r["ciclo1_nef12"]["rec"], 0))
    macro("MgLargaFija" + cod, num(100 * r["ciclo4_n12"]["rec"], 0))
    macro("MgGseis" + cod, num(100 * r["ciclo1_n6"]["rec"], 0))
    macro("MgGtres" + cod, num(100 * r["ciclo1_n3"]["rec"], 0))
    macro("MgGcuarenta" + cod, num(100 * r["ciclo1_n48"]["rec"], 0))

filas = []
for etq, nom in (("forzado", "driven leader"), ("maniobra", "single manoeuvre")):
    r = e58["rec"][etq]
    filas.append(f"{nom} & memoryless, weight $\\times1$/$3$/$6$/$12$/$24$ & " + " / ".join(
        f"${r[f'inst_m{x}']['rec']:.2f}$" for x in ("1", "3", "6", "12", "24")) + " \\\\")
    filas.append(f" & windows $0.25$/$0.5$/$1$/$2$/$4$\\,s, weight tied to window & " + " / ".join(
        f"${r[k]['rec']:.2f}$" for k in ("ciclo0.25_nef3", "ciclo0.5_nef6", "ciclo1_nef12", "ciclo2_nef24", "ciclo4_nef48")) + " \\\\")
    filas.append(f" & windows $0.25$/$0.5$/$1$/$2$/$4$\\,s, weight of $1$\\,s & " + " / ".join(
        f"${(r[k] if k in r else r['ciclo1_nef12'])['rec']:.2f}$" for k in ("ciclo0.25_n12", "ciclo0.5_n12", "ciclo1_n12", "ciclo2_n12", "ciclo4_n12")) + " \\\\")
    filas.append(f" & $1$\\,s window, weight $\\times\\tfrac14$/$\\tfrac12$/$1$/$2$/$4$ & " + " / ".join(
        f"${(r[k] if k in r else r['ciclo1_nef12'])['rec']:.2f}$" for k in ("ciclo1_n3", "ciclo1_n6", "ciclo1_n12x", "ciclo1_n24", "ciclo1_n48")) + " \\\\")
escribir("tab_ganancia.tex", filas)
filas = []
for f, et in (("0.5", "$\\times0.5$"), ("2", "$\\times2$")):
    for etq, nom in (("forzado", "driven"), ("maniobra", "manoeuvre")):
        r = e56["rec"][f"{etq}|{f}"]
        filas.append(f"{et} & {nom} & ${r['amplitud']['rec']:.2f}$ & " + " & ".join(
            f"${r[f'ciclo{v}']['rec']:.2f}$" for v in ("0.25", "0.5", "1", "2", "4")) + " \\\\")
escribir("tab_ruido_ventana.tex", filas)

# ---------------------------------------------------------------- plan 14: valores absolutos (absolutos_E48)
ab = J("absolutos_E48.json")
filas = []
for cl, nom, Nn in (("forzado|250|4", "driven, $\\tauW=4$\\,s", "250"), ("maniobra|250|4", "manoeuvre, $\\tauW=4$\\,s", "250"),
                    ("maniobra|250|1", "manoeuvre, $\\tauW=1$\\,s", "250"), ("forzado|1000|4", "driven, $\\tauW=4$\\,s", "1000"),
                    ("maniobra|1000|4", "manoeuvre, $\\tauW=4$\\,s", "1000"), ("maniobra|1000|1", "manoeuvre, $\\tauW=1$\\,s", "1000")):
    q = ab[cl]
    filas.append(f"{nom} & ${Nn}$ & ${q['R_ref']:.3f}$ & ${q['R_oraculo']:.3f}$ & ${q['R_observador']:.3f}$ & "
                 f"${q['R_sin_memoria']:.3f}$ & ${q['observador_menos_oraculo']:+.3f}$ "
                 f"{ic(*q['observador_menos_oraculo_ic95'], 3, True)} \\\\")
escribir("tab_absolutos.tex", filas)
macro("AbVentMuA", num(ab["maniobra|250|1"]["ventaja_oraculo"], 3))
macro("AbVentMuK", num(ab["maniobra|1000|1"]["ventaja_oraculo"], 3))
macro("AbVentMcA", num(ab["maniobra|250|4"]["ventaja_oraculo"], 3))
macro("AbVentMcK", num(ab["maniobra|1000|4"]["ventaja_oraculo"], 3))

# ---------------------------------------------------------------- plan 14: aves por grabación (E52b)
g = e52b
macro("GrNrec", str(len(g["unidades"])))
macro("GrLooAucMin", num(min(x["auc1"] for x in g["dejando_una_fuera"])))
macro("GrLooAucMax", num(max(x["auc1"] for x in g["dejando_una_fuera"])))
macro("GrIgualAuc", num(g["por_evento_igual_peso"]["auc1"]))
macro("GrIgualDelta", num(g["por_evento_igual_peso"]["ganancia"], 2, True))

# ---------------------------------------------------------------- plan 14: tetras por experimento (E54c)
c = e54c
macro("TeLooAucMin", num(c["loo_rango"]["auc1"][0]))
macro("TeLooAucMax", num(c["loo_rango"]["auc1"][1]))
macro("TeLooDeltaMin", num(c["loo_rango"]["ganancia"][0], 2, True))
macro("TeLooDeltaMax", num(c["loo_rango"]["ganancia"][1], 2, True))
macro("TeIgualAuc", num(c["igual_peso"]["auc1"]))
macro("TeIgualDelta", num(c["igual_peso"]["ganancia"], 2, True))
macro("TeTauExceso", num(c["agrupada"]["tau95_exceso"], 1))
macro("TeMayor", str(c["sin_el_mayor"]["giros"]))

# ---------------------------------------------------------------- plan 14: componentes de la huella (E59)
P59 = e59["por_celda"]
grandes = [f"{m}|{g}" for m in ("CoLd", "WiSt") for g in ("153", "172")]
macro("HlLatMin", num(min(v["t_lat"] for v in P59.values())))
macro("HlLatMax", num(max(v["t_lat"] for v in P59.values())))
macro("HlVminMin", num(min(P59[c]["v_min"] for c in grandes)))
macro("HlVminMax", num(max(P59[c]["v_min"] for c in grandes)))
macro("HlAlinMin", num(min(P59[f"CoLd|{g}"]["t_alin"] for g in ("115", "153", "172")), 1))
macro("HlAlinMax", num(max(P59[f"CoLd|{g}"]["t_alin"] for g in ("115", "153", "172")), 1))
for m in ("CoLd", "WiSt"):
    macro("HlOmL" + m + "Min", num(min(P59[f"{m}|{g}"]["omL_en_hu"] for g in ("57", "115", "153", "172")), 1))
    macro("HlOmL" + m + "Max", num(max(P59[f"{m}|{g}"]["omL_en_hu"] for g in ("57", "115", "153", "172")), 1))
    macro("HlOmV" + m + "Min", num(min(P59[f"{m}|{g}"]["omV_en_hu"] for g in ("57", "115", "153", "172")), 1))
    macro("HlOmV" + m + "Max", num(max(P59[f"{m}|{g}"]["omV_en_hu"] for g in ("57", "115", "153", "172")), 1))
ant = [P59[f"WiSt|{g}"]["t_propio"] - P59[f"WiSt|{g}"]["t_hu"] for g in ("57", "115", "153", "172")]
macro("HlAntesMin", num(min(ant), 1))
macro("HlAntesMax", num(max(ant), 1))
filas = []
for m in ("CoLd", "WiSt"):
    for g in ("57", "115", "153", "172"):
        v = P59[f"{m}|{g}"]
        filas.append(f"{m if g == '57' else ''} & ${g}^\\circ$ & ${v['t_hu']:.2f}$ & ${v['t_lat']:.2f}$ & ${v['t_propio']:.2f}$ & "
                     f"${v['t_alin']:.2f}$ & ${v['v_min']:.2f}$ & ${v['omL_en_hu']:.1f}$ & ${v['omV_en_hu']:.1f}$ \\\\")
escribir("tab_componentes.tex", filas)

# ---------------------------------------------------------------- plan 14: ventana con el peso fijo (E61)
B61 = e61["mejor"]
macro("PfAgMax", num(B61["forzado|a5"]["ventana"], 0))
macro("PfAgMin", num(B61["forzado|a40"]["ventana"]))
macro("PfNh", num(B61["forzado|f0.5"]["ventana"], 1))
macro("PfNd", num(B61["forzado|f2"]["ventana"], 0))
macro("PfNdRecF", num(100 * B61["forzado|f2"]["rec"], 0))
macro("PfNdRecM", num(100 * B61["maniobra|f2"]["rec"], 0))
def atado(etq, c):
    if c.startswith("a"):
        return e55["mejor"][f"{etq}|{c[1:]}"]["ventana"]
    return e56["mejor"][f"{etq}|{c[1:]}"]["ventana"]


filas = []
for c, et in (("a5", "$a_{\\max}=5$"), ("a20", "$a_{\\max}=20$"), ("a40", "$a_{\\max}=40$"),
              ("f0.5", "noise $\\times0.5$"), ("f2", "noise $\\times2$")):
    for etq, nom in (("forzado", "driven"), ("maniobra", "manoeuvre")):
        sr = e61["rec"][f"{etq}|{c}"]["serie"]
        filas.append(f"{et if etq == 'forzado' else ''} & {nom} & " + " & ".join(f"${sr[v]:.2f}$" for v in ("0.25", "0.5", "1.0", "2.0", "4.0"))
                     + f" & ${B61[f'{etq}|{c}']['ventana']:g}$ & ${atado(etq, c):g}$ \\\\")
escribir("tab_peso_fijo.tex", filas)


# ---------------------------------------------------------------- plan 14: residuo del receptor (E60)
R60, I60 = e60["rec"], e60["id"]
CEL = (("forzado|4", "Fz"), ("maniobra|4", "Mc"), ("maniobra|1", "Mu"))
for c, cod in CEL:
    macro("RrEm" + cod, num(100 * R60[c]["emisor"]["rec"], 0))
    macro("RrLoc" + cod, num(100 * R60[c]["par_local"]["rec"], 0))
perd = [100 * (R60[c]["emisor"]["rec"] - R60[c]["par_propio"]["rec"]) for c, _ in CEL]
macro("RrPerdMin", num(min(perd), 0)); macro("RrPerdMax", num(max(perd), 0))
macro("RrHitFz", num(100 * I60["forzado|4"]["emisor"]["HIT"], 0))
macro("RrHitMMin", num(100 * min(I60[c]["emisor"]["HIT"] for c in ("maniobra|4", "maniobra|1")), 0))
macro("RrHitMMax", num(100 * max(I60[c]["emisor"]["HIT"] for c in ("maniobra|4", "maniobra|1")), 0))
macro("RrAucMin", num(min(I60[c]["emisor"]["AUROC4"] for c, _ in CEL)))
macro("RrAucMax", num(max(I60[c]["emisor"]["AUROC4"] for c, _ in CEL)))
macro("RrFa", num(100 * max(I60[c][m]["FA"] for c, _ in CEL for m in ("emisor", "par_propio", "par_local")), 2))
macro("RrPersMu", num(I60["maniobra|1"]["emisor"]["PERS"], 1))
filas = []
for c, nom in (("forzado|4", "driven, $\\tauW=4$\\,s"), ("maniobra|4", "manoeuvre, $\\tauW=4$\\,s"), ("maniobra|1", "manoeuvre, $\\tauW=1$\\,s")):
    r, i = R60[c], I60[c]["emisor"]
    filas.append(f"{nom} & ${r['emisor']['rec']:.2f}$ & ${r['par_propio']['rec']:.2f}$ & ${r['par_local']['rec']:.2f}$ "
                 f"{ic(*r['par_local']['ic95'])} & ${i['HIT']:.2f}$ & ${i['AUROC4']:.2f}$ & ${100 * i['FA']:.2f}$ \\\\")
escribir("tab_receptor.tex", filas)


# ---------------------------------------------------------------- plan 14: revisor (cruz con líder forzado, E61b)
cz2 = e49["cruz"]
orden = ((7, 1), (15, 1), (30, 1), (15, 0.5), (15, 2))
macro("CruzAd", ", ".join(num(cz2[f"k{k}|f{f:g}"]["adaptado_cal"]["rec"]) for k, f in orden[:-1]) + " and " + num(cz2["k15|f2"]["adaptado_cal"]["rec"]))
macro("CruzSm", ", ".join(num(cz2[f"k{k}|f{f:g}"]["amplitud_cal"]["rec"], 3) for k, f in orden[:-1]) + " and " + num(cz2["k15|f2"]["amplitud_cal"]["rec"], 3))
e61b = J("E61b_bootstrap_ventana.json")
fr = [v["frac"][f"{v['mejor']:g}" if f"{v['mejor']:g}" in v["frac"] else str(v["mejor"])] for v in e61b.values()]
macro("EbMin", num(100 * min(fr), 0))
macro("EbCien", str(sum(f >= 0.995 for f in fr)))
macro("EbTotal", str(len(fr)))


# ---------------------------------------------------------------- plan 14: prior (E62)
for c, cod in (("forzado|4", "Fz"), ("maniobra|4", "Mc"), ("maniobra|1", "Mu")):
    macro("PrBase" + cod, num(100 * e62["rec"][c]["0.004"]["rec"], 0))
    macro("PrCinco" + cod, num(100 * e62["rec"][c]["0.05"]["rec"], 0))


# ---------------------------------------------------------------- plan 14: varios informados (E63)
CE = ("forzado|4", "maniobra|4", "maniobra|1")
for pk, cod in (("0.048", "Fr"), ("0.173", "Ja"), ("0.004", "Ba")):
    v = [100 * e63["rec"][c][pk]["rec"] for c in CE]
    macro("Mi" + cod + "Min", num(min(v), 0)); macro("Mi" + cod + "Max", num(max(v), 0))
va = [e63["abs"][c]["ventaja"] for c in CE]
macro("MiVentMin", num(min(va))); macro("MiVentMax", num(max(va)))
pb = [100 * e62["rec"][c]["0.004"]["rec"] for c in ("forzado|4", "maniobra|4", "maniobra|1")]
macro("PrBaseMin", num(min(pb), 0)); macro("PrBaseMax", num(max(pb), 0))
filas = []
for c, nom in zip(CE, ("driven, $\\tauW=4$\\,s", "manoeuvre, $\\tauW=4$\\,s", "manoeuvre, $\\tauW=1$\\,s")):
    a, r = e63["abs"][c], e63["rec"][c]
    filas.append(f"{nom} & ${a['R_CoLd']:.3f}$ & ${a['R_oraculo']:.3f}$ & ${r['0.048']['rec']:.2f}$ {ic(*r['0.048']['ic95'])} & "
                 f"${r['0.173']['rec']:.2f}$ & ${r['0.004']['rec']:.2f}$ \\\\")
escribir("tab_multi.tex", filas)


# ---------------------------------------------------------------- plan 15: persistencia (E64)
e64 = J("E64_persistencia.json")
P64, R64 = e64["P"], e64["R"]


def pq(nom):
    q = P64[nom]
    return {"rec": q["ratio_incr"], "ic95": q["ratio_incr_ic"]}


frac("PeAdJa", pq("adaptado_pi0.173"))
frac("PeAdTres", pq("adaptado_pi0.3"))
frac("PeSmSeis", pq("inst_m6_pi0.173"))
frac("PeSmDoce", pq("inst_m12_pi0.173"))
macro("PeViSt", num(P64["ViSt"]["P_incr"]))
pe_sm = [P64[f"amplitud_pi{p}"]["ratio_incr"] for p in ("0.004", "0.01", "0.05", "0.173", "0.3")]
macro("PeSmMin", num(min(pe_sm)))
pe_ot = [P64[n]["ratio_incr"] for n in ("secuencial_pi0.004", "secuencial_pi0.173", "auto_pi0.004", "auto_pi0.173")]
macro("PeOtMin", num(min(pe_ot)))
macro("PeAtJa", num(P64["adaptado_pi0.173"]["att"], 3))
macro("PeFpSeis", num(100 * P64["inst_m6_pi0.173"]["fpr"], 0))
macro("PeFpDoce", num(100 * P64["inst_m12_pi0.173"]["fpr"], 0))
rsm = [100 * R64[c][n]["rec"] for c in ("forzado|4", "maniobra|4", "maniobra|1") for n in ("inst_m6_pi0.173", "inst_m12_pi0.173")]
macro("PeRecSmMax", num(max(rsm), 0))
filas = []
NOMS = [("integrating, $1$\\,s", "adaptado", ("0.004", "0.01", "0.05", "0.173", "0.3")),
        ("memoryless", "amplitud", ("0.004", "0.05", "0.173", "0.3")),
        ("sequential", "secuencial", ("0.004", "0.173")), ("self-calibrated", "auto", ("0.004", "0.173"))]
for et, base, pis in NOMS:
    for i, p in enumerate(pis):
        k = f"{base}_pi{p}"
        q = P64[k]
        rr = " & ".join(f"${100 * R64[c][k]['rec']:.0f}$" for c in ("forzado|4", "maniobra|4", "maniobra|1"))
        filas.append(f"{et if i == 0 else ''} & ${p}$ & ${q['ratio_incr']:.3f}$ {ic(*q['ratio_incr_ic'], 3)} & "
                     f"${q['att']:.3f}$ & ${100 * q['fpr']:.1f}$ & {rr} \\\\")
for et, k in (("memoryless $\\times6$", "inst_m6_pi0.173"), ("memoryless $\\times12$", "inst_m12_pi0.173")):
    q = P64[k]
    rr = " & ".join(f"${100 * R64[c][k]['rec']:.0f}$" for c in ("forzado|4", "maniobra|4", "maniobra|1"))
    filas.append(f"{et} & $0.173$ & ${q['ratio_incr']:.3f}$ {ic(*q['ratio_incr_ic'], 3)} & ${q['att']:.3f}$ & "
                 f"${100 * q['fpr']:.1f}$ & {rr} \\\\")
escribir("tab_persistencia.tex", filas)


# ---------------------------------------------------------------- plan 15: k = 70 (E65)
e65 = J("E65_k70.json")
for c, cod in (("forzado|4", "Fz"), ("maniobra|4", "Mc"), ("maniobra|1", "Mu")):
    pct("Ks" + cod, e65["rec"][c]["k70|adaptado_cal"], 0)
ksv = [100 * e65["rec"][c]["k70|adaptado_cal"]["rec"] for c in e65["rec"]]
macro("KsAdMin", num(min(ksv), 0)); macro("KsAdMax", num(max(ksv), 0))
macro("KsSmMax", num(max(100 * e65["rec"][c]["k70|amplitud_cal"]["rec"] for c in e65["rec"]), 0))
macro("KsVentMu", num(e65["abs"]["maniobra|1"]["ventaja"]))
macro("KsOrFz", num(e65["abs"]["forzado|4"]["R_oraculo"]))
for c, cod in (("forzado|4", "Fz"), ("maniobra|4", "Mc"), ("maniobra|1", "Mu")):
    macro("KsDif" + cod, num(e65["rec"][c]["k70|adaptado_cal"]["rec"] - e65["rec"][c]["k30|adaptado_cal"]["rec"], 2, True))


# ---------------------------------------------------------------- plan 15: déficit del motor (E67)
e67 = J("E67_deficit.json")
D67 = e67["desvio"]
otras = [v for v in D67 if v not in ("base", "sin_si", "lider_azar")]
d15 = [100 * D67[v][f"{m}|15"] for v in otras for m in ("ViSt", "WiSt")]
macro("DfQuinceMin", num(min(d15), 0)); macro("DfQuinceMax", num(max(d15), 0))
macro("DfBaseViSt", num(100 * D67["base"]["ViSt|15"], 0)); macro("DfBaseWiSt", num(100 * D67["base"]["WiSt|15"], 0))


# ---------------------------------------------------------------- plan 15: fracción de informados (E68)
e68 = J("E68_fraccion_informados.json")
R68, A68 = e68["rec"], e68["abs"]
C3 = ("forzado|4", "maniobra|4", "maniobra|1")


def rango(nombre, vals, nd=0):
    macro(nombre + "Min", num(min(vals), nd)); macro(nombre + "Max", num(max(vals), nd))


def mejor_sm(c):
    return max(v["rec"] for k, v in R68[c].items() if k.startswith("sin_memoria"))


for n, cod in ((1, "Uno"), (3, "Tres"), (12, "Doce"), (25, "Veinte")):
    rango("Mf" + cod, [100 * R68[f"N250|n{n}|{c}"]["integrador"]["rec"] for c in C3])
    rango("MfSm" + cod, [100 * mejor_sm(f"N250|n{n}|{c}") for c in C3])
rango("MfK", [100 * R68[f"N1000|n48|{c}"]["integrador"]["rec"] for c in C3])
rango("MfCab", [100 * R68[f"N{N}|n{n}|{c}"]["integrador"]["rec"] for c in C3 for N, n in ((250, 12), (1000, 48))])
rango("MfSmK", [100 * R68[f"N1000|n48|{c}"]["sin_memoria"]["rec"] for c in C3])
rango("MfVentDoce", [A68[f"N250|n12|{c}"]["ventaja"] for c in C3], 2)
rango("MfVentK", [A68[f"N1000|n48|{c}"]["ventaja"] for c in C3], 2)
for c, cod in zip(C3, ("Fz", "Mc", "Mu")):
    pct("MfDoce" + cod, R68[f"N250|n12|{c}"]["integrador"], 0)
    pct("MfK" + cod, R68[f"N1000|n48|{c}"]["integrador"], 0)
    macro("MfSmUno" + cod, num(100 * mejor_sm(f"N250|n1|{c}"), 0))
    macro("MfUno" + cod, num(100 * R68[f"N250|n1|{c}"]["integrador"]["rec"], 0))
filas = []
for N, n in ((250, 1), (250, 3), (250, 12), (250, 25), (1000, 48)):
    for i, (c, nom) in enumerate(zip(C3, ("driven, $\\tauW=4$\\,s", "manoeuvre, $\\tauW=4$\\,s", "manoeuvre, $\\tauW=1$\\,s"))):
        r, a = R68[f"N{N}|n{n}|{c}"], A68[f"N{N}|n{n}|{c}"]
        sm = [k for k in r if k.startswith("sin_memoria")]
        best = max(sm, key=lambda k: r[k]["rec"])
        gtxt = "" if best == "sin_memoria" else f" ($\\times{best.split('_x')[1]}$)"
        cab = f"${N}$ & ${n}$ & " if i == 0 else " & & "
        filas.append(cab + f"{nom} & ${a['ventaja']:.3f}$ & ${r['integrador']['rec']:.2f}$ {ic(*r['integrador']['ic95'])} & "
                     f"${r['sin_memoria']['rec']:.2f}$ & " + (f"${r[best]['rec']:.2f}${gtxt}" if len(sm) > 1 else "---") + " \\\\")
escribir("tab_fraccion.tex", filas)

# ---------------------------------------------------------------- plan 15: valor de leer al líder (E69)
e69 = J("E69_dos_etapas.json")
for cel, cod in (("F4", "Fz"), ("M4", "Mc"), ("M1", "Mu")):
    H = dict(zip(e69["H"][cel]["c"], e69["H"][cel]["H"]))
    G = dict(zip(e69["g"][cel]["T"], e69["g"][cel]["g"]))
    macro("LeHq" + cod, num(100 * H[0.2], 0))
    macro("LeHd" + cod, num(100 * H[0.1], 0))
    macro("LeHdRaz" + cod, num(H[0.1] / 0.1, 1))
    macro("LeGu" + cod, num(100 * G[1.0], 0))
    macro("LeGc" + cod, num(100 * G[4.0], 0))
P69 = e69["pred"]
macro("LeRepFrac", num(100 * P69["L2"]["frac_dentro"], 0))
macro("LeRepErr", num(P69["L2"]["error_mediano"]))
macro("LeLinFrac", num(100 * P69["L1"]["frac_dentro"], 0))
macro("LeTresFrac", num(100 * P69["L3"]["frac_dentro"], 0))
macro("LeNcel", str(P69["L2"]["n"]))
_e44 = J("E44_lectura.json")["wL_curso"]
_wl = [v for k in ("tractor|250|4", "pulso|250|4") for v in _e44[k][1:6]]   # tramos de 0,5 s entre 0,5 y 3 s
macro("LePostMin", num(min(_wl), 1)); macro("LePostMax", num(max(_wl), 1))
er_sm = [v["replica"]["rec"] - v["medido"]["rec"] for k, v in e69["celdas"].items() if k.split("|")[1].startswith("inst")]
macro("LeRepSm", num(sorted(er_sm)[len(er_sm) // 2], 2, True))
filas = []
for cel, nom in (("F4", "driven, $\\tauW=4$\\,s"), ("M4", "manoeuvre, $\\tauW=4$\\,s"), ("M1", "manoeuvre, $\\tauW=1$\\,s")):
    G = dict(zip(e69["g"][cel]["T"], e69["g"][cel]["g"]))
    H = dict(zip(e69["H"][cel]["c"], e69["H"][cel]["H"]))
    filas.append(nom + " & " + " & ".join(f"${G[t]:.2f}$" for t in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)) + " & "
                 + " & ".join(f"${H[c]:.2f}$" for c in (0.05, 0.1, 0.2, 0.35, 0.5)) + " \\\\")
escribir("tab_lectura.tex", filas)
for k, cod in (("L1", "Lin"), ("L3", "Tres")):
    macro("LeErr" + cod, num(P69[k]["error_mediano"]))
macro("LeCerFrac", num(100 * P69["L4"]["frac_dentro"], 0)); macro("LeCerRho", num(P69["L4"]["spearman"]))
er_ad = [v["replica"]["rec"] - v["medido"]["rec"] for k, v in e69["celdas"].items()
         if not k.split("|")[1].startswith("inst") and k.split("|")[1] not in ("ciclo1_n3", "ciclo1_n6")]
macro("LeRepAd", num(sorted(er_ad)[len(er_ad) // 2], 2, True))
macro("LeRepAdFrac", num(100 * sum(abs(x) <= 0.10 for x in er_ad) / len(er_ad), 0))

# ---------------------------------------------------------------- control del auditor externo (prioridad 3)
_p3 = os.path.join(RAIZ, "control_auditor", "resultados", "RESULTADOS_MEMORIA_GANANCIA.json")
with open(_p3) as fh:
    p3 = json.load(fh)
for x in p3["conditions"]:
    if x["tauW"] == 4.0 and x["variant"] in ("full_t0_joint", "full_t1_joint"):
        cod = ("AuMem" if x["variant"] == "full_t1_joint" else "AuIns") + ("F" if x["protocol"] == "forzado" else "M")
        macro(cod, num(100 * x["recovery"], 0))
for x in p3["primary_contrasts"]:
    if x["protocol"] == "maniobra" and x["scheme"] == "fpr":
        macro("AuFprDifM", num(x["difference_R"], 3, True))
        macro("AuFprDifMCI", ic(*x["simultaneous_CI95"], 3, True))


# ---------------------------------------------------------------- plan 15: barrido de la ventana con 16 réplicas (E66)
_NOMV = {"0.25": "Cuarto", "0.5": "Medio", "1": "Uno", "2": "Dos", "4": "Cuatro"}
for _etq, _cod in (("forzado", "Fz"), ("maniobra", "Mc")):
    for _v, _nv in _NOMV.items():
        macro("Vb" + _cod + _nv, num(100 * _C66[f"barrido|base|N250|{_etq}"]["n16"]["rec"][_v]["rec"], 0))
    macro("VbMejor" + _cod, num(float(_C66[f"barrido|base|N250|{_etq}"]["n16"]["mejor"]), 0))
    macro("VbMejorK" + _cod, num(float(_C66[f"barrido|base|N1000|{_etq}"]["n16"]["mejor"]), 0))
_P66 = e66["pred"]["V1"]
macro("PfAgMaxM", num(e61["mejor"]["maniobra|a5"]["ventana"], 0))
_bs = [max(c["n16"]["boot_mejor"].values()) for c in _C66.values() if "boot_mejor" in c.get("n16", {})]
macro("VbBootMin", num(100 * min(_bs), 0)); macro("VbBootCien", str(sum(b >= 0.995 for b in _bs))); macro("VbBootTotal", str(len(_bs)))
macro("VbIguales", str(_P66["iguales"])); macro("VbAdy", str(_P66["adyacentes"])); macro("VbTotal", str(_P66["total"]))

# ---------------------------------------------------------------- plan 16: peces cebra con informados entrenados (E73, E73b)
e73, e73b = J("E73_peces_entrenados.json"), J("E73b_variantes.json")
macro("ZfNtr", str(e73["n"]["ensayos"])); macro("ZfNgr", str(len(e73["n"]["grupos"])))
macro("ZfSig", num(e73["n"]["sigma_mediana"], 1))
for _c, _n in (("1.0|1", "UnoUno"), ("cuadro|1", "CuUno"), ("1.0|0", "UnoCero"), ("0.5|0.5", "MedMed"), ("1.0|3", "UnoTres")):
    macro("ZfA" + _n, num(e73["auc"][_c]))
    macro("ZfA" + _n + "CI", ic(*e73["ic"][_c]))
_d = e73["dif_1s_cuadro"]["1"]
macro("ZfGan", num(_d["dif"], 3, True)); macro("ZfGanCI", ic(*_d["ic95"], 3, True))
_q = e73["auc_tD"]
macro("ZfTdUno", num(_q["1.0|tD"]["auc"])); macro("ZfTdUnoCI", ic(*_q["1.0|tD"]["ic95"]))
macro("ZfTdCu", num(_q["cuadro|tD"]["auc"])); macro("ZfTdCuCI", ic(*_q["cuadro|tD"]["ic95"]))
_d = e73["posterior_dif_tD"]
macro("ZfGanTd", num(_d["dif"], 3, True)); macro("ZfGanTdCI", ic(*_d["ic95"], 3, True)); macro("ZfNtd", str(_d["n_ensayos"]))
_c = e73["control_intertrial"]
_vi = [q["auc"] for k, q in _c.items() if k.startswith("1.0|")]
macro("ZfIntMin", num(min(_vi))); macro("ZfIntMax", num(max(_vi)))
_r = e73["resp"]
macro("ZfTE", num(_r["tE_mediana"])); macro("ZfTD", num(_r["tD_mediana"]))
macro("ZfTDlo", num(_r["tD_iqr"][0])); macro("ZfTDhi", num(_r["tD_iqr"][1]))
macro("ZfPrim", num(100 * _r["E_antes_D"], 0))
macro("ZfGrSobre", str(e73["pred"]["Z1"]["grupos_sobre_0.5"]))
_z6 = e73["Z6"]
macro("ZfRho", num(_z6["rho_1s"], 2, True)); macro("ZfRhoCI", ic(*_z6["ic_1s"], 2, True))
_v = e73b["rapidez"]
macro("ZfVelPre", num(_v["auc"]["1.0|0"])); macro("ZfVelUno", num(_v["auc"]["1.0|1"]))
macro("ZfVelTd", num(_v["auc_tD"]["1.0|tD"])); macro("ZfVelCuMed", num(_v["auc"]["cuadro|0.5"]))
macro("ZfCicGan", num(e73b["ciclo"]["dif_1"]["dif"], 3, True)); macro("ZfCicGanCI", ic(*e73b["ciclo"]["dif_1"]["ic95"], 3, True))
macro("ZfCicGanTd", num(e73b["ciclo"]["dif_tD"]["dif"], 3, True)); macro("ZfCicGanTdCI", ic(*e73b["ciclo"]["dif_tD"]["ic95"], 3, True))
macro("ZfCincoGan", num(e73b["k5"]["dif_1"]["dif"], 3, True))

_filas = []
for _w, _et in (("cuadro", "one frame"), ("0.25", "$0.25$\\,s"), ("0.5", "$0.5$\\,s"), ("1.0", "$1$\\,s"), ("2.0", "$2$\\,s")):
    _filas.append(_et + " & " + " & ".join(f"${e73['auc'][f'{_w}|{_t}']:.2f}$" for _t in ("-1", "0", "0.5", "1", "1.5", "2", "3"))
                  + f" & ${e73['auc_tD'][f'{_w}|tD']['auc']:.2f}$ \\\\")
escribir("tab_entrenados.tex", _filas)

# ---------------------------------------------------------------- plan 16: la cabecera con otro ruido, agilidad y k (E70)
e70 = J("E70_cabecera_robustez.json")
_R70 = e70["rec"]
_C3 = ("forzado|4", "maniobra|4", "maniobra|1")
def _ri(v):
    return [100 * _R70[f"{v}|{c}"]["integrador"]["rec"] for c in _C3]
def _rsm(v):
    return [100 * max(_R70[f"{v}|{c}"][k]["rec"] for k in ("sin_memoria", "sin_memoria_x6", "sin_memoria_x12")) for c in _C3]
for _v, _cod in (("f0.5", "RuMed"), ("f2", "RuDos"), ("a5", "AgCinco"), ("a20", "AgVeinte"), ("k70", "Ks")):
    rango("Rb" + _cod, _ri(_v))
macro("RbAgCuarentaFz", num(_ri("a40")[0], 0))
rango("RbAgCuarentaM", _ri("a40")[1:])
macro("RbSmCuarentaFz", num(_rsm("a40")[0], 0))
_g2 = e70["pred"]["G2"]["pares"]
rango("RbColDos", [-100 * _g2[c][0] for c in _C3])
rango("RbColSeis", [100 * _g2[c][1] for c in _C3])
macro("RbMinResto", num(min(min(_ri(v)) for v in ("f0.5", "f2", "a5", "a20", "k70")), 0))
macro("RbPierde", str(len(e70["pred"]["G3"]["pierde"])))
_om = e70["omega_sobre_sigma"]
_filas = []
for _v, _et in (("ref", "reference"), ("f0.5", "noise $\\times0.5$"), ("f2", "noise $\\times2$"), ("a5", "$a_{\\max}=5$"),
                ("a20", "$a_{\\max}=20$"), ("a40", "$a_{\\max}=40$"), ("k70", "$k=70$")):
    _filas.append(f"{_et} & ${_om[_v]:.2f}$ & " + " & ".join(f"${x / 100:.2f}$" for x in _ri(_v)) + " & "
                  + " & ".join(f"${x / 100:.2f}$" for x in _rsm(_v)) + " \\\\")
escribir("tab_robustez.tex", _filas)

# ---------------------------------------------------------------- plan 16: comparación con semillas nuevas y ventana (E71)
e71 = J("E71_comparacion_justa.json")
_D71 = e71["dif"]
_tres = [c for c in _D71 if not c.startswith("N250|n1|")]
_gan = [100 * _D71[c]["int1_vs_gstar"]["dif"] for c in _tres if _D71[c]["int1_vs_gstar"]["ic95"][0] > 0]
macro("JuNgana", str(len(_gan))); macro("JuNtres", str(len(_tres)))
rango("JuGan", _gan)
_q = _D71["N250|n3|forzado|4"]["int1_vs_gstar"]
macro("JuTresFz", num(100 * _q["dif"], 1, True)); macro("JuTresFzCI", ic(*_q["ic95"], 1, True, k=100))
rango("JuUno", [-100 * _D71[f"N250|n1|{c}"]["int1_vs_gstar"]["dif"] for c in ("forzado|4", "maniobra|4")])
rango("JuK", [100 * v["int_vs_gmax"]["dif"] for v in e71["N1000"].values()])
_P71 = e71["prueba"]
rango("JuPrDoce", [100 * _P71[f"N250|n12|{c}"]["integrador"]["rec"] for c in ("forzado|4", "maniobra|4", "maniobra|1")])
_R4 = e71["R4"]
_v = lambda c, n: 100 * _R4[c]["rec"][n]["rec"]
rango("VrCorto", [_v(c, "integrador_0.1s") for c in _R4])
rango("VrLargo", [_v(c, "integrador_8s") for c in _R4])
rango("VrUno", [_v(c, "integrador") for c in _R4])
macro("VrRat", num(100 * min(e71["pred"]["W2"]["cociente"].values()), 1))
_filas = []
for c, et in (("forzado|4", "driven, $\\tauW=4$\\,s"), ("maniobra|4", "manoeuvre, $\\tauW=4$\\,s"), ("maniobra|1", "manoeuvre, $\\tauW=1$\\,s")):
    _filas.append(et + " & " + " & ".join(f"${_R4[c]['rec'][n]['rec']:.2f}$" for n in
                  ("integrador_0.1s", "integrador_0.25s", "integrador_0.5s", "integrador", "integrador_2s", "integrador_4s", "integrador_8s")) + " \\\\")
escribir("tab_ventana_cabecera.tex", _filas)
_filas = []
for c in _D71:
    N_, n_, etq, tw = c.split("|")
    et = {"forzado": "driven", "maniobra": "manoeuvre"}[etq] + f", $\\tauW={tw}$\\,s"
    q = _D71[c]["int1_vs_gstar"]
    el = e71["eleccion"][c]
    vw = "1" if el["ventana"] == "integrador" else el["ventana"].split("_")[1][:-1]
    g = "1" if el["ganancia"] == "sin_memoria" else el["ganancia"].split("_x")[1]
    _filas.append(f"{n_[1:]} & {et} & ${vw}$ & $\\times{g}$ & ${_P71[c]['integrador']['rec']:.2f}$ & "
                  f"${_P71[c][el['ventana']]['rec']:.2f}$ & ${_P71[c][el['ganancia']]['rec']:.2f}$ & "
                  f"${q['dif']:+.3f}$ [${q['ic95'][0]:+.3f}$, ${q['ic95'][1]:+.3f}$] \\\\")
escribir("tab_justa.tex", _filas)

# ---------------------------------------------------------------- plan 16: persistencia con falsas alarmas (E72)
e72 = J("E72_persistencia_fa.json")
_P72, _C72 = e72["P"], e72["c_P"]
def _fa(nom, cod, mult=1, nd=3):
    q = _P72[f"x{mult}|{nom}"]
    macro(cod, num(q["r"], nd)); macro(cod + "CI", ic(*q["r_ic"], nd))
_fa("oraculo", "FaOr"); _fa("adaptado_pi0.173", "FaIn"); _fa("adaptado_pi0.05", "FaInCinco")
_fa("CoLd", "FaCold", nd=2); _fa("ViSt", "FaVist", nd=2); _fa("amplitud_pi0.173", "FaSmUno", nd=2)
_q = _P72["x1|adaptado_pi0.173"]
macro("FaInRat", num(_q["P_sobre_Por"], 1)); macro("FaInRatCI", ic(*_q["P_sobre_Por_ic"], 1))
rango("FaSmAmp", [100 * _P72[f"x1|{n}"]["r"] for n in ("inst_m6_pi0.173", "inst_m12_pi0.173")])
macro("FaCP", num(_C72["x1|adaptado_pi0.173"]["c_P"])); macro("FaCPCinco", num(_C72["x1|adaptado_pi0.05"]["c_P"]))
macro("FaCPSm", num(_C72["x1|inst_m12_pi0.173"]["c_P"]))
macro("FaN", num(_P72["x1|oraculo"]["n_fp"], 0))
macro("FaDiezOr", num(_P72["x10|oraculo"]["r"], 3)); macro("FaDiezIn", num(_P72["x10|adaptado_pi0.173"]["r"], 3))
macro("FaDiezSm", num(_P72["x10|amplitud_pi0.173"]["r"], 2))
_filas = []
for n, et in (("ViSt", "ViSt (no reading)"), ("CoLd", "CoLd (nobody reads)"), ("oraculo", "oracle"),
              ("adaptado_pi0.173", "integrating, $\\pi_0=0.173$"), ("adaptado_pi0.05", "integrating, $\\pi_0=0.05$"),
              ("amplitud_pi0.173", "memoryless $\\times1$"), ("inst_m6_pi0.173", "memoryless $\\times6$"),
              ("inst_m12_pi0.173", "memoryless $\\times12$")):
    q = _P72[f"x1|{n}"]
    d = _P72.get(f"x10|{n}")
    cp = _C72.get(f"x1|{n}", {}).get("c_P")
    _filas.append(f"{et} & ${q['r']:.3f}$ [${q['r_ic'][0]:.3f}$, ${q['r_ic'][1]:.3f}$] & ${q['P_sobre_Por']:.1f}$ & "
                  + (f"${cp:.2f}$" if cp is not None and n not in ("oraculo",) else ("$1$" if n == "oraculo" else "---"))
                  + " & " + (f"${d['r']:.3f}$" if d else "---") + f" & ${q['pol']:.2f}$ \\\\")
escribir("tab_persistencia_fa.tex", _filas)

# ---------------------------------------------------------------- plan 16: ventana corta con agilidad alta (E70b)
e70b = J("E70b_ventana_agilidad.json")
_R7b = e70b["rec"]
macro("RbCortaCuarFz", num(100 * _R7b["a40|forzado|4"]["q"]["integrador_0.25s"]["rec"], 0))
macro("RbCortaVeinFz", num(100 * _R7b["a20|forzado|4"]["q"]["integrador_0.5s"]["rec"], 0))
rango("RbCorta", [100 * v["q"][v["corta"]]["rec"] for v in _R7b.values()])
macro("RbCortaGana", str(e70b["pred"]["A1"]["gana_al_sin_memoria"]))
rango("RbCuar", [100 * _R7b[f"a40|{c}"]["q"]["integrador_0.25s"]["rec"] for c in ("forzado|4", "maniobra|4", "maniobra|1")])

with open(os.path.join(AQUI, "numeros.tex"), "w") as fh:
    fh.write("% generado por numeros.py desde resultados/*.json; no editar a mano\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"{len(M)} macros en numeros.tex; tablas en gen/: {sorted(os.listdir(GEN))}")
