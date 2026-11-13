"""Etapa 53 (A3 y A4 del plan de fortalecimiento).

A3. Agilidad de la grajilla, omega_max = a_perp / v (percentil 90 por iniciadora, mediana
    sobre iniciadoras), con una sola selección de eventos (los 12 con al menos 12 aves) y su
    intervalo por remuestreo de eventos. Sensibilidad: los 21 eventos.
A4. Tabla de calibración del archivo continuo (Nat. Commun. 10:5174, Supp. Data 2-4) con el
    procedimiento de E9 (cuaderno Etapa9, celda 3): (a) los 18 bloques, como se publicó
    (ancla); (b) sin el bloque duplicado; (c) además sin los cuadros de mobbing que
    pertenecen a los ocho eventos curados (la versión independiente de esos eventos).

    PYTHONPATH=codigo python codigo/e53_agilidad_calibracion.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

import turns
import jackdaws3d as j3
from e16_helpers import cinematica

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 10000


def agilidad(min_aves, rng):
    por_ev = []
    for ctx, ev in turns.eventos():
        e = turns.cargar(ctx, ev)
        if len(e["ids"]) < min_aves:
            continue
        c = cinematica(e)
        if len(c) >= 3:
            por_ev.append(np.array([x["omega_max"] for x in c]))
    todo = np.concatenate(por_ev)
    b = rng.integers(0, len(por_ev), (N_BOOT, len(por_ev)))
    med = [np.median(np.concatenate([por_ev[i] for i in fila])) for fila in b]
    return {"omega_max": float(np.median(todo)), "ic95": [float(np.percentile(med, 2.5)),
                                                         float(np.percentile(med, 97.5))],
            "n_eventos": len(por_ev), "n_iniciadoras": int(len(todo)),
            "por_evento": [float(np.median(v)) for v in por_ev]}


def intervalos_curados(bloques):
    """Intervalo de tiempo, dentro de cada bloque continuo de mobbing, que ocupa cada evento
    curado de mobbing (emparejado por ternas de velocidad redondeadas a 0,01 m/s)."""
    out = {}
    for ctx, ev in turns.eventos("mobbing"):
        t = turns.cargar(ctx, ev)["tracks"]
        tern = {tuple(r) for r in np.round(t[:, 5:8], 2).tolist()}
        mejor, n_mejor, ts = None, 0, None
        for b in bloques:
            if b["ctx"] != "mobbing":
                continue
            m = np.array([tuple(r) in tern for r in np.round(b["d"][:, 5:8], 2).tolist()])
            if m.sum() > n_mejor:
                mejor, n_mejor, ts = b["nombre"], int(m.sum()), b["d"][m, 4]
        if mejor and n_mejor > 0.5 * len(tern):
            lo, hi = np.percentile(ts, [0.5, 99.5])
            out.setdefault(mejor, []).append((float(lo), float(hi), f"mobbing {ev}"))
    return out


def tabla(bloques, quitar=None):
    res = {}
    for modo in ("horizontal", "3d"):
        por = {"transit": [], "mobbing": []}
        for b in bloques:
            bb = b
            if quitar and b["nombre"] in quitar:
                d = b["d"]
                fuera = np.zeros(len(d), bool)
                for lo, hi, _ in quitar[b["nombre"]]:
                    fuera |= (d[:, 4] >= lo) & (d[:, 4] <= hi)
                bb = {**b, "d": d[~fuera]}
            e = j3.residuos(bb, modo=modo)
            e = e[np.isfinite(e)]
            if e.size < 500:
                continue
            mad = float(np.median(np.abs(e - np.median(e))) * 1.4826)
            por[b["ctx"]].append({"nombre": b["nombre"], "mad": mad, "n": int(e.size), "z": e / mad})
        for ctx in por:
            v = por[ctx]
            m = np.array([x["mad"] for x in v])
            nu, _, _ = stats.t.fit(np.concatenate([x["z"] for x in v]), floc=0.0)
            res[f"{modo}|{ctx}"] = {"bandadas": len(v), "sigma": float(np.median(m)),
                                    "rango": [float(m.min()), float(m.max())], "nu": float(nu),
                                    "nombres": [x["nombre"] for x in v]}
    return res


if __name__ == "__main__":
    os.chdir(RAIZ)
    rng = np.random.default_rng(53)
    L, out = ["=== E53 · agilidad de la grajilla y tabla de calibración ==="], {}
    P = L.append

    P("\n--- A3: omega_max = a_perp / v ---")
    for nom, m in (("12 eventos con al menos 12 aves", 12), ("todos los eventos", 0)):
        r = agilidad(m, rng)
        out[f"agilidad_min{m}"] = r
        P(f'{nom}: {r["omega_max"]:.3f} rad/s, IC 95 % [{r["ic95"][0]:.3f}; {r["ic95"][1]:.3f}] '
          f'({r["n_eventos"]} eventos con 3 o más iniciadoras, {r["n_iniciadoras"]} iniciadoras)')

    P("\n--- A4: calibración desde el archivo continuo ---")
    b18 = j3.bandadas(unicas=False)
    b17 = j3.bandadas(unicas=True)
    quitar = intervalos_curados(b17)
    out["intervalos_curados"] = quitar
    for nom, bl in quitar.items():
        P(f"   {nom}: se excluyen " + "; ".join(f"{ev} [{lo:.2f}; {hi:.2f}] s" for lo, hi, ev in bl))
    for nom, bloques, q in (("18 bloques (como en E9)", b18, None), ("17 bandadas distintas", b17, None),
                            ("17 bandadas, sin los cuadros de los eventos curados", b17, quitar)):
        t = tabla(bloques, q)
        out[nom] = t
        P(f"\n{nom}:")
        for k, v in t.items():
            P(f'   {k:20s} {v["bandadas"]:2d} bandadas  sigma_theta {v["sigma"]:.4f} rad/s '
              f'[{v["rango"][0]:.3f}; {v["rango"][1]:.3f}]  nu {v["nu"]:.2f}')
        h, d3 = t["horizontal|transit"]["sigma"], t["3d|transit"]["sigma"]
        hm, dm = t["horizontal|mobbing"]["sigma"], t["3d|mobbing"]["sigma"]
        P(f'   factor entre contextos (2D): {hm / h:.2f}; exceso de la proyección horizontal: '
          f'{100 * (h / d3 - 1):.0f} % (tránsito), {100 * (hm / dm - 1):.0f} % (mobbing)')
        out[nom]["factor_contexto_2d"] = hm / h
        out[nom]["exceso_2d"] = {"transit": h / d3 - 1, "mobbing": hm / dm - 1}
    e9 = json.load(open("resultados/E9_validacion.json"))["sigma"]
    t = out["18 bloques (como en E9)"]
    dm = max(abs(t["horizontal|transit"]["sigma"] - e9["tr_2d"]), abs(t["horizontal|mobbing"]["sigma"] - e9["mo_2d"]),
             abs(t["3d|transit"]["sigma"] - e9["tr_3d"]), abs(t["3d|mobbing"]["sigma"] - e9["mo_3d"]))
    out["ancla_E9"] = {"delta_max": dm, "cumple": bool(dm < 1e-9)}
    P(f"\nancla: los 18 bloques frente a E9_validacion.json: |Δ| máx = {dm:.2e}")
    with open("resultados/E53_agilidad_calibracion.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
    with open("resultados/E53_agilidad_calibracion.txt", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
