"""Instrumentación de la Etapa 15: ¿es el defecto de un modelo o de una clase?

La tesis del proyecto ---que el «estado informado» de los modelos de liderazgo no
tiene manifestación observable persistente, y que por tanto el supuesto de que los
vecinos lo leen es de comunicación y no de percepción--- se midió sobre WiSt. Si
vale sólo ahí, es una crítica a un artículo. Si vale también en **CoLd** (Couzin
2005, el modelo de individuos informados más citado del área), es una propiedad
estructural de toda la familia.

`huella` mide cuánto dura la firma cinemática de una decisión en cualquiera de los
modelos, con la misma definición operativa para todos.
"""

from __future__ import annotations

import numpy as np

from flocking import Params, Flock


def huella(p: Params, model: str, t_burn=6.0, t_obs=5.0, agente=7, ang=2.0,
           umbral_rel=1.5, n_rep=8):
    """Cuánto dura la firma cinemática de una decisión, y con qué amplitud.

    A un agente se le fija una intención sostenida y se sigue su residuo
    `|eps| = |omega_j - <omega>_vecinos|`. La **duración de la huella** es el tiempo
    hasta que ese residuo cae por debajo de `umbral_rel` veces la anchura de la línea
    base y ya no vuelve a subir de forma sostenida.

    Es la misma magnitud que E4 midió sobre WiSt, aquí aplicable a cualquier modelo
    para poder comparar la familia entera con un solo criterio.
    """
    dur, pico, series, sigs = [], [], [], []
    for rep in range(n_rep):
        q = Params(**{**p.__dict__, "seed": p.seed + rep})
        f = Flock(q, model)
        for _ in range(int(round(t_burn / q.dt))):
            f.step()

        # anchura de la línea base en esta misma corrida
        base = []
        for _ in range(int(round(2.0 / q.dt))):
            f.step()
            i, _ = f._neighbors()
            base.append(np.abs(f.omega - f.omega[i].mean(axis=1)))
        base = np.concatenate(base)
        sig = float(np.median(base) * 1.4826)
        thr = umbral_rel * max(sig, 1e-9)
        sigs.append(sig)

        f.will[agente] = 1.0
        f.d_des[agente] = np.array([np.cos(ang), np.sin(ang)])
        serie = []
        for s in range(int(round(t_obs / q.dt))):
            f.step()
            f.will[agente] = 1.0                 # intención SOSTENIDA
            if s % 5 == 0:
                i, _ = f._neighbors()
                e = np.abs(f.omega - f.omega[i].mean(axis=1))
                serie.append((s * q.dt, float(e[agente])))
        serie = np.array(serie)
        series.append(serie)

        # Duración: PRIMER descenso sostenido bajo el umbral después del pico.
        # Tomar el último cruce sería incorrecto: el residuo fluctúa alrededor de la
        # línea base y roza el umbral por azar durante toda la ventana, con lo que
        # cualquier huella parecería durar lo que dure la observación.
        w = max(3, len(serie) // 25)
        sm = np.convolve(serie[:, 1], np.ones(w) / w, mode="same")
        i_pico = int(np.argmax(sm))
        d = float(serie[-1, 0])
        for j in range(i_pico, len(sm)):
            if sm[j] < thr and np.all(sm[j:j + w] < thr):   # y no vuelve a subir
                d = float(serie[j, 0])
                break
        dur.append(d)
        pico.append(float(sm.max() / max(sig, 1e-9)))
    return {"model": model, "dur": float(np.median(dur)),
            "dur_sd": float(np.std(dur)), "pico_rel": float(np.median(pico)),
            "tau_c": p.tau_c, "series": series[0],
            "sigma_base": float(np.median(sigs))}
