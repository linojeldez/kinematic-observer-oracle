"""Parámetros de referencia del observador y mecanismos comparados.

`P0` es la configuración de referencia de todas las etapas (N = 250, k = 15, ruido
sigma_xi = sqrt(2), parámetros del observador calibrados en grajillas). `MECANISMOS` son los
cinco brazos que comparan las etapas E42 a E44: el modelo sin voluntad (ViSt), la amplitud
del giro, el observador adaptado de 1 s, el canal explícito y el oráculo.
"""

import numpy as np

from flocking import Params

CFG = dict(sigma_theta=0.361, nu=2.8, sigma_W=0.811, pi0=0.173, tau_obs=0.083)
P0 = Params(N=250, k=15, sigma_xi=np.sqrt(2.0), seed=11, **CFG)

MECANISMOS = [
    ("ViSt", "ViSt", dict()),
    ("amplitud", "WiSt", dict(will_mode="bayes")),
    ("adaptado", "WiSt", dict(will_mode="bayes", detector="ciclo", tau_det=1.0)),
    ("canal", "WiSt", dict(will_mode="bayes", call_on=True, tau_call=4.0)),
    ("oraculo", "WiSt", dict(will_mode="oracle")),
]
