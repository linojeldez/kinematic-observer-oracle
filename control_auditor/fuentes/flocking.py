"""
Motor de bandadas 2D para replicar y extender

    Balázs, Vásárhelyi & Vicsek (2020),
    "Adaptive leadership overcomes persistence-responsivity trade-off in flocking",
    J. R. Soc. Interface 17: 20190853.

Modelos implementados (Etapa 0):
    ViSt  - Vicsek estándar (promedio local de velocidades)
    CoLd  - Couzin leader   (will propio, los vecinos no lo perciben)
    CaSp  - inertial spin model de Cavagna
    WiSt  - will recognition sobre ViSt/CoLd  [nuevo en el paper]
    WiSp  - will recognition + spin           [nuevo en el paper]

Ganchos ya previstos para las etapas siguientes:
    - `will_mode`      -> 'oracle' (paper) | 'bayes' (M2, BayWi)
    - `beta`           -> atención softmax  (M3, AttWi)
    - `will_channel`   -> canal ruidoso     (M1)

Todo el estado es NumPy vectorizado; los vecinos topológicos se recalculan
cada paso con cKDTree. Coste ~O(N log N) por paso.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------


@dataclass
class Params:
    # --- bandada
    N: int = 250                # número de agentes
    k: int = 15                 # vecinos topológicos (sin contar el propio agente)
    v0: float = 8.0             # velocidad de vuelo (m/s)
    d0: float = 15.0            # separación inicial media (m)

    # --- dinámica del agente
    # Tabla S1 del suplementario del paper: hay DOS juegos de parámetros.
    #   'Purified' (figs. 2, 3, 4):  tau_c = 0.2 s, a_max = 10 m/s^2, sigma^2 = 1 (figs 2-3) o 2 (fig 4)
    #   'Realistic' (fig. 5, arena): tau_c = 1.0 s, a_max =  6 m/s^2, sigma^2 = 0.2
    # Los tests unitarios de responsividad y persistencia usan los PURIFIED.
    tau_c: float = 0.2          # tiempo de convergencia a la velocidad deseada (s)  [tabla S1]
    a_max: float = 10.0         # aceleración máxima (m/s^2)                         [tabla S1]
    sigma_xi: float = 1.0       # sqrt de la varianza del ruido (m/s^2), sigma^2 en m^2/s^4
    # 'wiener' escala el ruido como sigma/sqrt(dt) -> resultado independiente de dt,
    # y con sigma^2 = 1-2 m^2/s^4 deja la bandada libre en polarización ~0.99, que es
    # el régimen declarado en el paper ("approx. 0.98"). 'paper' aplica la fórmula
    # literal (ruido por paso), que a dt=0.01 resulta 10x más débil: pol ~ 1.000.
    noise_mode: str = "wiener"
    # Forma de la distribución del ruido. El paper usa gaussiana; los residuos de
    # grajillas reales ajustan una t de Student con nu = 2.37 (E4), donde ni
    # siquiera existe el cuarto momento. 'student' permite un ruido intermitente,
    # que es lo que produce el aire real: rachas, no agitación uniforme.
    noise_dist: str = "gauss"   # 'gauss' | 'student'
    noise_nu: float = 3.0       # grados de libertad si noise_dist='student' (>2)

    # --- modelo de spin (CaSp / WiSp)
    J: float = 1400.0           # acoplamiento spin <- desalineación
    tau_s: float = 0.8          # tiempo de vida del spin (s)

    # --- integración
    dt: float = 0.01
    seed: int = 0

    # --- viento turbulento (propuesta M20)
    wind_sigma: float = 0.0     # intensidad del campo de viento (m/s^2). 0 = sin viento
    wind_ell: float = 200.0     # longitud de correlación espacial (m)
    wind_tau: float = 2.0       # tiempo de correlación (s)
    wind_modes: int = 8         # nº de modos de Fourier del campo
    air_frame: bool = False     # alinearse en el marco del AIRE (rechazo de modo común)
    # Intermitencia. Una suma de modos de Fourier con amplitudes gaussianas es
    # gaussiana por construcción, y no reproduce las colas de los residuos reales
    # (comprobado). La turbulencia de verdad es intermitente: largos ratos de calma
    # y rachas raras y violentas. Se modela con un factor de escala global de cola
    # pesada, renovado con el tiempo de correlación del campo.
    wind_intermittent: bool = False
    wind_nu: float = 2.5        # grados de libertad de la intermitencia (>2)

    # --- reconocimiento de voluntad
    will_mode: str = "oracle"   # 'oracle' (paper) | 'bayes' (M2, BayWi)
    beta: float | None = None   # None = suma sin normalizar (paper); float = softmax
    # Qué usa cada mecanismo como saliencia (eje de la taxonomía de M9):
    #   'will'        -> w_j, leído (paper) o inferido (BayWi)
    #   'visual'      -> velocidad angular aparente del vecino  (AMS 2024)
    #   'visual_step' -> la misma, umbralizada en s_thresh      (MST 2025)
    # Las dos visuales exigen beta finito: son reglas de ponderación, no de
    # voluntad añadida, así que la suma no acotada no les aplica.
    salience: str = "will"
    s_thresh: float = 0.5       # umbral de 'visual_step', en unidades de v0/d0
    s_sat: float = 1.0          # saturación perceptual de la saliencia visual (Weber)
    r_body: float = 0.5         # radio corporal (m): dos aves no ocupan el mismo punto
    tau_W: float = np.inf       # vida media del will (inf = no decae, como en los tests unitarios)

    # --- canal de voluntad ruidoso (propuesta M1)
    # El paper concede a los vecinos un oráculo perfecto: leen w_j sin error, sin
    # coste y sin alcance. Estos tres parámetros lo rompen.
    lambda_fp: float = 0.0      # tasa de falsas alarmas por agente (s^-1); 8.42e-5 medida en peces
    p_loss: float = 0.0         # probabilidad de que un vecino NO reciba la señal
    R_c: float = np.inf         # alcance del canal (m); inf = sin límite
    c_signal: float = np.inf    # velocidad de propagación (m/s); 343 = sonido en aire

    # --- canal acústico (propuesta M19)
    # E4 midió el hueco que este canal debe cubrir: la huella cinemática de una
    # decisión dura ~1 s y el estado dura tau_W. La llamada NO es direccional
    # (Liao 2024, Dibnah 2022): transporta un bit de alerta que SOSTIENE la
    # señal, y el movimiento sigue resolviendo hacia dónde.
    call_on: bool = False       # activar M19
    tau_call: float = 4.0       # duración de una llamada (s) -- la variable de interés
    R_call: float = np.inf      # alcance acústico (m)
    p_call_det: float = 0.9     # prob. de llamar estando informado          (p_d)
    p_call_fa: float = 0.02     # prob. de llamar SIN estarlo                (p_fa)
    p_hear: float = 1.0         # prob. de oír una llamada dentro del alcance

    # --- detector bayesiano (propuesta M2, BayWi)
    # Valores por defecto medidos en E2 sobre grajillas reales, NO ajustados.
    sigma_theta: float = 0.356  # ruido angular robusto de la línea base (rad/s)  [E2]
    nu: float = 2.8             # grados de libertad de la t   [E4, ajuste ML tipificado por evento]
    sigma_W: float = 0.81       # escala angular de una maniobra intencional (rad/s)  [E4]
    pi0: float = 0.173          # prior de intención           [E2, fracción de iniciadores]
    tau_obs: float = 0.10       # ventana de observación de la velocidad angular (s)  [E4]
    bayes_lik: str = "student"  # 'student' (colas pesadas) | 'gauss' (control)
    # --- estadístico del detector (E18)
    # 'amplitud' es el de E4: log-cociente instantáneo sobre |eps|, o sea la
    # MAGNITUD de la desviación. 'ciclo' es el de Attanasi et al. 2015: la
    # FRACCIÓN DE TIEMPO desalineado en una ventana. Ellos muestran, con control
    # negativo explícito (su ec. S3), que la amplitud NO discrimina iniciadores y
    # la persistencia sí. Cambiarlo es la prueba de si el 3-9 % de E4 es una
    # propiedad de la señal o del estadístico elegido.
    detector: str = "amplitud"  # 'amplitud' | 'ciclo' | 'secuencial'
    # --- normalización divisiva con memoria (E26; Fahimipour et al. 2023, ec. 4)
    # Port adimensional de la forma reducida: la entrada agregada de alarma que
    # recibe cada agente alimenta una memoria exponencial y la ganancia divide lo
    # que perciba después. Sus unidades (deg/s) no son portables; su ESTRUCTURA y
    # su tau calibrado (0.1 s, Table S3) sí. gamma_gain grande = normalización
    # débil (gamma->inf equivale a gain_on=False).
    gain_on: bool = False
    tau_gain: float = 0.1       # memoria de la ganancia (s)  [Table S3: 0.1]
    gamma_gain: float = 1.0     # constante de inhibición, adimensional
    tau_gain_base: float = 20.0 # EMA lenta que ancla la línea base poblacional
    tau_det: float = 1.0        # ventana del ciclo de trabajo (s). Attanasi usa 1 s
    c_det: float = 1.5          # umbral de desalineación, en unidades de sigma_theta

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("tau_W", "R_c", "c_signal", "R_call"):
            if np.isinf(d[k]):
                d[k] = None
        return json.dumps(d, sort_keys=True)


MODELS = ("ViSt", "CoLd", "CaSp", "WiSt", "WiSp")


# ----------------------------------------------------------------------------
# Utilidades vectoriales
# ----------------------------------------------------------------------------


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _t_scale_from_mad(nu):
    """Escala de una t_nu que tiene la misma desviación robusta que una gaussiana.

    sigma_MAD = 1.4826 * mediana(|X|). Para la t con escala s eso es
    1.4826 * s * F^-1(0.75); para la gaussiana, sigma. Igualarlas es lo que hace
    comparables 'student' y 'gauss' en la misma sigma_theta.
    """
    from scipy.stats import t as _t
    return 1.0 / (1.4826 * float(_t.ppf(0.75, nu)))


def _cross_z(a, b):
    """Componente z del producto vectorial de dos campos de vectores 2D."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c], axis=1)


def hex_disk(N, d0, rng):
    """N agentes en red hexagonal (paso d0) recortada a un disco, con jitter leve.

    Reproduce "N agents are placed homogeneously within a circle, separated by
    15 m on average" del paper.
    """
    m = int(np.ceil(np.sqrt(N))) + 4
    ii, jj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    x = d0 * (ii + 0.5 * (jj % 2))
    y = d0 * (np.sqrt(3) / 2) * jj
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1))][:N]
    pts = pts - pts.mean(axis=0)
    pts += rng.normal(0.0, 0.05 * d0, size=pts.shape)
    return pts


# ----------------------------------------------------------------------------
# Campo de viento turbulento (propuesta M20)
# ----------------------------------------------------------------------------


class WindField:
    """Campo de aceleración espacialmente correlacionado, tipo racha turbulenta.

    Suma de `n_modes` modos de Fourier con número de onda |q| = 2*pi/ell y
    amplitudes que evolucionan como procesos de Ornstein-Uhlenbeck de tiempo
    característico `tau`. Da un campo suave, con correlación espacial de escala
    `ell` y correlación temporal `tau`, a coste O(N * n_modes) por paso.

    El límite ell -> infinito es un viento uniforme (modo común puro): desplaza
    la bandada entera sin desordenarla. El límite ell -> 0 recupera ruido
    espacialmente blanco. Lo interesante está en medio, que es donde vuela un
    ave real: la ley 2/3 de Kolmogorov dice que dos aves separadas por r
    comparten casi el mismo viento cuando r << ell.
    """

    def __init__(self, p: Params, rng):
        self.p, self.rng = p, rng
        m = p.wind_modes
        ang = rng.uniform(0, 2 * np.pi, m)
        self.q = (2 * np.pi / p.wind_ell) * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.phase = rng.uniform(0, 2 * np.pi, m)
        self.dirs = rng.normal(size=(m, 2))
        self.dirs /= np.linalg.norm(self.dirs, axis=1, keepdims=True)
        self.amp = rng.normal(0.0, 1.0, m)
        self.gust = 1.0

    def step(self):
        p = self.p
        a = np.exp(-p.dt / p.wind_tau)
        self.amp = a * self.amp + np.sqrt(1 - a ** 2) * self.rng.normal(size=self.amp.shape)
        if p.wind_intermittent:
            # factor de racha: se renueva en escala wind_tau y sigue 1/sqrt(chi2/nu),
            # que es justo lo que convierte una gaussiana en t de Student.
            if self.rng.random() < p.dt / p.wind_tau:
                nu = p.wind_nu
                self.gust = float(np.sqrt(nu / self.rng.chisquare(nu)) *
                                  np.sqrt((nu - 2.0) / nu))

    def at(self, pos):
        """Aceleración del viento en cada posición: (N, 2)."""
        ph = pos @ self.q.T + self.phase                      # (N, m)
        w = np.cos(ph) * self.amp                             # (N, m)
        out = w @ self.dirs                                   # (N, 2)
        return out * (self.gust * self.p.wind_sigma / np.sqrt(self.p.wind_modes / 2))


# ----------------------------------------------------------------------------
# Simulación
# ----------------------------------------------------------------------------


class Flock:
    """Estado de una bandada y su regla de actualización."""

    def __init__(self, p: Params, model: str):
        assert model in MODELS, f"modelo desconocido: {model}"
        self.p, self.model = p, model
        self.rng = np.random.default_rng(p.seed)

        self.pos = hex_disk(p.N, p.d0, self.rng)
        self.vel = np.tile(np.array([p.v0, 0.0]), (p.N, 1))
        self.spin = np.zeros(p.N)
        self.will = np.zeros(p.N)                       # will propio, en [0, 1]
        self.d_des = np.tile(np.array([1.0, 0.0]), (p.N, 1))  # dirección deseada del informado

        # agente 0 del paper: el más cercano al centro geométrico
        self.leader = int(np.argmin(np.linalg.norm(self.pos - self.pos.mean(0), axis=1)))
        self.forced = None       # dirección impuesta al líder (None = libre)
        self.t = 0.0
        self.wind = WindField(p, self.rng) if p.wind_sigma > 0 else None
        self.u_wind = np.zeros((p.N, 2))   # velocidad de deriva inducida por el viento

        # estado del observador bayesiano (M2): rumbo y su velocidad suavizada
        self.theta = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        self.omega = np.zeros(p.N)
        # estado del detector por ciclo de trabajo (E18). Se inicializa en q0 (la
        # fracción esperada bajo H0) en la primera llamada a _llr_ciclo, cuando
        # q0 ya está calculado.
        self.delta = np.zeros(p.N)
        self._paso = 0
        self._delta_paso = -1
        self._q01 = None
        # acumulador del detector secuencial (E22)
        self.llr_acc = np.zeros(p.N)
        self._acc_paso = -1
        # normalización divisiva (E26)
        self.m_gain = np.zeros(p.N)
        self.m_base = -1.0
        self._gain_paso = -1
        self.n_fp = 0                      # falsas alarmas emitidas (M1), para contabilidad

        # canal acústico (M19): tiempo que le queda de sonar a la llamada de cada
        # uno, y quién decidió llamar en el episodio de alarma en curso
        self.call_t = np.zeros(p.N)
        self.calling = np.zeros(p.N, dtype=bool)
        self._was_alarmed = np.zeros(p.N, dtype=bool)   # estado del paso ANTERIOR
        self.n_call = 0

        # Buffer circular para el retardo de propagación (M1). Guarda el pasado
        # reciente de lo que un vecino EMITE -- su will y su velocidad angular --
        # para que el observador lea el estado de hace Delta_ij = d_ij/c_signal.
        self._buf_n = 1
        if np.isfinite(p.c_signal) and p.c_signal > 0:
            # profundidad suficiente para el diámetro inicial de la bandada
            r0 = float(np.linalg.norm(self.pos - self.pos.mean(0), axis=1).max())
            self._buf_n = max(2, int(np.ceil(4.0 * r0 / p.c_signal / p.dt)) + 2)
        self._buf_will = np.zeros((self._buf_n, p.N))
        self._buf_om = np.zeros((self._buf_n, p.N))
        self._buf_call = np.zeros((self._buf_n, p.N))
        self._head = 0

    # -- reglas de alineamiento ------------------------------------------------

    def _neighbors(self):
        """Índices y distancias de los k vecinos topológicos (idx[:,0] = uno mismo)."""
        if self.p.k >= self.p.N - 1:                    # campo medio: sin kd-tree
            idx = np.broadcast_to(np.arange(self.p.N), (self.p.N, self.p.N))
            if not np.isfinite(self.p.R_c):             # nadie va a mirar las distancias
                return idx, None
            d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
            return idx, d
        d, idx = cKDTree(self.pos).query(self.pos, k=self.p.k + 1)
        return np.atleast_2d(idx), np.atleast_2d(d)     # (N, k+1)

    def _v_vist(self, idx):
        """Ec. (2.1): promedio local de velocidades, normalizado.

        Con `air_frame`, el promedio se hace sobre velocidades relativas al aire:
        el agente resta su PROPIA estimación de la deriva del viento a la de sus
        vecinos. Como todos comparten casi el mismo viento (r << ell), el modo
        común se cancela y sólo queda el viento diferencial. Es rechazo de modo
        común, igual que en un amplificador diferencial.
        """
        if self.p.air_frame and self.wind is not None:
            v_rel = self.vel[idx] - self.u_wind[:, None, :]
            return _normalize(v_rel.sum(axis=1))
        return _normalize(self.vel[idx].sum(axis=1))

    def _v_cold(self, v_vist):
        """Ec. (2.2): el informado tira de su propia dirección deseada."""
        w = self.will[:, None]
        return _normalize((1.0 - w) * v_vist + w * self.d_des)

    def _lag(self, buf, actual, idx, dist):
        """Lee el estado de cada vecino con su retardo: Delta_ij = d_ij / c_signal.

        Un observador no ve el estado ACTUAL de un vecino lejano, sino el de hace
        Delta_ij. Para el sonido en aire (343 m/s) y una bandada de 200 m eso es
        medio segundo, comparable a la duración de la maniobra que se intenta
        detectar. Con c_signal infinita no hay retardo y se lee `actual`, sin
        pasar por el buffer: si no, el propio buffer metería un retardo espurio
        de un paso. Devuelve (N, k+1).
        """
        p = self.p
        if not (np.isfinite(p.c_signal) and p.c_signal > 0) or dist is None:
            return actual[idx]
        lag = np.minimum((dist / p.c_signal / p.dt).astype(np.int64), self._buf_n - 1)
        return buf[(self._head - lag) % self._buf_n, idx]

    def _llr_kinematic(self, idx):
        """Evidencia que aporta la CINEMÁTICA de cada vecino. (N,) por emisor.

        El observable es el mismo residuo que se midió en aves reales (E2):

            eps_j = omega_j - <omega>_{vecinos de j}

        es decir, cuánto se desvía la velocidad angular de j de la que le dicta su
        propio vecindario. Bajo H0 (sin intención) eps se distribuye con escala
        sigma_theta; bajo H1 (maniobra intencional) con escala sqrt(sigma_theta^2 +
        sigma_W^2). Es un test de ESCALA, no de posición: la intención no sesga el
        giro en una dirección, lo hace más grande.

        La verosimilitud es una t de Student, no una gaussiana: con nu ~ 2.8 medido
        en grajillas, el estadístico cuadrático gaussiano diverge con eps y un solo
        dato atípico produce certeza absoluta. El log-cociente de la t SATURA en
        nu*log(s1/s0), así que la evidencia que un vecino puede aportar está acotada.
        """
        p = self.p
        eps = self.omega - self.omega[idx].mean(axis=1)
        s0 = p.sigma_theta
        s1 = float(np.hypot(p.sigma_theta, p.sigma_W))
        if p.detector == "ciclo":
            return self._llr_ciclo(eps, s0, s1)
        if p.bayes_lik == "gauss":
            llr = np.log(s0 / s1) + 0.5 * eps ** 2 * (1.0 / s0 ** 2 - 1.0 / s1 ** 2)
        else:
            # sigma_theta es la anchura ROBUSTA (equivalente-MAD), no el parámetro
            # de escala de la t. Sin esta conversión, la t y la gaussiana estarían
            # centradas en anchuras distintas y compararlas mediría la escala, no
            # la forma.
            nu = p.nu
            c = _t_scale_from_mad(nu)
            t0, t1 = s0 * c, s1 * c
            llr = np.log(t0 / t1) + 0.5 * (nu + 1.0) * (
                np.log1p(eps ** 2 / (nu * t0 ** 2))
                - np.log1p(eps ** 2 / (nu * t1 ** 2)))
        if p.detector == "secuencial":
            return self._llr_secuencial(llr)
        return llr

    def _llr_secuencial(self, llr_inst):
        """LLR secuencial con olvido: la versión CON PRINCIPIOS del ciclo (E22).

        El ciclo de trabajo binariza la excedencia antes de acumular, lo que tira
        información. El acumulador correcto para una señal de duración ~tau_det es
        la media móvil exponencial del LLR instantáneo completo, escalada por el
        número efectivo de muestras independientes (tau_det/tau_obs, porque el
        residuo ya viene suavizado con tau_obs y ése es su tiempo de correlación).

        Es la reparación honesta de la etiqueta «óptimo» de E4: aquel detector era
        el óptimo para señal instantánea; éste lo es (aprox.) para señal extendida.
        Arranca en 0 --- levemente optimista bajo H0, cuya media es negativa --- y
        el asentamiento previo al test (3 s >> tau_det) lo equilibra.
        """
        p = self.p
        if self._acc_paso != self._paso:
            a = np.exp(-p.dt / max(p.tau_det, 1e-6))
            self.llr_acc = a * self.llr_acc + (1.0 - a) * llr_inst
            self._acc_paso = self._paso
        n_ef = max(p.tau_det / max(p.tau_obs, 1e-6), 1.0)
        return n_ef * self.llr_acc

    def _llr_ciclo(self, eps, s0, s1):
        """Evidencia por CICLO DE TRABAJO, no por amplitud. (N,).

        Attanasi et al. 2015 (JRSI 12:20150319) definen el factor de tiempo de
        desalineación como la fracción de una ventana `tau` en que el individuo se
        desvía por encima de un umbral (su ec. S2), y muestran que **la amplitud
        de la desviación no discrimina iniciadores mientras que su persistencia
        sí** --- su ec. (S3) es justamente el control negativo. Su ventana es de
        1 s, elegida empíricamente.

        Aquí se traslada a la referencia local: ellos comparan contra la dirección
        MEDIA GLOBAL de la bandada, lo que confunde desalineación con posición en
        el grupo; `eps` compara contra los vecinos, que es lo que un observador
        embebido puede medir y lo que cancela el sesgo geométrico.

        El log-cociente es binomial. Si `q0` y `q1` son las probabilidades de
        superar el umbral bajo cada hipótesis --- calculadas de la MISMA t de
        Student que usa el detector de amplitud, sin parámetros nuevos --- entonces
        observar una fracción `d` sobre `n_ef` muestras independientes da

            LLR = n_ef * [ d*log(q1/q0) + (1-d)*log((1-q1)/(1-q0)) ]

        con `n_ef = tau_det / tau_obs`, porque el residuo ya viene suavizado con
        constante `tau_obs` y ése es su tiempo de correlación.
        """
        p = self.p
        umbral = p.c_det * p.sigma_theta

        if self._q01 is None:
            from scipy.stats import t as _t
            c = _t_scale_from_mad(p.nu)
            q0 = 2.0 * _t.sf(umbral / (s0 * c), p.nu)
            q1 = 2.0 * _t.sf(umbral / (s1 * c), p.nu)
            self._q01 = (float(np.clip(q0, 1e-6, 1 - 1e-6)),
                         float(np.clip(q1, 1e-6, 1 - 1e-6)))
            # delta arranca en q0, la fracción esperada bajo H0 — no en cero.
            # Arrancar en cero es evidencia espuria a favor de H0 («nunca se ha
            # desviado») que tarda ~tau_det en purgarse; con tau_det comparable
            # al asentamiento previo al test, la EMA llega sin equilibrar y el
            # colapso a tau_det grande sale inflado.
            self.delta[:] = self._q01[0]
        q0, q1 = self._q01

        # El ciclo de trabajo se actualiza UNA vez por paso aunque el detector se
        # consulte varias (w_hat_emisor vuelve a llamar a w_hat en diagnóstico).
        if self._delta_paso != self._paso:
            a = np.exp(-p.dt / max(p.tau_det, 1e-6))
            self.delta = a * self.delta + (1.0 - a) * (np.abs(eps) > umbral)
            self._delta_paso = self._paso

        n_ef = max(p.tau_det / max(p.tau_obs, 1e-6), 1.0)
        return n_ef * (self.delta * np.log(q1 / q0)
                       + (1.0 - self.delta) * np.log((1 - q1) / (1 - q0)))

    def _llr_acoustic(self, idx, dist):
        """Evidencia que aporta OÍR (o no oír) la llamada de cada vecino. (N, k+1).

        Un bit por par, así que la evidencia es un escalón de dos valores:

            oigo    ->  log(p_d / p_fa)          (positiva: p_d > p_fa)
            no oigo ->  log((1-p_d)/(1-p_fa))    (negativa, y pequeña)

        Lo que hace valioso este canal no es su fuerza sino su DURACIÓN. E4 midió
        que la huella cinemática de una decisión se apaga en ~1 s mientras el
        estado dura tau_W: el término de arriba sigue disponible tau_call segundos,
        que es exactamente el hueco que la cinemática deja abierto.

        Es un canal a distancia y con retardo: se lee del buffer, se recorta a
        R_call y se pierde con probabilidad 1 - p_hear.
        """
        p = self.p
        oido = self._lag(self._buf_call, self.call_t, idx, dist) > 0.0
        if np.isfinite(p.R_call) and dist is not None:
            oido &= dist <= p.R_call
        if p.p_hear < 1.0:
            oido &= self.rng.random(oido.shape) < p.p_hear
        pd = min(max(p.p_call_det, 1e-6), 1 - 1e-6)
        pf = min(max(p.p_call_fa, 1e-6), 1 - 1e-6)
        return np.where(oido, np.log(pd / pf), np.log((1 - pd) / (1 - pf)))

    def w_hat(self, idx, dist=None):
        """Posterior de intención sobre cada vecino: (N, k+1).

        Los dos canales son condicionalmente independientes dada la intención, así
        que sus log-cocientes se SUMAN antes del prior. Sin M19 el término acústico
        es cero y todos los observadores llegan al mismo posterior sobre j.
        """
        p = self.p
        llr = self._llr_kinematic(idx)[idx]             # (N, k+1), por emisor
        if p.call_on:
            llr = llr + self._llr_acoustic(idx, dist)
        logit = llr + np.log(p.pi0 / (1.0 - p.pi0))
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))

    def w_hat_emisor(self, idx, dist=None):
        """Posterior sobre cada agente visto como EMISOR: (N,). Para diagnóstico.

        Con un solo canal todos los observadores coinciden, así que es exacto. Con
        M19 el posterior depende del par (unos oyen la llamada y otros no) y esto
        devuelve la media sobre quienes lo tienen por vecino, que es lo que hay que
        puntuar cuando la etiqueta es del emisor.
        """
        w = self.w_hat(idx, dist)
        tot = np.bincount(idx.ravel(), weights=w.ravel(), minlength=self.p.N)
        cnt = np.bincount(idx.ravel(), minlength=self.p.N)
        return tot / np.maximum(cnt, 1)

    def _perceived_will(self, idx, dist):
        """Lo que cada agente CREE que quiere cada vecino: (N, k+1).

        Aquí es donde el oráculo del paper deja de serlo. Tres capas:
          1. la fuente  -> w_j exacto (oracle) o el posterior de w_hat (bayes);
          2. el canal   -> alcance R_c, pérdida p_loss y retardo d_ij/c_signal (M1);
          3. uno mismo  -> la propia intención siempre se conoce sin error.
        """
        p = self.p
        if p.will_mode == "oracle":
            w_nb = self._lag(self._buf_will, self.will, idx, dist)
        else:
            w_nb = self.w_hat(idx, dist)
        if np.isfinite(p.R_c):
            w_nb = np.where(dist <= p.R_c, w_nb, 0.0)
        if p.p_loss > 0.0:
            w_nb = np.where(self.rng.random(w_nb.shape) < p.p_loss, 0.0, w_nb)
        if p.gain_on:
            # E26: «motion that is surprising» — la ganancia cae cuando la entrada
            # agregada es alta y SOSTENIDA (cascadas), no ante el evento único.
            u = w_nb.sum(axis=1)
            if self._gain_paso != self._paso:
                a = np.exp(-p.dt / max(p.tau_gain, 1e-6))
                self.m_gain = a * self.m_gain + (1.0 - a) * u
                ab = np.exp(-p.dt / max(p.tau_gain_base, 1e-6))
                mb = float(self.m_gain.mean())
                self.m_base = mb if self.m_base < 0 else ab * self.m_base + (1.0 - ab) * mb
                self._gain_paso = self._paso
            g = (p.gamma_gain + 1.0) / (p.gamma_gain
                                        + self.m_gain / max(self.m_base, 1e-9))
            # la ganancia modula lo que se percibe de los VECINOS; la voluntad
            # propia (columna own, restaurada abajo) no se auto-suprime. Cap en 1:
            # la ganancia de Fahimipour SOLO suprime (1/(gamma+m) es monótona
            # decreciente); amplificar bajo la línea base sería un artefacto del
            # anclaje adimensional.
            w_nb = w_nb * np.clip(g, 0.0, 1.0)[:, None]
        own = idx == np.arange(p.N)[:, None]
        return np.where(own, self.will[:, None], w_nb)

    def salience(self, idx, dist):
        """Saliencia s_ij: cuánta atención merece el vecino j según cada mecanismo.

        Es el eje de la taxonomía de M9. Los mecanismos publicados desde 2020 para
        romper el compromiso se diferencian en qué usan como saliencia y en cuán
        agresiva es la concentración beta:

          'will'        w_j (oráculo del paper) o su posterior (BayWi)
          'visual'      velocidad angular aparente de j en la retina de i (AMS 2024)
          'visual_step' la misma, umbralizada -> winner-take-all (MST 2025)

        La saliencia visual es puramente geométrica: no requiere leer estados
        internos ni comunicar nada. Es |r x u| / |r|^2, la tasa a la que j se
        desplaza en el campo visual de i, normalizada a orden 1 por v0/d0.
        """
        p = self.p
        if p.salience == "will":
            return self._perceived_will(idx, dist)
        r = self.pos[idx] - self.pos[:, None, :]        # (N, k+1, 2)
        u = self.vel[idx] - self.vel[:, None, :]
        # La velocidad angular aparente va como 1/|r|^2 y DIVERGE cuando dos
        # agentes casi se tocan. Un ojo real no: satura. Se aplica compresión
        # tipo Weber, s/(1+s/s_sat), que es suave, acotada y además es la misma
        # ley que Liao et al. (2024) invocan para el canal sensorial.
        d2 = np.maximum((r ** 2).sum(-1), p.r_body ** 2)
        s = np.abs(_cross_z(r, u)) / d2 * (p.d0 / p.v0)
        s = s / (1.0 + s / p.s_sat)
        own = idx == np.arange(p.N)[:, None]
        s = np.where(own, 0.0, s)
        if p.salience == "visual_step":
            return (s > p.s_thresh).astype(float)
        return s

    def _v_wist(self, idx, dist, v_cold):
        """Ec. (2.5): los vecinos SÍ perciben el will -> jerarquía estrella local.

        beta=None reproduce el paper (suma no acotada).
        beta float activa la atención softmax de la propuesta M3 (AttWi).
        """
        w_nb = self.salience(idx, dist)                 # (N, k+1)
        v_nb = self.vel[idx] / self.p.v0                # (N, k+1, 2)
        if self.p.beta is None:
            extra = (w_nb[:, :, None] * v_nb).sum(axis=1)
            return _normalize(v_cold + extra)
        # M3 (AttWi): el softmax reemplaza al promedio local, pero se conserva la
        # estructura CoLd — si no, un agente informado se seguiría a sí mismo hacia
        # su velocidad ACTUAL en vez de hacia su dirección deseada, y el líder
        # dejaría de liderar. beta=0 y will=0 devuelve exactamente ViSt.
        # softmax numéricamente estable: restar el máximo es matemáticamente
        # neutro y evita que beta grande desborde el exponencial
        a = np.exp(self.p.beta * (w_nb - w_nb.max(axis=1, keepdims=True)))
        a /= a.sum(axis=1, keepdims=True)
        v_att = _normalize((a[:, :, None] * v_nb).sum(axis=1))
        w = self.will[:, None]
        return _normalize((1.0 - w) * v_att + w * self.d_des)

    def _v_spin(self, target):
        """Ecs. (2.3)-(2.4) / (2.6)-(2.7): el spin gira la velocidad hacia `target`."""
        p = self.p
        self.spin = self.spin * np.exp(-p.dt / p.tau_s) + \
            p.J * _cross_z(self.vel / p.v0, target) * p.dt
        return _normalize(_rotate(self.vel, self.spin * p.dt))

    def desired_direction(self):
        idx, dist = self._neighbors()
        v_vist = self._v_vist(idx)
        if self.model == "ViSt":
            out = v_vist
        elif self.model == "CoLd":
            out = self._v_cold(v_vist)
        elif self.model == "CaSp":
            out = self._v_spin(v_vist)
        elif self.model == "WiSt":
            out = self._v_wist(idx, dist, self._v_cold(v_vist))
        elif self.model == "WiSp":
            out = self._v_spin(self._v_wist(idx, dist, self._v_cold(v_vist)))
        if self.forced is not None:
            out = out.copy()
            out[self.leader] = self.forced              # ec. (3.1)
        return out

    # -- integración -----------------------------------------------------------

    def step(self):
        p = self.p
        v_des = p.v0 * self.desired_direction()
        dv = v_des - self.vel
        mag = np.linalg.norm(dv, axis=1)
        acc = _normalize(dv) * np.minimum(mag / p.tau_c, p.a_max)[:, None]

        s = p.sigma_xi if p.noise_mode == "paper" else p.sigma_xi / np.sqrt(p.dt)
        if p.noise_dist == "student":
            # t multivariada por agente: gaussiana dividida por una escala chi.
            # Se reescala para que la VARIANZA siga siendo sigma^2 — así el
            # contraste con 'gauss' aísla la FORMA de las colas y no su potencia.
            nu = p.noise_nu
            g = self.rng.normal(0.0, s * np.sqrt((nu - 2.0) / nu), size=(p.N, 2))
            noise = g / np.sqrt(self.rng.chisquare(nu, size=(p.N, 1)) / nu)
        else:
            noise = self.rng.normal(0.0, s, size=(p.N, 2))  # ec. (2.8)

        if self.wind is not None:
            self.wind.step()
            w = self.wind.at(self.pos)
            noise = noise + w
            # el ave siente el viento: estima su propia deriva integrándolo
            self.u_wind += (w - self.u_wind / p.tau_c) * p.dt

        self.vel += (acc + noise) * p.dt
        self.pos += self.vel * p.dt

        # velocidad angular suavizada: es lo único que un observador puede medir.
        # Media móvil exponencial de constante tau_obs. Sin suavizar, la omega
        # instantánea es el incremento de un camino aleatorio y su escala depende
        # de dt, no del ave.
        th = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        dth = np.angle(np.exp(1j * (th - self.theta))) / p.dt
        a_obs = np.exp(-p.dt / p.tau_obs)
        self.omega = a_obs * self.omega + (1.0 - a_obs) * dth
        self.theta = th

        if np.isfinite(p.tau_W):
            self.will *= np.exp(-p.dt / p.tau_W)
        if p.lambda_fp > 0.0:
            # M1: falsas alarmas. Un agente se sobresalta sin causa —una racha, un
            # reflejo, un vecino que se le cruza— y maniobra como si tuviera
            # información. Proceso de Poisson por agente, dirección uniforme.
            fire = self.rng.random(p.N) < p.lambda_fp * p.dt
            if self.forced is not None:
                fire[self.leader] = False               # el líder real no se sobresalta
            n = int(fire.sum())
            if n:
                a = self.rng.uniform(0.0, 2 * np.pi, n)
                self.d_des[fire] = np.stack([np.cos(a), np.sin(a)], axis=1)
                self.will[fire] = 1.0
                self.n_fp += n

        if p.call_on:
            # El ave llama MIENTRAS está alarmada, y su llamada sigue disponible
            # tau_call segundos DESPUÉS de que la alarma se apague. Por eso
            # tau_call es la variable de interés de esta etapa: mide cuánto
            # sobrevive la señal al estado que la originó, que es exactamente el
            # hueco que E4 encontró abierto (huella ~1 s, estado ~4 s).
            #
            # Se decide una vez por episodio, en el flanco: un ave que no llamó
            # al alarmarse no empieza a hacerlo después.
            self.call_t = np.maximum(self.call_t - p.dt, 0.0)
            activo = self.will > 0.5
            # el flanco se mide contra el paso ANTERIOR, no contra el estado de
            # este paso: si no, un agente cuya alarma se fija desde fuera (el
            # líder del banco de responsividad) nunca parecería recién alarmado
            nuevo = activo & (~self._was_alarmed)
            self._was_alarmed = activo
            if nuevo.any():
                self.calling &= activo                  # cierra episodios viejos
                decide = nuevo & (self.rng.random(p.N) < p.p_call_det)
                self.calling |= decide
                self.n_call += int(decide.sum())
            self.call_t[self.calling & activo] = p.tau_call
            self.calling &= activo
            if p.p_call_fa > 0.0:
                # vocalización espuria: el falso positivo del propio canal, con
                # tasa por unidad de tiempo, independiente de la cinemática
                esp = (~activo) & (self.rng.random(p.N) < p.p_call_fa * p.dt)
                if esp.any():
                    self.call_t[esp] = p.tau_call

        # el buffer guarda lo que cada agente EMITE, para que un vecino lejano
        # lo lea con su retardo
        self._head = (self._head + 1) % self._buf_n
        self._buf_will[self._head] = self.will
        self._buf_om[self._head] = self.omega
        self._buf_call[self._head] = self.call_t

        self.t += p.dt
        self._paso += 1
        # Devuelve SOLO la parte determinista: el esfuerzo de control (M5) mide
        # cuánto maniobra el agente, no cuánto lo zarandea el ruido. Sumando el
        # ruido, su magnitud (sigma/sqrt(dt) ~ 14 m/s^2) domina por completo y
        # el observable sale idéntico para todos los modelos.
        return acc

    # -- observables -----------------------------------------------------------

    def V(self):
        return self.vel.mean(axis=0)

    def polarization(self):
        return np.linalg.norm(_normalize(self.vel).mean(axis=0))

    def cohesion(self, d_link=None):
        """¿Sigue siendo UNA bandada? Tercer eje, que el trabajo de partida no mide.

        Ni la responsividad ni la persistencia notan que el grupo se parta en dos:
        ambas se calculan sobre la velocidad media, que sigue existiendo. Pero
        Papadopoulou muestra que el coste real de la saliencia individual es la
        escisión, no la lentitud.

        Devuelve la fracción de agentes en el mayor grupo conexo del grafo de
        proximidad (umbral d_link) y el radio de giro relativo al inicial.
        """
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import coo_matrix

        d_link = d_link if d_link is not None else 2.5 * self.p.d0
        pairs = np.array(list(cKDTree(self.pos).query_pairs(d_link)), dtype=int)
        n = self.p.N
        if pairs.size:
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, lab = connected_components(g, directed=False)
            frac = np.bincount(lab).max() / n
        else:
            frac = 1.0 / n
        rg = float(np.sqrt(((self.pos - self.pos.mean(0)) ** 2).sum(1).mean()))
        return {"frac_max": float(frac), "rg": rg}


# ----------------------------------------------------------------------------
# Banco de pruebas 1: responsividad  (secciones 3.1-3.2 del paper)
# ----------------------------------------------------------------------------


def run_response(p: Params, model: str, phi: float, t_R: float, t_s: float = 3.0,
                 record_agents: bool = False, record_detector: bool = False,
                 every_coh: int = 25, pulse: bool = False):
    """Un test de responsividad: se relaja t_s, se gira al agente 0 en phi y se mide.

    Con `pulse=True` el líder ejecuta UNA maniobra y se le suelta: en cuanto su
    velocidad queda a <10° del objetivo (o a los 2.5 s como tope, la duración de
    la maniobra más larga según la cota a_max/v0), `forced` se retira y su will
    deja de reimponerse — decae con tau_W, que es la semántica del modelo para
    un estado que sobrevive a la maniobra. En el protocolo estándar (pulse=False,
    el del paper de partida) el líder es un tractor persistente durante todo t_R,
    con lo que «informado» y «desviándose de forma sostenida» coinciden por
    construcción; el pulso los separa (E20).

    Devuelve dict con:
        r         - ec. (3.2), la responsividad de esta corrida
        proj      - serie temporal de V(t)·n_phi / v0 durante t_R
        effort    - esfuerzo de control medio (propuesta M5)
        pol       - polarización media durante la ventana
        frac_max  - fracción en el mayor grupo conexo, al final de la ventana
        frac_min  - la peor a lo largo de la ventana (¿llegó a partirse?)
        n_fp      - falsas alarmas emitidas (M1)
        theta     - (opcional) rumbos por agente, para medir propagación (M6)
        dist0     - (opcional) distancia inicial de cada agente al líder
        det_score - (opcional) posterior w_hat submuestreado, y det_true la verdad
    """
    f = Flock(p, model)
    n_phi = np.array([np.cos(phi), np.sin(phi)])

    for _ in range(int(round(t_s / p.dt))):
        f.step()

    f.forced = n_phi
    f.will[f.leader] = 1.0
    f.d_des[f.leader] = n_phi
    dist0 = np.linalg.norm(f.pos - f.pos[f.leader], axis=1)

    n_steps = int(round(t_R / p.dt))
    proj, pol, eff = np.empty(n_steps), np.empty(n_steps), 0.0
    theta = np.empty((n_steps, p.N), dtype=np.float32) if record_agents else None
    fracs, score, truth = [], [], []

    soltado = False
    for s in range(n_steps):
        acc = f.step()
        if pulse and not soltado:
            v = f.vel[f.leader]
            alin = float(v @ n_phi) / max(float(np.linalg.norm(v)), 1e-9)
            if alin > np.cos(np.deg2rad(10.0)) or s * p.dt > 2.5:
                f.forced = None
                soltado = True
        if not soltado:
            f.will[f.leader] = 1.0                      # el líder mantiene will=1
        proj[s] = f.V() @ n_phi / p.v0
        pol[s] = f.polarization()
        eff += np.linalg.norm(acc, axis=1).mean()
        if record_agents:
            theta[s] = np.arctan2(f.vel[:, 1], f.vel[:, 0])
        if s % every_coh == 0:
            fracs.append(f.cohesion()["frac_max"])
            if record_detector:
                i, d_ = f._neighbors()
                score.append(f.w_hat_emisor(i, d_).astype(np.float32))
                truth.append((f.will > 0.5).astype(np.int8))

    out = {"r": float(proj.mean()), "proj": proj, "pol": float(pol.mean()),
           "effort": float(eff / n_steps), "phi": phi, "model": model,
           "frac_max": float(fracs[-1]), "frac_min": float(min(fracs)),
           "n_fp": int(f.n_fp)}
    if record_agents:
        out["theta"], out["dist0"] = theta, dist0
    if record_detector:
        out["det_score"], out["det_true"] = np.array(score), np.array(truth)
    return out


def phi_grid(d_phi_deg: float = 10.0):
    """Ángulos de prueba: regla del punto medio sobre (0, pi].

    Desviación deliberada del paper (que usa 10°, 20°, ..., 180°): con la rejilla
    del extremo derecho, una bandada que NO responde da r(phi) = cos(phi) y su R
    integrado sale -1/n en vez de 0, es decir el estimador está sesgado. Con los
    puntos medios la simetría de cos se cancela exactamente y R = 0 para la
    bandada sorda, que es justo lo que el paper afirma que significa R = 0.
    """
    n = int(round(180.0 / d_phi_deg))
    return np.deg2rad(d_phi_deg * (np.arange(n) + 0.5))


def responsivity(p: Params, model: str, t_R: float | None = None, n_rep: int = 5,
                 d_phi_deg: float = 10.0, seed0: int = 0, keep_curves: bool = False):
    """R = (1/pi) ∫ <r(phi)> dphi, ec. (3.3). Discretización angular de 10° por defecto."""
    t_R = t_R if t_R is not None else np.sqrt(p.N)
    phis = phi_grid(d_phi_deg)
    rows = []
    for i, phi in enumerate(phis):
        for rep in range(n_rep):
            q = Params(**{**asdict(p), "seed": seed0 + 1000 * i + rep})
            res = run_response(q, model, phi, t_R)
            rows.append({"phi": phi, "rep": rep, "r": res["r"],
                         "effort": res["effort"], "pol": res["pol"],
                         **({"proj": res["proj"]} if keep_curves else {})})
    r_mean = np.array([np.mean([x["r"] for x in rows if x["phi"] == phi]) for phi in phis])
    return {"R": float(r_mean.mean()), "phis": phis, "r_phi": r_mean, "rows": rows,
            "model": model, "k": p.k, "N": p.N, "t_R": t_R}


# ----------------------------------------------------------------------------
# Banco de pruebas 2: persistencia  (sección 3.3 del paper)
# ----------------------------------------------------------------------------


def angular_diffusion(p: Params, model: str, t_P: float = 600.0, n_rep: int = 40,
                      t_burn: float = 10.0, seed0: int = 0, sample_dt: float = 0.5):
    """Coeficiente de difusión D_Phi de la dirección global: <(ΔΦ)^2> = D_Phi t."""
    every = max(1, int(round(sample_dt / p.dt)))
    n_samp = int(round(t_P / p.dt)) // every
    dPhi = np.empty((n_rep, n_samp))
    coh, nfp = [], []

    for rep in range(n_rep):
        q = Params(**{**asdict(p), "seed": seed0 + rep})
        f = Flock(q, model)
        for _ in range(int(round(t_burn / q.dt))):
            f.step()
        V = f.V()
        phi_prev = phi_unwrap = np.arctan2(V[1], V[0])
        phi0 = phi_unwrap
        for s in range(n_samp):
            for _ in range(every):
                f.step()
            V = f.V()
            a = np.arctan2(V[1], V[0])
            phi_unwrap += np.angle(np.exp(1j * (a - phi_prev)))
            phi_prev = a
            dPhi[rep, s] = phi_unwrap - phi0
        coh.append(f.cohesion()["frac_max"])
        nfp.append(f.n_fp)

    t = (np.arange(1, n_samp + 1) * every * p.dt)
    msd = (dPhi ** 2).mean(axis=0)
    # ajuste sin ordenada al origen sobre la mitad final (régimen difusivo)
    m = t > t[-1] / 2
    D = float((t[m] @ msd[m]) / (t[m] @ t[m]))
    return {"D": D, "t": t, "msd": msd, "model": model,
            "frac_max": float(np.mean(coh)), "n_fp": float(np.mean(nfp))}


def persistence(p: Params, model: str, D_ref: float | None = None, **kw):
    """P = (D^s_Phi / N) / D_Phi, ec. (3.5).

    La referencia D^s_Phi/N es, por construcción del paper, el D_Phi del modelo
    ViSt de campo medio (k = N). Si no se pasa, se calcula aquí.
    """
    if D_ref is None:
        q = Params(**{**asdict(p), "k": p.N - 1})
        D_ref = angular_diffusion(q, "ViSt", **kw)["D"]
    D = angular_diffusion(p, model, **kw)["D"]
    return {"P": float(D_ref / D), "D": D, "D_ref": float(D_ref), "model": model}


# ----------------------------------------------------------------------------
# Observables de la propuesta M5 (se calculan sobre `proj` ya simulado)
# ----------------------------------------------------------------------------


def response_observables(proj, dt, phi, smooth=0.2):
    """Descompone la curva de respuesta en latencia, exactitud y sobre-impulso.

    Todo se mide sobre el **progreso** del giro,

        prog(t) = [V(t)·n_phi/v0 - cos(phi)] / [1 - cos(phi)],

    que vale 0 si la bandada sigue recta y 1 si completó el giro. Sin esta
    normalización los observables no son comparables entre ángulos: a phi=45°
    la proyección ya arranca en cos(45°)=0.71 y cualquier umbral tipo "0.5"
    se cumple en el primer paso, aunque la bandada no haya girado nada.
    """
    base = np.cos(phi)
    prog = (np.asarray(proj) - base) / (1.0 - base)
    w = max(1, int(round(smooth / dt)))
    sm = np.convolve(prog, np.ones(w) / w, mode="same")
    above = np.flatnonzero(sm >= 0.5)
    t50 = float(above[0] * dt) if above.size else np.nan
    r_inf = float(sm[int(0.7 * len(sm)):].mean())
    return {"t50": t50, "r_inf": r_inf, "overshoot": float(sm.max() - r_inf),
            "r_excess": float(prog.mean())}


def propagation_speed(theta, dist0, dt, target_frac=0.5):
    """Velocidad del frente de giro: pendiente de (retardo de giro) vs (distancia al líder).

    Sirve para el test solitónico de la propuesta M6: c_prop(phi) constante
    (onda de spin lineal) frente a c_prop creciente con phi (no reciprocidad).
    """
    th = np.unwrap(theta, axis=0)
    dth = th - th[0]
    final = dth[-1]
    good = np.abs(final) > 1e-3
    delay = np.full(theta.shape[1], np.nan)
    for i in np.flatnonzero(good):
        hit = np.flatnonzero(np.abs(dth[:, i]) >= target_frac * abs(final[i]))
        if hit.size:
            delay[i] = hit[0] * dt
    m = np.isfinite(delay) & (dist0 > 0)
    if m.sum() < 10:
        return {"c_prop": np.nan, "delay": delay}
    A = np.stack([dist0[m], np.ones(m.sum())], axis=1)
    slope, _ = np.linalg.lstsq(A, delay[m], rcond=None)[0]
    return {"c_prop": float(1.0 / slope) if slope > 0 else np.inf, "delay": delay}


# ----------------------------------------------------------------------------
# Barridos paralelos (las réplicas son independientes -> paralelismo por corrida)
# ----------------------------------------------------------------------------


def _job_response(args):
    cfg, model, phi, t_R, seed = args
    q = Params(**{**cfg, "seed": seed})
    res = run_response(q, model, phi, t_R)
    return {"model": model, "k": q.k, "N": q.N, "phi": phi, "seed": seed,
            "r": res["r"], "effort": res["effort"], "pol": res["pol"],
            "frac_max": res["frac_max"], "frac_min": res["frac_min"],
            "n_fp": res["n_fp"], "proj": res["proj"].astype(np.float32)}


def sweep_responsivity(base: Params, models, ks, n_rep=5, d_phi_deg=10.0,
                       t_R=None, workers=None):
    """Barrido (modelo x k x ángulo x réplica) en paralelo. Devuelve lista de dicts."""
    from concurrent.futures import ProcessPoolExecutor

    phis = phi_grid(d_phi_deg)
    jobs = []
    for model in models:
        for k in ks:
            cfg = {**asdict(base), "k": int(k)}
            tR = t_R if t_R is not None else np.sqrt(base.N)
            for i, phi in enumerate(phis):
                for rep in range(n_rep):
                    jobs.append((cfg, model, float(phi), float(tR),
                                 base.seed + 1000 * i + rep))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job_response, jobs, chunksize=4))


def _job_diffusion(args):
    cfg, model, t_P, seed, t_burn, sample_dt = args
    q = Params(**{**cfg, "seed": seed})
    d = angular_diffusion(q, model, t_P=t_P, n_rep=1, t_burn=t_burn,
                          seed0=seed, sample_dt=sample_dt)
    return {"model": model, "k": q.k, "J": q.J, "tau_s": q.tau_s, "seed": seed,
            "t": d["t"], "msd": d["msd"], "frac_max": d["frac_max"], "n_fp": d["n_fp"]}


def sweep_diffusion(base: Params, models, n_rep=40, t_P=600.0, t_burn=10.0,
                    sample_dt=0.5, workers=None):
    """Difusión angular en paralelo. `models` puede ser [(modelo, dict_overrides), ...]."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for item in models:
        model, over = item if isinstance(item, tuple) else (item, {})
        cfg = {**asdict(base), **over}
        for rep in range(n_rep):
            jobs.append((cfg, model, t_P, base.seed + rep, t_burn, sample_dt))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_job_diffusion, jobs, chunksize=1))

    out = {}
    for r in rows:
        key = (r["model"], r["k"], r["J"], r["tau_s"])
        out.setdefault(key, []).append(r)
    res = {}
    for key, rs in out.items():
        msd = np.mean([r["msd"] for r in rs], axis=0)
        t = rows[0]["t"]
        m = t > t[-1] / 2
        res[key] = {"D": float((t[m] @ msd[m]) / (t[m] @ t[m])), "t": t, "msd": msd,
                    "frac_max": float(np.mean([r["frac_max"] for r in rs])),
                    "n_fp": float(np.mean([r["n_fp"] for r in rs]))}
    return res
