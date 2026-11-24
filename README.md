# What an observer of kinematics recovers of the oracle in models of collective leadership

Code and results for the paper of the same title (Lino Jeldez Salinas; manuscript in
preparation, title provisional).

Models of collective motion in which a few informed individuals steer the group often
assume that the other members can recognise who is informed (an *oracle*). The paper
measures how much of the benefit of that assumption an observer that only sees motion
recovers, in the models of Balázs, Vásárhelyi & Vicsek (2020, *J. R. Soc. Interface*
17:20190853): with one or several informed agents, under other noise, agility and
neighbourhood size, against memoryless observers tuned on separate replicates, and in
persistence with and without false alarms. It then asks what animal data can say about it,
including zebrafish groups in which the informed individuals are known in advance.

This repository contains only what the paper uses: the simulation engine, the scripts of
the stages whose results appear in the paper, their archived outputs, the separate control
reported in the supplement, and the scripts that turn those outputs into the paper's
figures and numbers.

> **Resumen en español.** Código y resultados del Paper A: el motor de simulación, los
> scripts de las etapas que usa el paper (E30, E35 y E42 a E77), sus salidas archivadas, el
> control independiente del suplemento y los generadores de figuras y cifras. Los
> comentarios del código están en español.

## Layout

| Folder | Contents |
|---|---|
| `codigo/` | Simulation engine, shared harnesses, data loaders and one script per stage |
| `resultados/` | Archived output of every stage: `E*.json` (results), `E*.txt` (report), `E*_crudo.json` (raw per-run values) |
| `figuras/` | `numeros.py` (every number and table of the paper, as LaTeX macros), `figuras_corto.py` (figures) and the six figures in `fig/` |
| `control_auditor/` | Separate control of memory against weight of evidence with matched false-positive rates (supplement), with its own frozen engine, pre-declared design, scripts and results |
| `datos/` | Empty. The animal data are third-party; see [`datos/README.md`](datos/README.md) |

## Requirements

Python 3.12 with NumPy, SciPy, Matplotlib and pandas (`pip install -r requirements.txt`).
The results were produced with Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0, Matplotlib
3.10.9 and pandas 3.0.3. The figures use the Latin Modern fonts of a TeX Live installation when it is present
and fall back to STIX otherwise.

## Reproducing

Run everything from the repository root, with the code folder on the path:

```bash
export PYTHONPATH=codigo
python figuras/numeros.py                 # numbers and tables -> figuras/numeros.tex, figuras/gen/
python figuras/figuras_corto.py           # figures -> figuras/fig/
python codigo/e71_comparacion_justa.py    # any stage: writes resultados/E71_*
```

There are three levels of reproduction, from seconds to hours:

1. **Numbers, tables and figures from the archived outputs** (seconds; no simulation and no
   animal data). `figuras/numeros.py` reads only `resultados/` and
   `control_auditor/resultados/` and writes the 813 macros and 31 table bodies, 26 of which
   the manuscript includes; `figuras/figuras_corto.py` draws the figures from the same files
   (the footprint figure simulates its sixteen short runs when it is drawn).
2. **Analyses that reread archived raw runs** (seconds to minutes; no new simulation). E75
   reads the heading series of E64 and E72, E71 its raw runs and those of E68, and E42,
   E44, E48 and E49 admit `E42_SOLO_ANALISIS=1` (and the like).
3. **Full simulations and animal analyses** (minutes to about three hours per stage with
   five processes). The animal stages need the third-party data in `datos/`.

`figuras/numeros.py` reproduces the paper's macros and tables exactly; the section *From
archived results to the paper* below maps every table and figure to the stages it comes
from. The simulations use fixed seeds (in the
common harness, `7 + 1000·i_phi + replicate`; informed agents chosen with `5000 +
replicate`), so a rerun reproduces the archived values. Most simulation stages simulate
again when they are run (the number of processes is set with `FASE4_W`) and keep partial
results in a `.parcial` file so that an interrupted run resumes. The raw runs of the
simulation stages are in `resultados/*_crudo.json`; E75b and E76 keep their per-run values
in their own JSON.

Some scripts compare their results with archived values from earlier stages of the
project ("anclas"); the files they read for that (`E9`, `E15`, `E16`, `E22`, `E27`, `E28`,
`E44b`) are included in `resultados/`. Two checks of E42 against earlier stages that are
not part of the paper (E20 and E41) are skipped when it is rerun here; the archived
`E42_escalera_N.json` records them as passed. The docstrings refer to a lab notebook
(`docs/bitacora/`) where each design and prediction was declared before running; the
notebook is not part of this repository, but every stage's report (`resultados/E*.txt`)
lists its predictions and whether they held.

## Stages

**Models (simulation).**

| Stage | Script | What it measures |
|---|---|---|
| E42, E42b | `e42_escalera_N.py`, `e42b_barrido_ventana.py` | Recovery of the oracle's advantage against flock size; sweep of the observer's window |
| E43, E43b | `e43_escalera_tauW.py`, `e43b_tauW_N1000.py` | Recovery against the decay time of the informed state, τ_W (N = 250 and 1000) |
| E44 | `e44_lectura.py` | Where the oracle's advantage comes from: reading time (oracle with a gate) |
| E45 | `e45_control_huella.py` | Noise control of the footprint (unlabelled agents in CaSp) |
| E46 | `e46_referencia_cold.py` | CoLd as the reference of the single manoeuvre; single source of the recovery figures |
| E47 | `e47_huella_modelos.py` | Footprint against turn amplitude and parameters (CoLd, WiSt) |
| E48 | `e48_replicas.py`, `absolutos_E48.py` | 32 paired replicates in the headline cells; absolute responsivities |
| E49 | `e49_sensibilidad.py` | Sensitivity to k and to the noise; observer that estimates its own noise floor |
| E55 | `e55_agilidad.py` | Intervention on the agility of all agents (a_max 5 to 40 m/s²) |
| E56 | `e56_ventana_ruido.py` | Optimal window with half and double noise |
| E58 | `e58_memoria_ganancia.py` | Memory against weight of the evidence |
| E59 | `e59_componentes_huella.py` | Components of the footprint |
| E60 | `e60_residuo_receptor.py` | Residual computed from what the receiver sees; identification of the informed state |
| E61, E61b | `e61_ventana_peso_fijo.py`, `e61b_bootstrap_ventana.py` | Optimal window with the weight of the evidence fixed; its resampling uncertainty |
| E62 | `e62_prior.py` | Sensitivity of the recovery to the prior π0 |
| E63 | `e63_varios_informados.py` | Several informed agents with a prior equal to their fraction |
| E64 | `e64_persistencia.py` | Persistence without informed agents and without false alarms |
| E65 | `e65_k70.py` | Recovery at k = 70, where the engine matches the published responsivities |
| E66 | `e66_replicas.py` | 16 replicates in the window, agility, noise and fixed-weight sweeps |
| E67 | `e67_deficit.py` | Choices that the reference paper leaves open and the engine's deficit at small k |
| E68 | `e68_fraccion_informados.py` | Recovery against the fraction of informed agents (1 to 48 informed; N = 250 and 1000) |
| E69 | `e69_dos_etapas.py` | Value of reading the leader for a time and at a fraction of its state; a two-stage law (failed) |
| E70, E70b | `e70_cabecera_robustez.py`, `e70b_ventana_agilidad.py` | The headline regime (12 of 250 informed) under other noise, agility and k; a window matched to the agility |
| E71 | `e71_comparacion_justa.py` | Integrating against memoryless observer with the same tuning budget, evaluated on new seeds; window from 0.1 to 8 s |
| E72 | `e72_persistencia_fa.py` | Persistence with false alarms at the rate measured in fish |
| E75 | `e75_falsas_alarmas_escala.py` | The same runs by measurement interval, as excess heading diffusion, against CoLd with the same false alarms, and their cohesion (reads the heading series of E64 and E72; no simulation) |
| E75b | `e75b_cohesion_sin_alarmas.py` | Cohesion in time in one persistence run without false alarms (repeats replicate 0 of E64 in six conditions, recording the largest connected group and the longest edge of the minimum spanning tree) |
| E76 | `e76_muestras_efectivas.py` | Calibration of the integrating observer's weight without informed agents: exceedance rate and effective number of independent samples of the duty cycle |

**Animal data.**

| Stage | Script | Data | What it measures |
|---|---|---|---|
| E73, E73b | `e73_peces_entrenados.py`, `e73b_variantes.py` | Zebrafish with trained individuals | Turn-rate signature of the informed fish, known in advance, before and after the light cue; duty cycle, k = 5 and speed (centred filters, the analysis declared beforehand) |
| E74 | `e74_peces_solo_pasado.py` | Zebrafish with trained individuals | The same with filters that read only past frames, the response of the first untrained fish, and the window and its end chosen inside the resampling and evaluated on the groups left out |
| E77 | `e77_peces_criterio_respuesta.py` | Zebrafish with trained individuals | Sensitivity of the AUC in the trials without an individual response of the untrained fish to the response criterion (threshold of two to four standard deviations held for 0.125 to 0.5 s) |
| E51, E51b | `e51_huella_aves.py`, `e51b_inyeccion.py` | Jackdaws, turning events | Footprint without a fixed threshold; known-truth injection control |
| E52, E52b | `e52_auc_ventana_aves.py`, `e52b_aves_por_grabacion.py` | Jackdaws, turning events | Initiator/follower AUC against the window; recording as resampling unit |
| E53 | `e53_agilidad_calibracion.py` | Jackdaws, continuous recordings | Agility and calibration of the observer |
| E54, E54b, E54c | `e54_peces.py`, `e54b_prospectivo.py`, `e54c_tetras_por_experimento.py` | Tetras | Footprint and AUC against the window; prospective control; per experiment |
| E57, E57b | `e57_palomas.py`, `e57b_potencia.py` | Homing pigeons, manipulated leader | Effect of the manipulation; sensitivity to injected model traces |
| E35 | `e35_espinosos.py` | Sticklebacks | AUC against the window and footprint |
| E30 | `e30_palomas.py` | Homing pigeons pursued by a robotic falcon | AUC against the window |

**Shared modules.** `flocking.py` (2D engine: ViSt, CoLd, CaSp, WiSt with an oracle or an
approximate Bayesian observer), `parametros.py` (reference parameters `P0` and the compared mechanisms),
`fase4.py` (common harness: runs, pairing, recovery and intervals), `plan16_comun.py`
(harness of E70 to E72), `e15_helpers.py` (footprint in the model), `e16_helpers.py`
(kinematic residual on real tracks), `turns.py`, `jackdaws3d.py`, `e27_tetras.py` and
`watts_palomas.py` (data loaders), and `carga_segura.py`, which reads the zebrafish
trajectory files (pickled Python objects) admitting only the numpy classes they contain,
so that no third-party code is executed.

## From archived results to the paper

Table and figure numbers are those of the manuscript of 23 November 2026 (the manuscript
itself is not in this repository). Every number quoted in its text is a macro written by
`figuras/numeros.py`; each table body is a file `figuras/gen/<name>.tex` written by the same
script from the stage outputs listed here.

| Paper | Body written by `numeros.py` | Stage outputs read |
|---|---|---|
| Table 1, regimes and recovery | `tab_regimenes` | E68, E71, E70, E48, E62, E65 |
| Table 2, the two observers on new replicates | `tab_justa_cuerpo` | E71 |
| Footprint table in the text | `tab_huella` | E47, E51 |
| S1, parameters | written by hand | `codigo/flocking.py`, `codigo/parametros.py` |
| S2, engine against the published responsivities | written by hand | validation runs; E67 recomputes k = 5, 15 and 70 |
| S3, footprint against the model parameters | written by hand | E47 |
| S4, components of the footprint | `tab_componentes` | E59 |
| S5, single leader by group size | `tab_cabecera` | E48, E46 |
| S6, single manoeuvre by decay time | `tab_tauW` | E46, E48 |
| S7, absolute responsivities | `tab_absolutos` | `absolutos_E48` |
| S8, interventions on what neighbours read | written by hand | E44 |
| S9, persistence without false alarms | `tab_persistencia` | E64 |
| S10, neighbourhood size and noise | `tab_malla` | E49 |
| S11, observer that estimates its noise floor | `tab_auto` | E49 |
| S12, what each agent can compute | `tab_receptor` | E60 |
| S13, twelve informed agents, three priors | `tab_multi` | E63 |
| S14, recovery against the number of informed agents | `tab_fraccion` | E68 |
| S15, persistence with false alarms | `tab_persistencia_fa` | E72 |
| S16, false alarms by interval, cost of reading, cohesion | `tab_fa_escala` | E75 (raw runs of E64 and E72) |
| S17, comparison with the same tuning budget | `tab_justa` | E71 |
| S18, window in the headline regime | `tab_ventana_cabecera` | E71 |
| S19, other noise, agility and neighbourhood size | `tab_robustez` | E70 |
| S20, memory and weight of the evidence | `tab_ganancia` | E58 |
| S21, value of reading the leader | `tab_lectura` | E69 |
| S22, agility of the agents | `tab_agilidad` | E55, E66 |
| S23, window with the weight fixed | `tab_peso_fijo` | E61, E66, E55, E56 |
| S24, outcomes declared beforehand | written by hand | the report of each stage, `resultados/E*.txt` |
| S25, calibration from jackdaw recordings | `tab_calib` | E53 |
| S26, zebrafish, centred filters | `tab_entrenados` | E73 |
| S27, zebrafish, past-only filters | `tab_entrenados_sp` | E74 |
| S28, zebrafish, windows tied to the response | `tab_entrenados_resp` | E74 |

`numeros.py` also writes five table bodies that the manuscript does not include
(`tab_cruz`, `tab_deteccion`, `tab_especies`, `tab_ruido_ventana`, `tab_tetraN`). The
cohesion run without false alarms (E75b), the calibration of the observer's weight (E76), the
sensitivity to the response criterion of the zebrafish (E77) and the separate control enter
the text as macros only.

| Figure | File in `figuras/fig/` | Stage outputs read |
|---|---|---|
| 1, the two ingredients of the measurement | `f_esquema.pdf` | none (drawing) |
| 2, residual of an agent with a sustained intention | `f2_footprint.pdf` | none: sixteen short runs simulated when it is drawn |
| 3, recovery, robustness and persistence | `f_frontera.pdf` | E68, E70, E70b, E64, E72 |
| 4, recovery against window, group size and decay time | `f_escalera.pdf` | E46, E66 |
| 5, zebrafish with trained individuals | `f_entrenados.pdf` | E74, E73 |
| 6, initiators and followers against the window | `f_especies.pdf` | E52b, E54, E54b |

## Separate control

`control_auditor/` holds the control described in the supplement as a separate analysis:
memory against weight of the evidence, with both observers calibrated to the same mean
weight and the same share of weights above one half in flight without informed agents. It
was designed and run apart from the main analysis, with its own frozen copy of the engine,
a design declared before running (`PREDECLARACION.md`, `ADENDA_ESTIMANDO.md`), its scripts
and its results; its documents are in Spanish. The per-run files (about 200 MB) are not
included; the tables and `resultados/RESULTADOS_MEMORIA_GANANCIA.json`, which the paper
reads, are.

## License

Code and results: MIT (see `LICENSE`). The animal data are not redistributed here; each
data set keeps its own license.

## Citation

See `CITATION.cff`. Each release of this repository is archived in Zenodo with its own DOI.
