# What an observer of kinematics recovers of the oracle in models of collective leadership

Code and results for the paper of the same title (Lino Jeldez Salinas; manuscript in
preparation, title provisional).

Models of collective motion in which a few informed individuals steer the group often
assume that the other members can recognise who is informed (an *oracle*). The paper
measures how much of the benefit of that assumption an observer that only sees motion
recovers, in the models of Balázs, Vásárhelyi & Vicsek (2020, *J. R. Soc. Interface*
17:20190853), and what the animal data (jackdaws, tetras, homing pigeons) can and cannot
say about it.

This repository contains only what the paper uses: the simulation engine, the scripts of
the stages whose results appear in the paper (E42 to E63), their archived outputs, and the
scripts that turn those outputs into the paper's figures and numbers.

> **Resumen en español.** Código y resultados del Paper A: el motor de simulación, los
> scripts de las etapas que usa el paper (E42 a E63), sus salidas archivadas y los
> generadores de figuras y cifras. Los comentarios del código están en español.

## Layout

| Folder | Contents |
|---|---|
| `codigo/` | Simulation engine, shared harness, data loaders and one script per stage |
| `resultados/` | Archived output of every stage: `E*.json` (results), `E*.txt` (report), `E*_crudo.json` (raw per-run values, so that the analyses can be redone without simulating) |
| `figuras/` | `figuras_corto.py` (figures), `numeros.py` (every number and table of the paper, as LaTeX macros) and the four figures in `fig/` |
| `datos/` | Empty. The animal data are third-party; see [`datos/README.md`](datos/README.md) |

## Requirements

Python 3.12 with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). The
results were produced with Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0 and Matplotlib 3.10.9.

## Reproducing

Run everything from the repository root, with the code folder on the path:

```bash
export PYTHONPATH=codigo
python figuras/numeros.py                 # numbers and tables -> figuras/numeros.tex, figuras/gen/
python figuras/figuras_corto.py           # figures -> figuras/fig/
python codigo/e46_referencia_cold.py      # any stage: writes resultados/E46_*
```

The stages that simulate reuse the raw runs stored in `resultados/*_crudo.json` when they
are present; delete a `_crudo.json` to simulate again. Several stages also admit
`E42_SOLO_ANALISIS=1`, `E44_SOLO_ANALISIS=1`, `E48_SOLO_ANALISIS=1` or
`E49_SOLO_ANALISIS=1` to redo only the analysis. The simulations use fixed seeds (in the
common harness, `7 + 1000·i_phi + replicate`), so a rerun reproduces the archived values.
The stages with animal data need the data in `datos/` first.

Some scripts compare their results with archived values from earlier stages of the
project ("anclas"); the files they read for that (`E9`, `E15`, `E16`, `E22`, `E27`, `E28`,
`E44b`) are included in `resultados/`. Two checks of E42 against earlier stages that are
not part of the paper (E20 and E41) are skipped when it is rerun here; the archived
`E42_escalera_N.json` records them as passed. The docstrings refer to a lab notebook
(`docs/bitacora/`) where each design and prediction was declared before running; the
notebook is not part of this repository.

## Stages

**Models (simulation).**

| Stage | Script | What it measures |
|---|---|---|
| E42 | `e42_escalera_N.py` | Recovery of the oracle's advantage against flock size, for a memoryless and an adapted observer |
| E42b | `e42b_barrido_ventana.py` | Sweep of the observer's window on the long side |
| E43, E43b | `e43_escalera_tauW.py`, `e43b_tauW_N1000.py` | Recovery against the decay time of the intention, τ_W (N = 250 and 1000) |
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

**Animal data.**

| Stage | Script | Data | What it measures |
|---|---|---|---|
| E51, E51b | `e51_huella_aves.py`, `e51b_inyeccion.py` | Jackdaws, turning events | Footprint without a fixed threshold; known-truth injection control |
| E52, E52b | `e52_auc_ventana_aves.py`, `e52b_aves_por_grabacion.py` | Jackdaws, turning events | Initiator/follower AUC against the window; recording as resampling unit |
| E53 | `e53_agilidad_calibracion.py` | Jackdaws, continuous recordings | Agility and calibration of the observer |
| E54, E54b, E54c | `e54_peces.py`, `e54b_prospectivo.py`, `e54c_tetras_por_experimento.py` | Tetras | Footprint and AUC against the window; prospective control; per experiment |
| E57, E57b | `e57_palomas.py`, `e57b_potencia.py` | Homing pigeons | Manipulated leader; power by injection of model traces |

**Shared modules.** `flocking.py` (2D engine: ViSt, CoLd, CaSp, WiSt with an oracle or a
Bayesian observer), `parametros.py` (reference parameters `P0` and the compared mechanisms),
`fase4.py` (common harness: runs, pairing, recovery and intervals), `e15_helpers.py`
(footprint in the model), `e16_helpers.py` (kinematic residual on real tracks), `turns.py`,
`jackdaws3d.py`, `e27_tetras.py` and `watts_palomas.py` (data loaders).

## License

Code and results: MIT (see `LICENSE`). The animal data are not redistributed here; each
data set keeps its own license.

## Citation

See `CITATION.cff`. Each release of this repository is archived in Zenodo with its own DOI.
