# Animal data

The animal data are third-party and are not redistributed here. Download each set from its
source and place it in this folder with the layout below; the loaders in `codigo/` read
these paths. Only the stages with animal data need them.

| Data | Source | Place it as | Used by |
|---|---|---|---|
| Zebrafish, five trained and ten desensitised individuals (Romero-Ferrero et al. 2023, *Phil. Trans. R. Soc. B* 378:20220073) | Trajectories published by the authors: GitLab `polavieja_lab/royalsocietypaper` links a public folder with `all_sessions_trials_stimstart` and `all_sessions_trials_intertrial` | `datos/polavieja2023/stimstart/*session-11*_stimstart.npy` and `datos/polavieja2023/intertrial/*session-11*_intertrial.npy` (only session 11, the mixed groups) | `carga_segura.py`: E73, E73b |
| Jackdaws, 21 curated turning events (Ling et al. 2019, *J. R. Soc. Interface* 16:20190450) | figshare `10.6084/m9.figshare.10011374` | `datos/giros/code/turns_transit/group_*.mat`, `group_delay_*.mat`; `datos/giros/code/turns_mobbing/…`; `datos/giros/code/Info_turnStart_T.mat` and `Info_turnStart_M.mat` | `turns.py`: E51, E51b, E52, E52b, E53 |
| Jackdaws, continuous recordings (Ling et al. 2019, *Nat. Commun.* 10:5174) | Supplementary Data 2 to 4 of the article | `datos/ling2019_natcomm/DataS2.txt`, `DataS3.txt`, `DataS4.txt` | `jackdaws3d.py`: E53 |
| Tetras, *Hemigrammus rhodostomus* (Lecheval et al. 2018) | Dryad `10.5061/dryad.9m6d2` | `datos/lecheval_tetras/<group size>/*.csv`, with the folders `4`, `5`, `8` and `10` | `e27_tetras.py`: E54, E54b, E54c |
| Sticklebacks (Jolles et al. 2017, *Curr. Biol.* 27:2862) | Cambridge Apollo `10.17863/CAM.12136` | `datos/jolles_espinosos/B-cm1-id-full.csv` | E35 |
| Homing pigeons with a manipulated leader (Watts et al. 2016) | Dryad `10.5061/dryad.508j2` | `datos/watts_palomas/Data/*.csv` | `watts_palomas.py`: E57, E57b |
| Homing pigeons pursued by a robotic falcon (Papadopoulou et al. 2022, *R. Soc. Open Sci.* 9:211898) | Dryad `10.5061/dryad.905qfttkc` | `datos/palomas/all_emp_data.csv` | E30 |

The zebrafish files are pickled Python objects; read them with `codigo/carga_segura.py`
rather than with `numpy.load(..., allow_pickle=True)`. Their repository states no licence.

The parameters of the reference model come from Balázs, Vásárhelyi & Vicsek (2020) and
Mendeley `10.17632/3brbybg4nk.1`; they are written in `codigo/flocking.py` and
`codigo/parametros.py`, so that data set is not needed to run the code. `codigo/validar_vs_dataset.py`
carries the published responsivities of its file `v-w-resp-kscan-250A-sqrttime.p` and, when the
data set is unpacked in `datos/persistence-responsivity trade off/`, checks them against it.
