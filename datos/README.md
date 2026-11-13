# Animal data

The animal data are third-party and are not redistributed here. Download each set from its
source and place it in this folder with the layout below; the loaders in `codigo/` read
these paths. Only the stages E51 to E57 need them.

| Data | Source | Place it as | Used by |
|---|---|---|---|
| Jackdaws, 21 curated turning events (Ling et al. 2019, *J. R. Soc. Interface* 16:20190450) | figshare `10.6084/m9.figshare.10011374` | `datos/giros/code/turns_transit/group_*.mat`, `group_delay_*.mat`; `datos/giros/code/turns_mobbing/…`; `datos/giros/code/Info_turnStart_T.mat` and `Info_turnStart_M.mat` | `turns.py`: E51, E51b, E52, E52b, E53 |
| Jackdaws, continuous recordings (Ling et al. 2019, *Nat. Commun.* 10:5174) | Supplementary Data 2 to 4 of the article | `datos/ling2019_natcomm/DataS2.txt`, `DataS3.txt`, `DataS4.txt` | `jackdaws3d.py`: E53 |
| Tetras, *Hemigrammus rhodostomus* (Lecheval et al. 2018) | Dryad `10.5061/dryad.9m6d2` | `datos/lecheval_tetras/<group size>/*.csv`, with the folders `4`, `5`, `8` and `10` | `e27_tetras.py`: E54, E54b, E54c |
| Homing pigeons with a manipulated leader (Watts et al. 2016) | Dryad `10.5061/dryad.508j2` | `datos/watts_palomas/Data/*.csv` | `watts_palomas.py`: E57, E57b |

The parameters of the reference model come from Balázs, Vásárhelyi & Vicsek (2020) and
Mendeley `10.17632/3brbybg4nk.1`; they are written in `codigo/flocking.py` and
`codigo/parametros.py`, so that data set is not needed to run the code.
