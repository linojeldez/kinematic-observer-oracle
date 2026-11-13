# Calibración y equivalencia nula

Parámetros ajustados sólo en cuatro semillas de entrenamiento sin individuos informados. Validación en ocho semillas nuevas, sin reajuste. Cada observación es una arista dirigida sin autoenlace; los intervalos remuestrean semillas completas, no aristas. FPR es fracción de pesos mayores de0.5, no episodios de alarma por segundo.

| Esquema | Memoria s | Ganancia | Intercepto total | FPR [IC95] | Atención media [IC95] | Objetivo requerido cumplido |
|---|---:|---:|---:|---|---|---|
| fpr | 0 | 12.048000 | -6.449840 | 0.04919 [0.04798, 0.05037] | 0.05245 [0.05137, 0.05350] | True |
| mean | 0 | 12.048000 | -0.328276 | 0.16492 [0.16282, 0.16703] | 0.17031 [0.16873, 0.17211] | True |
| joint | 0 | 1.821045 | -0.985224 | 0.04931 [0.04814, 0.05052] | 0.17208 [0.17137, 0.17284] | True |
| fpr | 1 | 12.048000 | 3.638213 | 0.04665 [0.04400, 0.04919] | 0.19295 [0.18973, 0.19596] | True |
| mean | 1 | 12.048000 | 3.462858 | 0.03352 [0.03158, 0.03543] | 0.17046 [0.16750, 0.17326] | True |
| joint | 1 | 13.860512 | 4.186899 | 0.04661 [0.04390, 0.04916] | 0.17024 [0.16689, 0.17344] | True |

Los esquemas FPR y mean equiparan sólo su magnitud objetivo. Sus otras magnitudes pueden diferir mucho. El esquema joint busca las dos simultáneamente. Las tolerancias son operacionales, fijadas antes de evaluar respuestas.

| Esquema | Diferencia FPR memoria−instantáneo [IC95] | Diferencia atención [IC95] | Equivalencia de la pareja en métricas exigidas |
|---|---|---|---|
| fpr | -0.00254 [-0.00441, -0.00075] | +0.14050 [+0.13759, +0.14305] | True |
| mean | -0.13140 [-0.13323, -0.12958] | +0.00015 [-0.00206, +0.00234] | True |
| joint | -0.00270 [-0.00471, -0.00082] | -0.00183 [-0.00465, +0.00081] | True |

El intercepto está calibrado como parte de una regla de atención; estos pesos logísticos no quedan validados automáticamente como probabilidades posteriores de información privada. Los intervalos son condicionales al entrenamiento y al motor.
