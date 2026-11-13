# Prioridad 3 Memoria frente a ganancia de evidencia

Fecha: 12 de noviembre de 2026. Todas las simulaciones, fuentes congeladas, configuraciones, calibraciones y resultados nuevos se guardan dentro de esta carpeta. Este trabajo no edita el paper ni descarga datos.

**Resultado principal.** Con falsos positivos y atención basal equiparados dentro de los márgenes predefinidos, la política con memoria de 1 s obtiene mayor responsividad en los dos protocolos principales. Ese control operacional no elimina la ventaja; no implica igualdad exacta de las tasas ni de toda la distribución de pesos. El control conjunto ajusta también la ganancia y no estima el efecto de cambiar sólo la memoria; ese efecto corresponde al contraste a igual ganancia e intercepto.

La ventaja no es universal: maniobra, falsos positivos equiparados: diferencia R -0.0373 [-0.0483, -0.0262]. En esa comparación gana el instantáneo.

La magnitud también depende de la calibración: en maniobra, el temporal con ajuste conjunto recupera 6.86 % de la ventaja del oráculo, frente a 56.51 % con parámetros fijos.

## Qué se puso a prueba

Se comparó un observador que usa la evidencia Student-t completa en cada instante con otro que promedia exactamente la misma evidencia durante 1 s. La ganancia multiplicativa y el intercepto logístico se separaron de la memoria. La entrada conserva el mismo suavizado de 0.083 s: instantáneo significa sin acumulador adicional.

El score instantáneo es el log-cociente entre las dos densidades Student-t del residual cinemático. Para tau > 0 se usa S(t) = a S(t−dt) + (1−a) score(t), con a = exp(−dt/tau); para tau = 0 se usa el score directamente. El peso es logistic(b + alpha S), sin multiplicador automático tau/tau_obs. La memoria Student empieza en cero y la binaria en su probabilidad nula q0; ambas reciben el asentamiento común. La voluntad propia sigue siendo conocida por cada agente.

El resultado principal es la diferencia de responsividad del grupo, no una prueba de identificación de información privada. Se calcula R como el promedio temporal de la velocidad media de los 250 agentes proyectada sobre la dirección objetivo, dividido por v0; incluye al líder. La recuperación es (R − R_base)/(R_oráculo − R_base), se refiere a una política oráculo concreta y puede superar 100 %.

## Diseño realizado

Se completaron 5760 corridas nuevas de giro, agrupadas en 50 condiciones y 320 medias por réplica y condición. Cada réplica resume 18 ángulos; las condiciones principales tienen 8 réplicas pareadas y el factorial complementario 4. No se trata cada ángulo como una réplica independiente.

Parámetros de referencia: N=250, k=15, ruido Wiener de intensidad numérica sqrt(2), dt=0.01 s. Protocolos principales: líder forzado y maniobra única, tauW=4 s, horizonte 15.8 s. La maniobra con tauW=1 s es una validación secundaria. Se recalcularon CoLd/ViSt y oráculo con las mismas semillas y el mismo asentamiento; no se reutilizaron denominadores de las corridas antiguas con asentamiento distinto.

La predeclaración interna se escribió antes de medir las respuestas nuevas. La adenda precisó que la calibración cuenta aristas de percepción sin autoenlaces y que la ganancia central efectiva es 12.048. Se conservan ambas decisiones, la revisión del diseño y las calibraciones preliminares sustituidas. No hay registro externo ni cohorte biológica nueva.

## Calibración independiente de la respuesta

Se ajustaron parámetros sólo con cuatro semillas de simulaciones sin informados. Los registros nulos tienen 12 s de asentamiento y 12 s de medición. Luego se congelaron y se validaron en ocho semillas nuevas 911000–911007; no se reajustaron usando esa validación ni las respuestas al giro. La primera prueba por emisor quedó archivada como preliminar, y su validación anterior no se reutilizó como validación final.

El estimando de equiparación es la distribución de pesos sobre aristas dirigidas receptor–emisor, excluyendo el propio agente. La tasa de falso positivo es la fracción de pesos mayores de 0.5 en el nulo, no una tasa de episodios de alarma por segundo. La atención es el peso medio sobre esas aristas. Objetivos de diseño: 0.05 y 0.173, respectivamente; no son prevalencias animales medidas.

| Esquema | Memoria s | Factor alpha | Intercepto total | FPR validada [IC95] | Atención validada [IC95] |
|---|---:|---:|---:|---|---|
| Falsos positivos equiparados | 0 | 12.04800 | -6.44984 | 0.04919 [0.04798, 0.05037] | 0.05245 [0.05137, 0.05350] |
| Atención media equiparada | 0 | 12.04800 | -0.32828 | 0.16492 [0.16282, 0.16703] | 0.17031 [0.16873, 0.17211] |
| Ambas magnitudes equiparadas | 0 | 1.82105 | -0.98522 | 0.04931 [0.04814, 0.05052] | 0.17208 [0.17137, 0.17284] |
| Falsos positivos equiparados | 1 | 12.04800 | 3.63821 | 0.04665 [0.04400, 0.04919] | 0.19295 [0.18973, 0.19596] |
| Atención media equiparada | 1 | 12.04800 | 3.46286 | 0.03352 [0.03158, 0.03543] | 0.17046 [0.16750, 0.17326] |
| Ambas magnitudes equiparadas | 1 | 13.86051 | 4.18690 | 0.04661 [0.04390, 0.04916] | 0.17024 [0.16689, 0.17344] |

La comparación FPR-match deja libre la atención y la comparación mean-match deja libre la FPR. El ajuste joint controla las dos; requiere ganancias distintas porque las distribuciones de la evidencia instantánea y temporal son diferentes. No se denomina igualdad exacta: se exige que los intervalos de diferencias pareadas queden dentro de ±0.01 para FPR y ±0.015 para atención, además de comprobar cada objetivo.

Las tres parejas cumplen sus criterios operacionales: sí. La comprobación numérica completa está en [CALIBRACION_Y_EQUIVALENCIA.md](CALIBRACION_Y_EQUIVALENCIA.md). Los intervalos son condicionales a los parámetros entrenados y a este motor.

![Validación nula de la calibración](<figuras/validacion_nula.png>)

## Contrastes principales de memoria

La tabla muestra memoria 1 s menos evidencia instantánea. Los intervalos simultáneos controlan conjuntamente los ocho contrastes definidos antes de las respuestas, mediante el percentil 95 del máximo desvío bootstrap centrado. Se remuestrean 8 réplicas completas y pareadas 10 000 veces. Es una aproximación bootstrap con pocas semillas, no una garantía exacta de cobertura. Las cifras se redondean por separado; las diferencias se calculan antes de redondear.

| Protocolo tauW 4 s | Control | R instantáneo | R memoria | Diferencia R [IC95 simultáneo] | Diferencia de recuperación, puntos porcentuales [IC95 individual] |
|---|---|---:|---:|---|---|
| forzado | Misma ganancia e intercepto | 0.1413 | 0.2997 | +0.1585 [0.1475, 0.1695] | 54.01 [50.23, 57.36] |
| forzado | Falsos positivos equiparados | 0.1438 | 0.2186 | +0.0748 [0.0638, 0.0858] | 25.50 [23.28, 27.44] |
| forzado | Atención media equiparada | 0.1368 | 0.2331 | +0.0963 [0.0852, 0.1073] | 32.81 [30.54, 34.74] |
| forzado | Ambas magnitudes equiparadas | 0.1105 | 0.2414 | +0.1309 [0.1198, 0.1419] | 44.60 [42.42, 46.17] |
| maniobra | Misma ganancia e intercepto | 0.0613 | 0.2005 | +0.1392 [0.1282, 0.1502] | 53.17 [50.67, 55.56] |
| maniobra | Falsos positivos equiparados | 0.0989 | 0.0616 | -0.0373 [-0.0483, -0.0262] | -14.23 [-15.43, -13.01] |
| maniobra | Atención media equiparada | 0.0532 | 0.0691 | +0.0159 [0.0048, 0.0269] | 6.07 [4.84, 7.22] |
| maniobra | Ambas magnitudes equiparadas | 0.0402 | 0.0706 | +0.0304 [0.0194, 0.0414] | 11.61 [10.23, 13.05] |

![Ocho contrastes principales de memoria](<figuras/contrastes_principales.png>)

### Lectura por control

**Misma ganancia e intercepto.** forzado: la memoria conserva una ventaja resuelta en la banda simultánea; maniobra: la memoria conserva una ventaja resuelta en la banda simultánea.

**Falsos positivos equiparados.** forzado: la memoria conserva una ventaja resuelta en la banda simultánea; maniobra: el instantáneo obtiene mayor responsividad en la banda simultánea.

**Atención media equiparada.** forzado: la memoria conserva una ventaja resuelta en la banda simultánea; maniobra: la memoria conserva una ventaja resuelta en la banda simultánea.

**Ambas magnitudes equiparadas.** forzado: la memoria conserva una ventaja resuelta en la banda simultánea; maniobra: la memoria conserva una ventaja resuelta en la banda simultánea.


## Recuperación y valores absolutos

| Protocolo | tauW | Política | R [IC95] | R base | R oráculo | Recuperación % [IC95] |
|---|---:|---|---|---:|---:|---|
| forzado | 4 | Instantáneo, parámetros fijos | 0.1413 [0.1341, 0.1491] | 0.0852 | 0.3786 | 19.11 [17.41, 21.00] |
| forzado | 4 | Instantáneo, ajuste conjunto | 0.1105 [0.1059, 0.1154] | 0.0852 | 0.3786 | 8.63 [8.06, 9.27] |
| forzado | 4 | Memoria 1 s, parámetros fijos | 0.2997 [0.2923, 0.3071] | 0.0852 | 0.3786 | 73.12 [70.08, 76.22] |
| forzado | 4 | Memoria 1 s, ajuste conjunto | 0.2414 [0.2324, 0.2490] | 0.0852 | 0.3786 | 53.23 [50.86, 55.08] |
| maniobra | 1 | Instantáneo, parámetros fijos | 0.0434 [0.0386, 0.0488] | 0.0196 | 0.1266 | 22.23 [20.54, 23.86] |
| maniobra | 1 | Instantáneo, ajuste conjunto | 0.0299 [0.0260, 0.0342] | 0.0196 | 0.1266 | 9.61 [8.85, 10.25] |
| maniobra | 1 | Memoria 1 s, parámetros fijos | 0.1321 [0.1284, 0.1358] | 0.0196 | 0.1266 | 105.19 [103.07, 107.46] |
| maniobra | 1 | Memoria 1 s, ajuste conjunto | 0.0488 [0.0423, 0.0555] | 0.0196 | 0.1266 | 27.24 [24.02, 30.49] |
| maniobra | 4 | Instantáneo, parámetros fijos | 0.0613 [0.0561, 0.0670] | 0.0526 | 0.3144 | 3.34 [2.49, 4.20] |
| maniobra | 4 | Instantáneo, ajuste conjunto | 0.0402 [0.0361, 0.0446] | 0.0526 | 0.3144 | -4.75 [-5.14, -4.44] |
| maniobra | 4 | Memoria 1 s, parámetros fijos | 0.2005 [0.1980, 0.2030] | 0.0526 | 0.3144 | 56.51 [54.68, 58.23] |
| maniobra | 4 | Memoria 1 s, ajuste conjunto | 0.0706 [0.0632, 0.0781] | 0.0526 | 0.3144 | 6.86 [5.23, 8.51] |

Si el IC95 del denominador alcanza cero, no se informa un IC de recuperación como estable. Una media del denominador no positiva deja también el punto de recuperación sin definir. Esta regla se fijó antes de ejecutar el análisis final; no afecta la diferencia R principal.

Los CSV incluyen todos los factores, interceptos, denominadores y diagnósticos sobre líder y no líderes; la tabla anterior selecciona las parejas centrales fijadas por el diseño, no una selección posterior por éxito.

Condiciones con intervalo de recuperación marcado como no estable: 0 de 50.


## Atención durante el giro: diagnóstico descriptivo

Se muestran las parejas a igual ganancia y con ajuste conjunto. Son promedios durante los 15.8 s de respuesta, no nuevas restricciones de calibración. Los intervalos individuales se calculan por réplica; el líder puede dejar de estar informado durante la maniobra, por lo que su peso no es una sensibilidad diagnóstica frente a una etiqueta fija.

| Protocolo tauW 4 s | Política | Peso medio recibido [IC95] | Peso recibido del líder por agente [IC95] | Fracción del peso destinada al líder [IC95] |
|---|---|---|---|---|
| forzado | Instantáneo, parámetros fijos | 0.14074 [0.14045, 0.14101] | 0.01260 [0.01229, 0.01288] | 0.00591 [0.00576, 0.00604] |
| forzado | Instantáneo, ajuste conjunto | 0.17260 [0.17250, 0.17271] | 0.01163 [0.01153, 0.01173] | 0.00448 [0.00444, 0.00452] |
| forzado | Memoria 1 s, parámetros fijos | 0.00271 [0.00265, 0.00276] | 0.00540 [0.00519, 0.00560] | 0.05321 [0.05154, 0.05469] |
| forzado | Memoria 1 s, ajuste conjunto | 0.17569 [0.17514, 0.17622] | 0.02271 [0.02208, 0.02340] | 0.00849 [0.00827, 0.00872] |
| maniobra | Instantáneo, parámetros fijos | 0.13923 [0.13899, 0.13945] | 0.01245 [0.01215, 0.01271] | 0.00593 [0.00578, 0.00607] |
| maniobra | Instantáneo, ajuste conjunto | 0.17249 [0.17239, 0.17259] | 0.01264 [0.01248, 0.01278] | 0.00487 [0.00480, 0.00492] |
| maniobra | Memoria 1 s, parámetros fijos | 0.00245 [0.00242, 0.00247] | 0.00506 [0.00490, 0.00521] | 0.05730 [0.05583, 0.05897] |
| maniobra | Memoria 1 s, ajuste conjunto | 0.17248 [0.17222, 0.17275] | 0.02022 [0.01968, 0.02068] | 0.00773 [0.00752, 0.00791] |

## Validación secundaria con tauW de 1 s

| Control | R instantáneo | R memoria | Diferencia R e IC95 individual |
|---|---:|---:|---|
| Misma ganancia e intercepto | 0.0434 | 0.1321 | +0.0887 [0.0862, 0.0913] |
| Ambas magnitudes equiparadas | 0.0299 | 0.0488 | +0.0188 [0.0159, 0.0219] |

Esta validación no se combina con la familia principal de ocho contrastes. Sus intervalos son descriptivos y no se usan para elegir qué condiciones conservar.

## Mismo estadístico binario con y sin memoria

Las dos ramas usan el mismo indicador de excedencia, la misma ganancia 12.048 y el mismo intercepto. Así se evita atribuir a memoria un cambio simultáneo de amplitud a ciclo. Los contrastes de este bloque son secundarios y sus intervalos no tienen la corrección de la familia principal.

| Protocolo tauW 4 s | Estadístico | R instantáneo | R memoria 1 s | Diferencia R [IC95 individual] |
|---|---|---:|---:|---|
| forzado | full | 0.1413 | 0.2997 | +0.1585 [0.1468, 0.1689] |
| forzado | binary | 0.1404 | 0.2934 | +0.1530 [0.1392, 0.1642] |
| maniobra | full | 0.0613 | 0.2005 | +0.1392 [0.1327, 0.1451] |
| maniobra | binary | 0.0675 | 0.1899 | +0.1224 [0.1186, 0.1263] |

## Controles complementarios y límites

El factorial Student incluye memorias 0, 0.25, 1 y 4 s y factores 1, 12.048 y 48 con intercepto fijo. Los controles binarios comparan el mismo indicador de excedencia con y sin memoria a igual factor. Son complementarios; no se fuerza una FPR del 5 % en un score instantáneo binario que sólo admite dos valores deterministas.

La rejilla secundaria se acotó antes de las respuestas a cuatro ventanas; no se ejecutaron las ventanas de 0.5 y 2 s del bosquejo inicial del plan. La campaña resuelve los controles centrales de memoria frente a ganancia, pero no completa una búsqueda fina de la ventana óptima.

![Factorial complementario de memoria y ganancia](<figuras/factorial_memoria_ganancia.png>)

El factorial muestra una interacción: aumentar la memoria no sustituye cualquier ganancia. Con alpha=1, el rendimiento cambia poco entre ventanas y permanece bajo; con ganancias mayores aparecen diferencias temporales claras en las medias. Subir la ganancia instantánea a 48 no reproduce el rendimiento de la memoria de 1 s a esa misma ganancia en ninguno de los dos protocolos. Es una lectura descriptiva de la rejilla de cuatro réplicas, sin afirmar un óptimo universal ni localizar con precisión la ventana.

La prueba nula de asentamiento comparó 12 y 24 s para tau de 4 s, tres ganancias y cuatro semillas. Diferencias máximas de medias: FPR=0.000000, atención=0.000567. Activó la regla de 24 s: no. Ese test es una sensibilidad al asentamiento, no una demostración de estacionariedad completa.

El diagnóstico de atención se muestrea cada 0.1 s; R se integra cada 0.01 s. Los pesos durante el giro no tienen por qué conservar sus tasas nulas: esa diferencia forma parte de la respuesta al estímulo. La calibración y las incertidumbres están verificadas en el nulo, y no deben extrapolarse a tasas biológicas de falsas alarmas.

La ganancia, el intercepto y la memoria definen una política de atención. La calibración logística no prueba que el peso sea una probabilidad posterior bien calibrada de voluntad informada. Se conservan la memoria por emisor y el acceso cinemático del motor anterior; no se resuelve aquí la factibilidad de percepción local ni el reconocimiento animal de información privada.

Equiparar ocupación por encima de 0.5 y peso medio no iguala toda la distribución de pesos ni la duración de las rachas. Cada política también actúa durante su asentamiento: el contraste estima su efecto total en lazo cerrado, incluidas las diferencias de estado inicial inducidas por ella. La incertidumbre no integra la búsqueda de parámetros ni variación entre arquitecturas del modelo.

## Objeciones cerradas y pendientes

Respecto del dictamen previo, se atiende el componente memoria–ganancia de la objeción mayor 4, se documenta la especificación de estos controles para la objeción 1 y se mejora la incertidumbre condicional de la objeción 6. No quedan cerradas por completo esas objeciones: H3, la incertidumbre estructural y la incorporación de los métodos al PDF siguen pendientes. El cierre que se informa aquí pertenece al expediente de resultados.

| Objeción | Estado alcanzado |
|---|---|
| Cambiar tau cambia a la vez el multiplicador automático de evidencia | Separado en el módulo nuevo: alpha y tau son entradas independientes; se conserva un factorial y el contraste a igual alpha |
| Amplitud instantánea frente a ciclo cambia también el estadístico | Se compara Student completo con Student completo y, como control secundario, binario con binario |
| La diferencia podría atribuirse a mayor tasa basal de pesos altos | Control ejecutado y validado. En maniobra, FPR-match favorece al instantáneo; no se sostiene superioridad temporal en toda calibración. El ajuste conjunto sí favorece al temporal en ambos protocolos |
| La diferencia podría atribuirse sólo a mayor peso medio | Se calibra y valida mean-match, además de joint con las dos magnitudes |
| Los parámetros se optimizaron mirando R o los mismos giros de evaluación | Se separaron entrenamiento nulo, validación nula y semillas nuevas de giro; parámetros congelados antes de validar y responder |
| El resultado sólo tiene cuatro réplicas del barrido antiguo | Los contrastes centrales nuevos tienen ocho réplicas independientes del entrenamiento; el factorial sigue con cuatro y se identifica como complementario |
| La memoria es una explicación universal o única de todo reconocimiento | Sigue abierta: la inferencia se limita al motor, estadístico, ganancias y protocolos probados |
| La mejor ventana está determinada causalmente por una escala social | Sigue abierta; este factorial no identifica esa escala ni una duración óptima precisa |
| Los animales reconocen información privada por este mecanismo | Sigue abierta; no se añadieron observaciones ni intervenciones animales |
| El PDF ya incorpora los cierres | Pendiente de edición; el paper no es destino de escritura de esta campaña |

La existencia de un control y la dirección que éste revela son cuestiones distintas. Las objeciones de confusión se responden con el diseño y los resultados, aun si una comparación no favorece a la memoria. No se interpreta falta de diferencia resuelta como equivalencia de rendimiento.

## Qué falta para reforzar la atribución

No se necesitan descargas para los siguientes controles computacionales. En orden: repetir el entrenamiento con cohortes independientes para incluir incertidumbre de calibración; equiparar la distribución nula completa de pesos, además de su media y ocupación, y estudiar las rachas; comparar estímulos desde estados iniciales comunes para separar el efecto del asentamiento; intervenir escalas sociales y de maniobra para probar H3. Son extensiones pendientes, no resultados de esta campaña. Una afirmación de reconocimiento de información privada en animales requiere además etiquetas o intervenciones adecuadas y evaluación causal; estas simulaciones no las sustituyen.

## Evidencia y reproducción

- [Predeclaración](PREDECLARACION.md), [precisión del estimando](ADENDA_ESTIMANDO.md), [criterio para cocientes](CRITERIO_COCIENTES.md) y [revisión independiente del diseño](REVISION_DISENO.md).
- `calibracion/`: búsqueda nula, parámetros congelados, validación y equivalencia pareada; se conserva la fase preliminar por emisor.
- `resultados/corridas/`: cada ensayo con semilla, ángulo, política, parámetros y diagnóstico. Son los crudos nuevos; no se sobrescribieron E58 ni sus antecedentes.
- `resultados/RESULTADOS_MEMORIA_GANANCIA.json`, `resumen_condiciones.csv`, `contrastes_principales.csv`, `respuestas_por_replica.csv` y `respuestas_por_angulo.csv`.
- `piloto/`: anclas de equivalencia con el motor histórico, anclaje adicional del secuencial y test de asentamiento; `evidencia/`: hashes de fuentes, configuración efectiva y comparación de originales.
- [Configuración efectiva y reconstrucción](evidencia/PARAMETROS_EFECTIVOS.json): el bloque `result.params` de cada crudo describe los parámetros comunes previos al enrutamiento; para ViSt/CoLd/oráculo, `new_flock` fuerza `will_mode=oracle`. Los infinitos del JSON crudo se guardan como null; su significado se conserva en esa configuración. Reproducir desde `job` y las funciones guardadas evita interpretar esos campos como una política distinta.
- Los scripts de simulación reanudan checkpoints compatibles y detienen la ejecución si cambian las fuentes. El README de esta carpeta enumera los comandos y el alcance de cada archivo.

El informe se refiere al expediente nuevo. Cualquier cambio concurrente en el manuscrito original se registra aparte y no se restaura automáticamente.

## Auditoría final de cifras e integridad

Se verificaron individualmente las 5760 corridas: R y sus ventanas de 4 s se reconstruyen a partir de la serie guardada, las medias por condición coinciden con los crudos, las recuperaciones usan sus denominadores correspondientes y los ocho contrastes coinciden entre JSON y CSV. Discrepancias por encima de 2 × 10⁻¹²: ninguna. El mayor error de reconstrucción de R es 2.22e-16.

Se mantienen idénticos 409 de los 409 archivos originales inventariados. Cambiados: 0; ausentes: 0. El inventario incluye el PDF y las fuentes del paper corto, el código, resultados y documentos previos. No se descargaron datos.

Las verificaciones y sus límites están en [VERIFICACION_FINAL.json](evidencia/VERIFICACION_FINAL.json). Se revisaron las tres figuras finales; el registro está en [REVISION_VISUAL.md](evidencia/REVISION_VISUAL.md).

Además se contrastaron 328 valores impresos en las seis tablas del MD (40 filas) con sus salidas numéricas: 0 discrepancias a la precisión decimal declarada. Registro: [AUDITORIA_TABLAS_MD.json](evidencia/AUDITORIA_TABLAS_MD.json).
