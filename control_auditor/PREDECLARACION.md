# Controles de memoria y ganancia del observador

Fecha: 12 de noviembre de 2026. Diseño interno fijado antes de evaluar las nuevas respuestas de giro; no es un registro externo. Se usa el motor congelado dentro de esta carpeta y todas las escrituras quedan en prioridad3. El paper y sus fuentes originales no son destinos de escritura.

## Pregunta y estimando principal

¿Una memoria adicional de 1 s mejora la responsividad respecto de la misma evidencia instantánea cuando se fija su ganancia o se equiparan las tasas basales de atención? Se conserva el mismo filtro de entrada tau_obs=0.083 s: instantáneo significa ausencia de acumulador adicional. No se afirma que la política oráculo sea un óptimo ni que rendimiento equivalga a identificar información privada.

La comparación principal usa el mismo log-cociente Student-t completo, antes de cualquier binarización. Su media exponencial con constante tau se multiplica por alpha, independiente de tau. El peso es logistic(alpha*score+b), con intercepto total b. Una rama binaria sin acumulador frente a la misma evidencia binaria con memoria de1s es un control secundario. No se aplica automáticamente tau/tau_obs. El autoconocimiento de la voluntad propia y el resto de la política son iguales entre condiciones.

## Condiciones fijadas

N=250, k=15, v0=8, amax=10, tau_c=0.2, ruido Wiener sqrt(2), dt=0.01, sigma_theta=0.361, sigma_W=0.811, nu=2.8, pi0=0.173. Dos protocolos principales: líder forzado y maniobra única, tauW=4s, horizonte15.8s, 18 ángulos medios cada10grados. Cada réplica es la media de los18ángulos.

Se estudia memoria tau={0,0.25,1,4}s y ganancia alpha={1,1/0.083,48}, con b=logit(0.173), sin seleccionar la ganancia por su respuesta. Las celdas centrales Student tau0y1 aalpha=1/0.083 tendrán8réplicas; las otras10celdas del factorial tendrán4. El control binario tau0y1 a esa ganancia tendrá8. La rejilla es secundaria y no se interpreta como localización precisa de un óptimo físico.

Las condiciones centrales Student tau0y1 se repiten bajo tres calibraciones sin información de la tarea: FPR basal equiparada, peso medio basal equiparado, y ambas magnitudes si el ajuste conjunto es alcanzable. Cada pareja tendrá8réplicas en ambos protocolos. La validación de tauW=1s usa maniobra única, Student tau0y1 con ganancia fija y con ajuste conjunto, más sus referencias,8réplicas. Se conservarán todos los resultados.

## Calibración y separación de muestras

Objetivos operacionales definidos por diseño: FPR=0.05 para peso>0.5 en simulación sin informados; media de pesos=0.173. No se interpretan como tasas biológicas medidas. FPR es ocupación agente-tiempo, no episodios por segundo.

En FPR-match se mantiene alpha=1/0.083 y se ajusta b mediante cuantiles nulos. En mean-match se mantiene esa alpha y se ajusta b mediante la media logística. En joint-match se ajustan alpha y b a ambos objetivos. Los empates del score binario pueden impedir alcanzar FPR5% con un umbral determinista: no se simulará una igualdad inexistente; la rama binaria queda como contraste secundario a igual ganancia.

Se usan registros sin informados para calibración: semillas900000..900003. Se permite inicialización con una fuente común ViSt y hasta4actualizaciones de calibración sobre el modelo cerrado; ninguna respuesta a un giro se usa para fijar parámetros. Después se congelan. La validación nula independiente usa910000..910007; no se usa para volver a ajustar los parámetros. Las respuestas nuevas usan920000+1000*índice_ángulo+réplica. Las réplicas0..7 son comunes entre condiciones; las celdas con4 usan0..3. Estas semillas no son las de E58.

Se medirán las tasas efectivamente obtenidas y su incertidumbre por semilla. La igualdad operacional sólo se aceptará si el intervalo95% de la diferencia pareada entre las dos memorias queda dentro de ±0.01 en FPR y/o ±0.015 en atención, según el control. También se mostrarán las desviaciones de cada objetivo. Si no se alcanza el emparejamiento, se reportará ese control como aproximado o fallido y no como prueba de igualdad.

## Asentamiento y diagnósticos

Asentamiento común inicial12s. Las memorias principales de1s tienen12constantes de tiempo. Para tau4s, exploratoria, se compara primero el régimen nulo con12y24s usando las semillas de entrenamiento; si la diferencia media de FPR supera0.01 o de atención0.015 se usará24s para la comparación afectada y su referencia, declarando el cambio antes de mirar respuestas. Se registrarán estado inicial y actualización por paso; los anclajes con el motor original usarán su asentamiento de3s exclusivamente para comprobar equivalencia.

Por corrida se guardan R, R en primeros4s, diagnósticos de pesos sobre líder y no líderes, tiempo de peso>0.5 y configuración efectiva. Se añaden R absolutas de CoLd/ViSt y oráculo, diferencias y denominadores. La FPR basal independiente es el criterio de equiparación; durante un giro la atención de seguidores se describe sin confundirla con alarmas biológicas.

## Contrastes y reglas de interpretación

Los ocho contrastes principales son R(Student tau1)−R(Student tau0), para las cuatro reglas {gananciafija,FPR-match,mean-match,joint-match} y los dos protocolos tauW4. Se usarán10 000remuestreos pareados por réplica y una banda simultánea bootstrap sobre esos ocho contrastes, además de intervalos individuales y diferencias en recuperación. Si joint no es factible, se conserva como intento fallido y no se sustituye retrospectivamente por una condición favorable. La muestra pequeña de8réplicas limita la precisión: los intervalos son condicionales a este motor y a los parámetros calibrados.

Si la memoria mantiene ventaja con tasas basales efectivamente equiparadas, queda respaldada su contribución para ese estadístico y régimen. Si una condición instantánea equiparada la iguala, se acota la interpretación de memoria. Si los intervalos no separan las alternativas, el resultado se declara inconcluso. No se exige obtener un signo favorable para considerar realizada la prueba.

La validación tauW1 y el factorial de4réplicas son secundarios. No se declarará equivalencia por falta de significación ni se generalizará a todas las ganancias, especies o arquitecturas de percepción. La causalidad animal, el acceso sensorial estricto y qué fija la escala de la huella permanecen fuera de esta campaña.
