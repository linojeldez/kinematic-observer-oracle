# Revisión estadística del diseño: memoria y ganancia

Fecha: 12 de noviembre de 2026. Documento previo a las corridas confirmatorias de prioridad 3. Esta revisión no ejecutó simulaciones ni cambió el paper, sus fuentes o datos. Se apoya en el diseño comunicado por el coordinador y la implementación congelada de `fuentes/codigo/flocking.py`, `fase4.py` y `e58_memoria_ganancia.py`. No supone que controles todavía propuestos ya hayan pasado.

## Decisiones operativas comunicadas antes de R

El coordinador adopta f*=0.05 y μ*=0.173 en comparaciones separadas, más un emparejamiento conjunto sólo si resulta factible; el intercepto b es el intercepto total, sin sumar de nuevo el prior. Fija8 réplicas centrales,4 en la rejilla secundaria y validaciónτW1 sin recalibración. Propone12s de asentamiento común, S0=0 para full-LLR e indicador inicializado enq0; comparará12frente a24s en nulos sin usar R para las memorias largas y celdas principales. Los márgenes de validación serán±0.01 en FPR y±0.015 en atención. Debe fijarse la regla exacta que considere material el cambio de asentamiento antes de examinar esos nulos.

Para reducir coste, esta revisión recomienda que el bloque confirmatorio conserveτ0frente a1s bajo fixed-gain, FPR-match y mean-match; joint puede ser secundario. El barrido secundario puede reducirse aτ{0,0.25,1,4}s×α{1,12.048192771,48}, reutilizando las celdas/semillas ya presentes en el bloque central. La comparación de ciclo0frente a1s a igual factor sigue siendo un control secundario útil. Reducir así el barrido sacrifica resolución del máximo temporal, que no es el objetivo primario de este experimento. No se afirma aquí que el coordinador haya aceptado todos estos recortes ni que las calibraciones ya hayan validado.

## 1. Decisión recomendada

El diseño debe separar tres comparaciones: **misma ganancia**, **misma ocupación de falsos positivos** y **mismo peso posterior medio**. Las últimas dos se calibran sólo en registros sin individuos informados, antes de examinar la responsividad de evaluación. Igualar las tres cantidades simultáneamente no es un requisito inicial y puede ser imposible dentro de una rejilla restringida. Cada comparación responde a una objeción distinta.

La prueba primaria debe usar **el mismo log-cociente Student-t completo**, instantáneo frente a memoria exponencial. La comparación amplitud frente a ciclo es secundaria porque cambia también la estadística. E58 ya descarta una explicación basada únicamente en amplificar la amplitud entre los multiplicadores probados, pero no aísla por completo memoria, ganancia y tasa de alarma.

## 2. Políticas que deben fijarse antes del experimento

Sea ℓ_j(t) el log-cociente Student-t completo sin ganancia temporal incorporada. Definir:

- τ=0: S_j(t)=ℓ_j(t).
- τ>0: S_j⁺=exp(−dt/τ)S_j+(1−exp(−dt/τ))ℓ_j(t).
- Posterior: p_j=logistic(α S_j+b).

El factor α debe aplicarse **después del filtro** y antes del posterior; no multiplicar otra vez por τ/τobs dentro de S. Para los controles a igual ganancia, α∈{1,1/0.083,48} y b=logit(0.173). El valor central efectivo es12.048192771, no12 exactos. Registrar τ=0,0.25,0.5,1,2,4s sin llamar «memoria nula» a la ausencia del filtro angular de entrada: τobs=0.083s sigue existiendo.

Para el ciclo, usar el score binomial **sin n_ef dentro**, con indicador instantáneo B=1{|ε|>1.5σ}, o su EMA con τ=1s. Aplicar el mismo α por fuera. Esto compara ciclo de memoria mínima frente a ciclo con memoria usando la misma forma de evidencia. Mantener q0/q1, prior, calibración, dinámica y autoconocimiento propio idénticos entre esas dos políticas.

Deben quedar registrados el estado inicial del acumulador y el asentamiento. El banco histórico tiene3s; esta campaña propone12s comunes y horizonte15.8s. CoLd/ViSt y oráculo deben usar también12s: no reutilizar sus valores de los crudos históricos con3s al construir la recuperación de una campaña con distinto asentamiento. No afirmar que3s equilibraban una memoria de4s. El control nulo12frente a24s puede distinguir transitorio de integración sin usar R para seleccionar el asentamiento.

## 3. Qué se entiende por FPR y atención

En el nulo todos los agentes tienen w=0 y no hay líder forzado. La «FPR» propuesta es la **ocupación agente-tiempo de p>0.5**, no una tasa de episodios por minuto. Añadir como diagnóstico la tasa de cruces ascendentes del umbral; las dos cantidades no son intercambiables.

Para representar el peso que realmente entra a la política, medir las tasas sobre aristas dirigidas i→j de vecinos, **excluyendo el propio agente**. La atención media es

`mu = media_t,i [ media_{j entre k vecinos de i} p_ij(t) ]`.

La FPR se define con la misma media sustituyendo p por1{p>0.5}. La media uniforme por emisor también se puede informar, pero no equivale necesariamente al peso medio recibido: los emisores tienen distinto grado de entrada y éste puede correlacionarse con su score. No mezclar ambas definiciones entre calibración y validación.

## 4. Calibración factible

### A. FPR igualada

Sólo cambiar α con un prior fijo no permite alcanzar cualquier FPR. Con b<0, el umbral del score es−b/α>0; al crecer α, el límite esP(S>0). Un buscador que no verifica alcanzabilidad puede devolver un extremo del intervalo fingiendo un emparejamiento.

Para un score continuo y α fijo, calibrar el intercepto mediante

`b = −alpha * cuantil_{1−f*}(S_null_train)`.

Recomendación de target de diseño, que debe congelar el coordinador antes de ver R: **f*=0.05**. Es un control algorítmico, no una estimación de una tasa biológica. Registrar cuantiles, tratamiento de empates y tasa alcanzada. No añadir aleatorización de empates de forma silenciosa.

El ciclo instantáneo es binario: su score tiene sólo dos valores. Con pesos deterministas y un umbral0.5 sólo admite ocupaciones0, P(B=1) o1, salvo detalles de empates. Por ello, una FPR5% puede ser inalcanzable para ese control. No interpolar entre políticas como si fuera un umbral realmente ejecutado. Usar el ciclo en igual ganancia y atención media, o declarar la FPR no emparejable.

### B. Atención media igualada

Para cada τ y α, resolver b de modo que la media logística en entrenamiento seaμ*. El coordinador fijó **μ*=0.173** antes de examinar R: es el valor numérico del prior nominal usado como objetivo operacional de atención, no una estimación de prevalencia real. Para α fijo la media es monótona y continua en b; cualquier target estrictamente entre0 y1 es alcanzable en la evaluación offline, incluso si el score es binario. No se garantiza que lo siga siendo con idénticos parámetros en otra dinámica.

### C. Dos objetivos simultáneos, sólo como extensión

Puede buscarse α y b para igualar FPR y atención simultáneamente, pero debe existir una solución dentro de límites fijados previamente. Para scores continuos, imponer el cuantil fija b en función de α; la atención restante se comprueba sobre esa curva. Si no hay solución dentro de la rejilla/bracket, reportar «no alcanzable» y conservar las comparaciones separadas. Este control deja de ser una comparación a α fijo y debe identificarse como otra familia de políticas calibradas.

No calibrar ningún intercepto o multiplicador usando R, recuperación, el ángulo óptimo ni la identidad/tiempo de un futuro líder.

## 5. Nulo offline y dinámica cerrada

Calibrar sobre una trayectoria nula común de ViSt es una buena primera comprobación: todas las variantes reciben la misma entrada. Pero cuando el observador gobierna WiSt, sus falsos positivos cambian la dinámica y con ella la distribución de S. Por tanto, **una igualdad offline no demuestra igualdad alcanzada en lazo cerrado**.

Dos vías válidas:

1. Calibrar en ViSt con semillas de entrenamiento, congelar parámetros y medir en nulos cerrados independientes las tasas realmente logradas. Si no coinciden, describirla como calibración offline y no atribuir la diferencia funcional a memoria a FPR realmente idéntica.
2. Calibrar en nulos cerrados con semillas de entrenamiento, ajustando b y verificando la tasa; después congelar y validar con otras semillas. La relación entre b y la tasa cerrada no tiene garantizada la monotonía por la realimentación social. El optimizador debe detectar esto y el informe conservar los intentos y el error residual.

La validación debe usar la misma duración y estado inicial relevantes para la tarea de respuesta, o documentar por separado el régimen estacionario. Medir tasas por ventanas iniciales/fin de registro ayuda a descubrir diferencias de asentamiento.

Criterios de emparejamiento propuestos para congelar antes de R:

- FPR objetivo5%; tolerancia absoluta±0.01 en ocupación.
- Atención objetivo0.173; tolerancia absoluta±0.015, fijada por el coordinador antes de R.
- Mostrar IC por semilla para cada tasa y para la diferencia entre políticas. Una afirmación de equivalencia debe basarse en que un IC de equivalencia predefinido —por ejemplo90%— quede dentro del margen, no en que una prueba de diferencia resulte no significativa.

Si no pasa el criterio, informar tasas logradas y error de calibración. No ampliar márgenes tras ver qué políticas responden mejor. Si las tasas tienen incertidumbre demasiado grande, el resultado será «emparejamiento no confirmado», aunque los puntos estén cerca.

## 6. Semillas, réplicas y contrastes

Separar tres conjuntos sin solapamiento: entrenamiento de calibración, validación del nulo y evaluación de responsividad. Como esquema concreto posible, usar bases410000000,420000000 y510000000; en respuesta, semilla=`base+10000*i_phi+replica`. El coordinador puede elegir otras bases antes de correr; lo imprescindible es guardar los IDs y comprobar no solapamiento con corridas históricas reutilizadas.

Dentro de cada contraste de respuesta, emparejar por semilla, ángulo, protocolo y duración del estado. Promediar primero los18 ángulos por réplica; los18 ángulos no son18 réplicas estocásticas adicionales. Comparar en N250,k15,ruido√2 los protocolos forzado y maniobra conτW4. Reservar maniobraτW1 como validación de transferencia de las mismas calibraciones, sin reajustar interceptos mirando su R.

Para mayor precisión sería preferible ampliar las réplicas independientes; el coordinador fijó **8 réplicas nuevas** para los contrastes centrales y4 para la rejilla secundaria por coste. Ese número es un diseño acotado válido si se conserva antes de observar R y se reporta la precisión efectivamente alcanzada; los IC con8 semillas describen incertidumbre condicional y no garantizan equivalencia ni generalidad. No elegir la mejor ventana y certificarla sobre las mismas4 réplicas secundarias, ni añadir réplicas hasta obtener significación. Una campaña posterior sería una extensión declarada, con su propio control de selección.

Contrastes primarios sugeridos: **τ=1 frente aτ=0, mismo Student-t y α=12.048192771**, en cada protocoloτW4, bajo FPR-match y mean-match. Son4 contrastes. El barrido completo de ventanas/ganancias es secundario. La celdaτW1 evalúa transferencia y debe etiquetarse como tal.

El resultado primario debe ser la diferencia pareada de responsividad absoluta:

`DeltaR = media(R_temporal − R_instantaneo)`.

Acompañar con diferencia de recuperación y R_CoLd/R_oraculo, pero no basar toda la inferencia en un cociente cuyo denominador cambie. Usar un bootstrap por réplica común a las políticas; para una afirmación confirmatoria que comprenda los4 contrastes, usar intervalos simultáneos o una corrección declarada. Una opción conservadora es IC98.75% por contraste (Bonferroni para familia de4), manteniendo IC95% descriptivos etiquetados. Si se declaran también primarios los dos contrastes fixed-gain, la familia pasa a6; si se añaden los dos joint, a8. Fijar esa familia antes de R y ajustar la corrección; no mantener un divisor4 tras ampliar el número de hipótesis primarias. No considerar todos losτ×α como pruebas primarias no corregidas.

Una diferencia positiva con IC que excluye0 establece ventaja en esas condiciones. Si se quiere afirmar ventaja material, fijar además un margen antes del análisis; propuestaδ_R=0.02 unidades de alineamiento medio, con interpretación explícita como elección de diseño. «No significativo» no demuestra equivalencia. Para decir que el instantáneo iguala al temporal se necesita un intervalo de equivalencia dentro de±δ_R o una incertidumbre que permita una conclusión más limitada.

Los IC de respuesta condicionan en los parámetros calibrados. Si se pretende incluir incertidumbre de calibración, repetir la calibración por conjuntos de semillas o remuestrearla de forma anidada; de lo contrario, declarar esa condición y mostrar la validación nula independiente. La enorme cantidad de cuadros no reemplaza semillas independientes.

## 7. Reglas de interpretación antes de las confirmatorias

| Resultado | Interpretación permitida |
|---|---|
| Temporal supera instantáneo en ambas familias de emparejamiento, con tasas validadas | Apoyo más fuerte a una ventaja de integrar la misma evidencia bajo los controles estudiados; no prueba optimalidad universal |
| Ventaja sólo a igual ganancia; desaparece al emparejar FPR o atención | El beneficio depende de la calibración/política de pesos; revisar la atribución exclusiva a memoria |
| Ventaja con un emparejamiento pero no el otro | Indica que FPR y atención son controles distintos; reportar el patrón, no elegir sólo la comparación favorable |
| Emparejamiento offline no se reproduce en nulos cerrados | No está cerrada la objeción de tasa/atención iguales; reportar tasas logradas y límites |
| Amplios IC o pocos eventos nulos | Inconcluso; no equivalencia ni refutación de toda memoria |
| Ciclo instantáneo no alcanza la FPR target | Límite de la política binaria; no un resultado de rendimiento ni motivo para ocultar esa variante |
| Ganancia óptima o ventana óptima elegida sobre evaluación | Exploratorio; confirmar con nuevas semillas o evaluación retenida |
| τW1 conserva ventaja sin recalibración | Transferencia limitada al nuevo protocolo; no reemplaza las pruebas primariasτW4 |

Todos los resultados de las celdas fijadas deben conservarse, incluidos fallos de calibración, nulos e inversiones de signo. Los filtros angulares, el grafo, el código del motor y el tratamiento de voluntad deben ser comunes entre políticas salvo la intervención declarada. No corregir simultáneamente otros problemas del paper durante este experimento, porque impediría atribuir diferencias a la intervención memoria/ganancia.

## 8. Lista corta para autorizar el bloque confirmatorio

El protocolo debería contener, antes de lanzar ese bloque: fórmula exacta del score sin escalado oculto; targets y márgenes congelados; estados iniciales y duraciones; IDs de semillas separados; reglas para empates/no alcanzabilidad; método de calibración sin usar R; tabla de tasas nulas de validación; contrastes primarios y multiplicidad; número fijo de réplicas; regla de resultados adversos. Si alguna calibración sigue sin validar, se puede ejecutar el banco descriptivo, pero no afirmar todavía «a igual FPR/atención».
