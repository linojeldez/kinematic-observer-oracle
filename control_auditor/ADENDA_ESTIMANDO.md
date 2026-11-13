# Precisión del estimando de calibración antes de las respuestas

La calibración y su validación usarán aristas dirigidas de percepción que excluyen al propio agente. Un mismo emisor aparece una vez por receptor que lo tiene como vecino. De esta forma se equiparan los pesos que realmente entran en la regla social, y se evita confundir la media por emisor con la media recibida cuando el grado de entrada se relaciona con el score.

Los objetivos permanecen FPR0.05 (fracción de pesos de aristas sin informados superiores a0.5) y peso medio0.173. Ambos se muestrean cada0.1s, con intervalos por semilla, no por arista. Las respuestas R se integran con todos los pasos de0.01s. Se registran también los diagnósticos por emisor como sensibilidad; no se mezclarán los estimandos.

La fuente ViSt puede aportar un calentamiento de la búsqueda, pero el ajuste final usa dinámicas cerradas del observador y aristas actuales. No se usan respuestas al giro para escoger parámetros. Se permiten los mismos cuatro pasos de ajuste, se congelan parámetros y se evalúan en semillas nulas retenidas antes de las respuestas. Las tolerancias de comparación entre memorias se mantienen.

Esta precisión se documenta antes de ejecutar las respuestas del experimento nuevo; no cambia objetivos a partir de resultados de responsividad.

## Precisión numérica y validación retenida final

La ganancia central efectiva se fija en12.048 en todos los controles a ganancia fija y en los ajustes FPR/mean, exactamente el valor usado al calibrar. Es el redondeo de1/0.083=12.0481927711, una diferencia relativa de0.0016%; se conservan los valores efectivos en cada configuración. El control binario de esta campaña usa también12.048; el anclaje al detector histórico usó1/0.083 exacto.

La primera calibración por emisores se conserva como preliminar sustituida. La validación definitiva por aristas usa las semillas911000..911007, distintas de las910000..910007 de esa fase preliminar. El ajuste utiliza hasta4iteraciones sobre entrenamiento con tolerancias de parada más estrictas (FPR0.001 y atención0.002), fijadas antes de la validación final y sin respuestas. Las tolerancias de equivalencia entre condiciones siguen siendo0.01 y0.015.
