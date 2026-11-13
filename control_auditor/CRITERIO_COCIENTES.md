# Precisión del análisis de recuperación

Se fija durante la ejecución y antes del análisis de respuestas completo: el estimando principal es la diferencia absoluta pareada de R. No se invalida la campaña si el comparador oráculo tiene una ventaja pequeña o no resuelta frente a la base.

La recuperación puntual se calcula sólo si la diferencia media R_oráculo − R_base es positiva. Su intervalo bootstrap se muestra como estable sólo si el IC95 del denominador tiene límite inferior positivo y ningún denominador remuestreado es exactamente cero. En caso contrario se conserva la estimación absoluta y se marca el intervalo del cociente como no estable. No se seleccionan ni descartan remuestreos por el signo del denominador. Se conserva la fracción bootstrap de denominadores no positivos.

Los intervalos de recuperación son descriptivos y aproximados. Un intervalo estable según este criterio no resuelve por sí solo todas las limitaciones de inferencia de cocientes. La banda simultánea de los ocho contrastes principales se calcula sobre diferencias R y no depende de esta regla.

Esta precisión no cambia políticas, semillas, tamaño muestral, calibración ni resultados crudos, y no presupone que el problema llegue a aparecer en la campaña. Evita convertir un denominador incierto en un fallo de ejecución o en un porcentaje engañoso.
