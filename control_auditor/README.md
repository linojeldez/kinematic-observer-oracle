# Experimento de memoria y ganancia

Las respuestas nuevas se guardan por ensayo y se pueden reanudar. El diseño y las semillas se fijaron en [PREDECLARACION.md](PREDECLARACION.md) y [ADENDA_ESTIMANDO.md](ADENDA_ESTIMANDO.md). La revisión estadística previa está en [REVISION_DISENO.md](REVISION_DISENO.md).

El informe final está en [INFORME_PRIORIDAD_3.md](INFORME_PRIORIDAD_3.md). Se completaron las 5760 corridas, las tablas y la revisión de las tres figuras. La auditoría numérica no encontró discrepancias por encima de la tolerancia de 2 × 10⁻¹² y los 409 originales inventariados permanecen idénticos.

La calibración nula y su validación están en `calibracion/`. Los controles pasan las tolerancias operacionales de FPR y/o atención exigidas; eso no garantiza una ventaja de rendimiento. La fase por emisores se conserva como preliminar; la definitiva usa aristas de percepción sin autoenlaces.

## Estructura

- `fuentes/`: motor congelado y antecedente E58.
- `scripts/`: simulación, calibración, análisis y figuras.
- `piloto/`: anclas exactas del motor y sensibilidad al asentamiento.
- `calibracion/`: entrenamiento, parámetros congelados antes de validar, semillas retenidas y muestras nulas.
- `resultados/corridas/`: un JSON por ensayo, con configuración, semilla, ángulo, R y diagnósticos.
- `logs/`: avance y resúmenes por fase.
- `evidencia/`: controles de integridad y verificaciones.
- `figuras/`: gráficos calculados a partir de las tablas finales.

El informe final se genera sólo cuando terminan las dos fases completas y pasan sus comprobaciones. `evidencia/VERIFICACION_FINAL.json` conserva la auditoría numérica y `evidencia/REVISION_VISUAL.md` documenta la inspección de las figuras.

## Reproducir

Se usó Python3.14 con NumPy2.5.2 y SciPy1.18.0 del equipo. El entorno numérico se activa con `PYTHONPATH=/opt/homebrew/Cellar/numpy/2.5.2/lib/python3.14/site-packages`; los scripts también requieren `PYTHONDONTWRITEBYTECODE=1` para conservar limpias las fuentes. Los gráficos usan R base, sin paquetes adicionales ni red.

Las comprobaciones de compatibilidad están en `scripts/test_motor_controles.py`. Las calibraciones ya congeladas permiten repetir las respuestas sin volver a ajustarlas:

```bash
python3 scripts/ejecutar_respuestas.py --phase fixed --workers 8
python3 scripts/ejecutar_respuestas.py --phase matched --workers 4
python3 scripts/analizar_respuestas.py
python3 scripts/verificar_expediente.py
python3 scripts/informe_resultados.py
python3 scripts/auditar_tablas_md.py
/usr/local/bin/Rscript --vanilla scripts/figuras_resultados.R
```

Esos comandos se ejecutan con el entorno descrito arriba y desde esta carpeta. Los checkpoints se reutilizan únicamente si coinciden configuración y hashes; no se regeneran corridas completas sin motivo. Una fuente incompatible hace fallar la ejecución.

`scripts/reforzar_cola.py` adelantó 144 trabajos ya previstos del final de la fase calibrada mediante checkpoints compatibles. Usó temporales separados y publicación atómica, sin cambiar políticas, semillas ni tamaño muestral. Su registro está en `logs/refuerzo_cola.json`; no es una condición experimental adicional.

La reconstrucción completa de calibración está en `scripts/ejecutar_calibracion.py`; volver a ajustarla es un experimento distinto a analizar sus parámetros ya congelados. Para verificar la comparación nula se usa `scripts/validar_equivalencia_nula.py`.

El paper y sus fuentes originales no son destinos de escritura. Los cambios concurrentes, si los hay, se documentan sin restaurarlos. El objetivo científico es separar factores de una política de atención en el motor, no demostrar reconocimiento animal de información privada.
