"""Informe de resultados de la campaña completa, generado desde sus tablas."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
x=json.loads((ROOT/'resultados/RESULTADOS_MEMORIA_GANANCIA.json').read_text())
c=json.loads((ROOT/'calibracion/calibracion_final.json').read_text())
equiv=json.loads((ROOT/'calibracion/equivalencia_pareada.json').read_text())
burn=json.loads((ROOT/'piloto/asentamiento_memoria.json').read_text())
labels={'fixed':'Misma ganancia e intercepto','fpr':'Falsos positivos equiparados',
        'mean':'Atención media equiparada','joint':'Ambas magnitudes equiparadas'}
policy_labels={'full_t0_ag0':'Instantáneo, parámetros fijos',
               'full_t1_ag0':'Memoria 1 s, parámetros fijos',
               'full_t0_joint':'Instantáneo, ajuste conjunto',
               'full_t1_joint':'Memoria 1 s, ajuste conjunto'}
def f(v,n=4):return f'{v:.{n}f}'
def inter(v,n=4):return '['+', '.join(f(q,n) for q in v)+']' if v is not None else 'IC no estable'
def percent(v):return f'{100*v:.2f}' if v is not None else 'no definido'
def percent_ci(v):return inter([100*q for q in v],2) if v is not None else 'IC no estable'

joint=[r for r in x['primary_contrasts'] if r['scheme']=='joint']
if all(r['operational_matching_validated'] and r['direction']=='memory' for r in joint):
    headline='Con falsos positivos y atención basal equiparados dentro de los márgenes predefinidos, la política con memoria de 1 s obtiene mayor responsividad en los dos protocolos principales. Ese control operacional no elimina la ventaja; no implica igualdad exacta de las tasas ni de toda la distribución de pesos. El control conjunto ajusta también la ganancia y no estima el efecto de cambiar sólo la memoria; ese efecto corresponde al contraste a igual ganancia e intercepto.'
elif all(r['operational_matching_validated'] and r['direction']=='instantaneous' for r in joint):
    headline='Con falsos positivos y atención basal equiparados simultáneamente, el instantáneo obtiene mayor responsividad en los dos protocolos principales. Estos controles no respaldan una ventaja general de añadir 1 s de memoria bajo esa política calibrada.'
else:
    pieces=[]
    for r in joint:
        status='ventaja temporal' if r['direction']=='memory' else 'ventaja instantánea' if r['direction']=='instantaneous' else 'diferencia no resuelta'
        if not r['operational_matching_validated']:status='equiparación no validada'
        pieces.append(r['protocol']+': '+status)
    headline='El ajuste conjunto de falsos positivos y atención no ofrece una conclusión uniforme: '+'; '.join(pieces)+'. El resultado requiere distinguir protocolos y precisión disponible.'
reversals=[r for r in x['primary_contrasts'] if r['direction']=='instantaneous']
if reversals:
    details=[r['protocol']+', '+labels[r['scheme']].lower()+': diferencia R '+f"{r['difference_R']:+.4f} "+inter(r['simultaneous_CI95']) for r in reversals]
    headline+='\n\nLa ventaja no es universal: '+ '; '.join(details)+'. En esa comparación gana el instantáneo.'
central={(r['protocol'],r['tauW'],r['variant']):r for r in x['conditions']}
man_joint=central[('maniobra',4.,'full_t1_joint')]
man_fixed=central[('maniobra',4.,'full_t1_ag0')]
headline+=f"\n\nLa magnitud también depende de la calibración: en maniobra, el temporal con ajuste conjunto recupera {percent(man_joint['recovery'])} % de la ventaja del oráculo, frente a {percent(man_fixed['recovery'])} % con parámetros fijos."

lines=['# Prioridad 3 Memoria frente a ganancia de evidencia','',
 'Fecha: 12 de noviembre de 2026. Todas las simulaciones, fuentes congeladas, configuraciones, calibraciones y resultados nuevos se guardan dentro de esta carpeta. Este trabajo no edita el paper ni descarga datos.',
 '', '**Resultado principal.** '+headline,
 '', '## Qué se puso a prueba','',
 'Se comparó un observador que usa la evidencia Student-t completa en cada instante con otro que promedia exactamente la misma evidencia durante1s. La ganancia multiplicativa y el intercepto logístico se separaron de la memoria. La entrada conserva el mismo suavizado de0.083s: instantáneo significa sin acumulador adicional.',
 'El score instantáneo es el log-cociente entre las dos densidades Student-t del residual cinemático. Para tau > 0 se usa S(t) = a S(t−dt) + (1−a) score(t), con a = exp(−dt/tau); para tau = 0 se usa el score directamente. El peso es logistic(b + alpha S), sin multiplicador automático tau/tau_obs. La memoria Student empieza en cero y la binaria en su probabilidad nula q0; ambas reciben el asentamiento común. La voluntad propia sigue siendo conocida por cada agente.',
 'El resultado principal es la diferencia de responsividad del grupo, no una prueba de identificación de información privada. Se calcula R como el promedio temporal de la velocidad media de los 250 agentes proyectada sobre la dirección objetivo, dividido por v0; incluye al líder. La recuperación es (R − R_base)/(R_oráculo − R_base), se refiere a una política oráculo concreta y puede superar100%.',
 '', '## Diseño realizado','',
 f"Se completaron {x['new_response_runs']} corridas nuevas de giro, agrupadas en {x['condition_count']} condiciones y {x['replicate_condition_rows']} medias por réplica y condición. Cada réplica resume18ángulos; las condiciones principales tienen8réplicas pareadas y el factorial complementario4. No se trata cada ángulo como una réplica independiente.",
 'Parámetros de referencia: N=250, k=15, ruido Wiener de intensidad numérica sqrt(2), dt=0.01s. Protocolos principales: líder forzado y maniobra única, tauW=4s, horizonte15.8s. La maniobra con tauW=1s es una validación secundaria. Se recalcularon CoLd/ViSt y oráculo con las mismas semillas y el mismo asentamiento; no se reutilizaron denominadores de las corridas antiguas con asentamiento distinto.',
 'La predeclaración interna se escribió antes de medir las respuestas nuevas. La adenda precisó que la calibración cuenta aristas de percepción sin autoenlaces y que la ganancia central efectiva es12.048. Se conservan ambas decisiones, la revisión del diseño y las calibraciones preliminares sustituidas. No hay registro externo ni cohorte biológica nueva.',
 '', '## Calibración independiente de la respuesta','',
 'Se ajustaron parámetros sólo con cuatro semillas de simulaciones sin informados. Los registros nulos tienen 12 s de asentamiento y 12 s de medición. Luego se congelaron y se validaron en ocho semillas nuevas911000–911007; no se reajustaron usando esa validación ni las respuestas al giro. La primera prueba por emisor quedó archivada como preliminar, y su validación anterior no se reutilizó como validación final.',
 'El estimando de equiparación es la distribución de pesos sobre aristas dirigidas receptor–emisor, excluyendo el propio agente. La tasa de falso positivo es la fracción de pesos mayores de0.5 en el nulo, no una tasa de episodios de alarma por segundo. La atención es el peso medio sobre esas aristas. Objetivos de diseño:0.05 y0.173, respectivamente; no son prevalencias animales medidas.',
 '', '| Esquema | Memoria s | Factor alpha | Intercepto total | FPR validada [IC95] | Atención validada [IC95] |',
 '|---|---:|---:|---:|---|---|']
for v in c['variants']:
    s=v['validation_summary']
    lines.append(f"| {labels[v['scheme']]} | {v['tau']:g} | {v['gain']:.5f} | {v['intercept']:.5f} | {s['fpr']:.5f} {inter(s['ci95_fpr'],5)} | {s['mean_attention']:.5f} {inter(s['ci95_mean_attention'],5)} |")
lines+=['',
 'La comparación FPR-match deja libre la atención y la comparación mean-match deja libre la FPR. El ajuste joint controla las dos; requiere ganancias distintas porque las distribuciones de la evidencia instantánea y temporal son diferentes. No se denomina igualdad exacta: se exige que los intervalos de diferencias pareadas queden dentro de±0.01 para FPR y±0.015 para atención, además de comprobar cada objetivo.',
 f"Las tres parejas cumplen sus criterios operacionales: {'sí' if equiv['all_required_pair_equivalence'] else 'no'}. La comprobación numérica completa está en [CALIBRACION_Y_EQUIVALENCIA.md](CALIBRACION_Y_EQUIVALENCIA.md). Los intervalos son condicionales a los parámetros entrenados y a este motor.",
 '', f'![Validación nula de la calibración](<{ROOT / "figuras/validacion_nula.png"}>)',
 '', '## Contrastes principales de memoria','',
 'La tabla muestra memoria1s menos evidencia instantánea. Los intervalos simultáneos controlan conjuntamente los ocho contrastes definidos antes de las respuestas, mediante el percentil95 del máximo desvío bootstrap centrado. Se remuestrean8réplicas completas y pareadas10000veces. Es una aproximación bootstrap con pocas semillas, no una garantía exacta de cobertura. Las cifras se redondean por separado; las diferencias se calculan antes de redondear.',
 '', '| Protocolo tauW4s | Control | R instantáneo | R memoria | Diferencia R [IC95 simultáneo] | Diferencia de recuperación, puntos porcentuales [IC95 individual] |',
 '|---|---|---:|---:|---|---|']
for r in x['primary_contrasts']:
    lines.append(f"| {r['protocol']} | {labels[r['scheme']]} | {r['R_instant']:.4f} | {r['R_memory']:.4f} | {r['difference_R']:+.4f} {inter(r['simultaneous_CI95'])} | {percent(r['difference_recovery'])} {percent_ci(r['difference_recovery_CI95'])} |")

lines+=['', f'![Ocho contrastes principales de memoria](<{ROOT / "figuras/contrastes_principales.png"}>)',
        '', '### Lectura por control','']
for scheme in ['fixed','fpr','mean','joint']:
    rr=[r for r in x['primary_contrasts'] if r['scheme']==scheme]
    outcomes=[]
    for r in rr:
        if not r['operational_matching_validated']:
            status='la equiparación no quedó validada; la diferencia es una sensibilidad'
        elif r['direction']=='memory':status='la memoria conserva una ventaja resuelta en la banda simultánea'
        elif r['direction']=='instantaneous':status='el instantáneo obtiene mayor responsividad en la banda simultánea'
        else:status='la diferencia no queda resuelta con la precisión disponible'
        outcomes.append(r['protocol']+': '+status)
    lines.append('**'+labels[scheme]+'.** '+'; '.join(outcomes)+'.\n')

lines+=['','## Recuperación y valores absolutos','',
 '| Protocolo | tauW | Política | R [IC95] | R base | R oráculo | Recuperación % [IC95] |',
 '|---|---:|---|---|---:|---:|---|']
names=['full_t0_ag0','full_t1_ag0','full_t0_joint','full_t1_joint']
for r in x['conditions']:
    if r['burn']!=12 or r['variant'] not in names:continue
    lines.append(f"| {r['protocol']} | {r['tauW']:g} | {policy_labels[r['variant']]} | {r['R']:.4f} {inter(r['R_CI95'])} | {r['R_base']:.4f} | {r['R_oracle']:.4f} | {percent(r['recovery'])} {percent_ci(r['recovery_CI95'])} |")
lines+=['','Si el IC95 del denominador alcanza cero, no se informa un IC de recuperación como estable. Una media del denominador no positiva deja también el punto de recuperación sin definir. Esta regla se fijó antes de ejecutar el análisis final; no afecta la diferencia R principal.', '', 'Los CSV incluyen todos los factores, interceptos, denominadores y diagnósticos sobre líder y no líderes; la tabla anterior selecciona las parejas centrales fijadas por el diseño, no una selección posterior por éxito.','',
        '## Atención durante el giro: diagnóstico descriptivo','',
        'Se muestran las parejas a igual ganancia y con ajuste conjunto. Son promedios durante los 15.8 s de respuesta, no nuevas restricciones de calibración. Los intervalos individuales se calculan por réplica; el líder puede dejar de estar informado durante la maniobra, por lo que su peso no es una sensibilidad diagnóstica frente a una etiqueta fija.', '',
        '| Protocolo tauW 4 s | Política | Peso medio recibido [IC95] | Peso recibido del líder por agente [IC95] | Fracción del peso destinada al líder [IC95] |',
        '|---|---|---|---|---|']
unstable=sum(not r['recovery_interval_stable'] for r in x['conditions'])
lines.insert(lines.index('## Atención durante el giro: diagnóstico descriptivo')-1,
    f"Condiciones con intervalo de recuperación marcado como no estable: {unstable} de {len(x['conditions'])}."+'\n')
for r in x['conditions']:
    if r['tauW']!=4 or r['variant'] not in names:continue
    d=r['diagnostics']
    def diag(name):
        q=d['diag_'+name]
        return f"{q['mean']:.5f} {inter(q['CI95'],5)}"
    lines.append(f"| {r['protocol']} | {policy_labels[r['variant']]} | {diag('nonself_pair_weight_mean')} | {diag('leader_weight_per_receiver')} | {diag('leader_attention_fraction')} |")
lines+=['',
        '## Validación secundaria con tauW de1s','',
        '| Control | R instantáneo | R memoria | Diferencia R e IC95 individual |',
        '|---|---:|---:|---|']
for r in x['secondary_tauW1']:
    lines.append(f"| {labels[r['scheme']]} | {r['R_instant']:.4f} | {r['R_memory']:.4f} | {r['difference_R']:+.4f} {inter(r['CI95'])} |")
lines+=['','Esta validación no se combina con la familia principal de ocho contrastes. Sus intervalos son descriptivos y no se usan para elegir qué condiciones conservar.','',
       '## Mismo estadístico binario con y sin memoria','',
       'Las dos ramas usan el mismo indicador de excedencia, la misma ganancia12.048 y el mismo intercepto. Así se evita atribuir a memoria un cambio simultáneo de amplitud a ciclo. Los contrastes de este bloque son secundarios y sus intervalos no tienen la corrección de la familia principal.','',
       '| Protocolo tauW4s | Estadístico | R instantáneo | R memoria1s | Diferencia R [IC95 individual] |',
       '|---|---|---:|---:|---|']
for r in x['secondary_statistic_controls']:
    if r['stat'].startswith('interaction'):continue
    lines.append(f"| {r['protocol']} | {r['stat']} | {r['R_instant']:.4f} | {r['R_memory']:.4f} | {r['difference_R']:+.4f} {inter(r['CI95'])} |")
lines+=['',
       '## Controles complementarios y límites','',
       'El factorial Student incluye memorias0,0.25,1y4s y factores1,12.048y48 con intercepto fijo. Los controles binarios comparan el mismo indicador de excedencia con y sin memoria a igual factor. Son complementarios; no se fuerza una FPR del5% en un score instantáneo binario que sólo admite dos valores deterministas.',
       'La rejilla secundaria se acotó antes de las respuestas a cuatro ventanas; no se ejecutaron las ventanas de 0.5 y 2 s del bosquejo inicial del plan. La campaña resuelve los controles centrales de memoria frente a ganancia, pero no completa una búsqueda fina de la ventana óptima.',
       '', f'![Factorial complementario de memoria y ganancia](<{ROOT / "figuras/factorial_memoria_ganancia.png"}>)', '',
       'El factorial muestra una interacción: aumentar la memoria no sustituye cualquier ganancia. Con alpha=1, el rendimiento cambia poco entre ventanas y permanece bajo; con ganancias mayores aparecen diferencias temporales claras en las medias. Subir la ganancia instantánea a 48 no reproduce el rendimiento de la memoria de 1 s a esa misma ganancia en ninguno de los dos protocolos. Es una lectura descriptiva de la rejilla de cuatro réplicas, sin afirmar un óptimo universal ni localizar con precisión la ventana.',
       f"La prueba nula de asentamiento comparó12y24s para tau4, tres ganancias y cuatro semillas. Diferencias máximas de medias: FPR={burn['maximum_absolute_FPR_difference']:.6f}, atención={burn['maximum_absolute_mean_weight_difference']:.6f}. Activó la regla de24s: {'sí' if burn['use_common_burn24_for_tau4_block_and_references'] else 'no'}. Ese test es una sensibilidad al asentamiento, no una demostración de estacionariedad completa.",
       'El diagnóstico de atención se muestrea cada0.1s; R se integra cada0.01s. Los pesos durante el giro no tienen por qué conservar sus tasas nulas: esa diferencia forma parte de la respuesta al estímulo. La calibración y las incertidumbres están verificadas en el nulo, y no deben extrapolarse a tasas biológicas de falsas alarmas.',
       'La ganancia, el intercepto y la memoria definen una política de atención. La calibración logística no prueba que el peso sea una probabilidad posterior bien calibrada de voluntad informada. Se conservan la memoria por emisor y el acceso cinemático del motor anterior; no se resuelve aquí la factibilidad de percepción local ni el reconocimiento animal de información privada.',
       'Equiparar ocupación por encima de 0.5 y peso medio no iguala toda la distribución de pesos ni la duración de las rachas. Cada política también actúa durante su asentamiento: el contraste estima su efecto total en lazo cerrado, incluidas las diferencias de estado inicial inducidas por ella. La incertidumbre no integra la búsqueda de parámetros ni variación entre arquitecturas del modelo.',
       '', '## Objeciones cerradas y pendientes','',
       'Respecto del dictamen previo, se atiende el componente memoria–ganancia de la objeción mayor 4, se documenta la especificación de estos controles para la objeción 1 y se mejora la incertidumbre condicional de la objeción 6. No quedan cerradas por completo esas objeciones: H3, la incertidumbre estructural y la incorporación de los métodos al PDF siguen pendientes. El cierre que se informa aquí pertenece al expediente de resultados.', '',
       '| Objeción | Estado alcanzado |', '|---|---|',
       '| Cambiar tau cambia a la vez el multiplicador automático de evidencia | Separado en el módulo nuevo: alpha y tau son entradas independientes; se conserva un factorial y el contraste a igual alpha |',
       '| Amplitud instantánea frente a ciclo cambia también el estadístico | Se compara Student completo con Student completo y, como control secundario, binario con binario |',
       '| La diferencia podría atribuirse a mayor tasa basal de pesos altos | Control ejecutado y validado. En maniobra, FPR-match favorece al instantáneo; no se sostiene superioridad temporal en toda calibración. El ajuste conjunto sí favorece al temporal en ambos protocolos |',
       '| La diferencia podría atribuirse sólo a mayor peso medio | Se calibra y valida mean-match, además de joint con las dos magnitudes |',
       '| Los parámetros se optimizaron mirando R o los mismos giros de evaluación | Se separaron entrenamiento nulo, validación nula y semillas nuevas de giro; parámetros congelados antes de validar y responder |',
       '| El resultado sólo tiene cuatro réplicas del barrido antiguo | Los contrastes centrales nuevos tienen ocho réplicas independientes del entrenamiento; el factorial sigue con cuatro y se identifica como complementario |',
       '| La memoria es una explicación universal o única de todo reconocimiento | Sigue abierta: la inferencia se limita al motor, estadístico, ganancias y protocolos probados |',
       '| La mejor ventana está determinada causalmente por una escala social | Sigue abierta; este factorial no identifica esa escala ni una duración óptima precisa |',
       '| Los animales reconocen información privada por este mecanismo | Sigue abierta; no se añadieron observaciones ni intervenciones animales |',
       '| El PDF ya incorpora los cierres | Pendiente de edición; el paper no es destino de escritura de esta campaña |',
       '', 'La existencia de un control y la dirección que éste revela son cuestiones distintas. Las objeciones de confusión se responden con el diseño y los resultados, aun si una comparación no favorece a la memoria. No se interpreta falta de diferencia resuelta como equivalencia de rendimiento.',
       '', '## Qué falta para reforzar la atribución', '',
       'No se necesitan descargas para los siguientes controles computacionales. En orden: repetir el entrenamiento con cohortes independientes para incluir incertidumbre de calibración; equiparar la distribución nula completa de pesos, además de su media y ocupación, y estudiar las rachas; comparar estímulos desde estados iniciales comunes para separar el efecto del asentamiento; intervenir escalas sociales y de maniobra para probar H3. Son extensiones pendientes, no resultados de esta campaña. Una afirmación de reconocimiento de información privada en animales requiere además etiquetas o intervenciones adecuadas y evaluación causal; estas simulaciones no las sustituyen.',
       '', '## Evidencia y reproducción','',
       '- [Predeclaración](PREDECLARACION.md), [precisión del estimando](ADENDA_ESTIMANDO.md), [criterio para cocientes](CRITERIO_COCIENTES.md) y [revisión independiente del diseño](REVISION_DISENO.md).',
       '- `calibracion/`: búsqueda nula, parámetros congelados, validación y equivalencia pareada; se conserva la fase preliminar por emisor.',
       '- `resultados/corridas/`: cada ensayo con semilla, ángulo, política, parámetros y diagnóstico. Son los crudos nuevos; no se sobrescribieron E58 ni sus antecedentes.',
       '- `resultados/RESULTADOS_MEMORIA_GANANCIA.json`, `resumen_condiciones.csv`, `contrastes_principales.csv`, `respuestas_por_replica.csv` y `respuestas_por_angulo.csv`.',
       '- `piloto/`: anclas de equivalencia con el motor histórico, anclaje adicional del secuencial y test de asentamiento; `evidencia/`: hashes de fuentes, configuración efectiva y comparación de originales.',
       '- [Configuración efectiva y reconstrucción](evidencia/PARAMETROS_EFECTIVOS.json): el bloque `result.params` de cada crudo describe los parámetros comunes previos al enrutamiento; para ViSt/CoLd/oráculo, `new_flock` fuerza `will_mode=oracle`. Los infinitos del JSON crudo se guardan como null; su significado se conserva en esa configuración. Reproducir desde `job` y las funciones guardadas evita interpretar esos campos como una política distinta.',
       '- Los scripts de simulación reanudan checkpoints compatibles y detienen la ejecución si cambian las fuentes. El README de esta carpeta enumera los comandos y el alcance de cada archivo.',
       '', 'El informe se refiere al expediente nuevo. Cualquier cambio concurrente en el manuscrito original se registra aparte y no se restaura automáticamente.']
verification=ROOT/'evidencia/VERIFICACION_FINAL.json'
if verification.exists():
    z=json.loads(verification.read_text())
    integrity=z['original_files']
    lines+=['', '## Auditoría final de cifras e integridad', '',
       f"Se verificaron individualmente las {z['response_runs']} corridas: R y sus ventanas de 4 s se reconstruyen a partir de la serie guardada, las medias por condición coinciden con los crudos, las recuperaciones usan sus denominadores correspondientes y los ocho contrastes coinciden entre JSON y CSV. Discrepancias por encima de 2 × 10⁻¹²: ninguna. El mayor error de reconstrucción de R es {z['maximum_absolute_errors']['R']:.2e}.",
       f"Se mantienen idénticos {integrity['unchanged_count']} de los {integrity['baseline_count']} archivos originales inventariados. Cambiados: {len(integrity['changed'])}; ausentes: {len(integrity['missing'])}. El inventario incluye el PDF y las fuentes del paper corto, el código, resultados y documentos previos. No se descargaron datos.",
       'Las verificaciones y sus límites están en [VERIFICACION_FINAL.json](evidencia/VERIFICACION_FINAL.json). '+('Se revisaron las tres figuras finales; el registro está en [REVISION_VISUAL.md](evidencia/REVISION_VISUAL.md).' if (ROOT/'evidencia/REVISION_VISUAL.json').exists() else 'Las figuras se revisan visualmente antes de entregar el expediente.')]
    md_audit=ROOT/'evidencia/AUDITORIA_TABLAS_MD.json'
    if md_audit.exists():
        a=json.loads(md_audit.read_text())
        lines+=['',f"Además se contrastaron {a['numeric_values']} valores impresos en las seis tablas del MD ({a['data_rows']} filas) con sus salidas numéricas: {a['discrepancies']} discrepancias a la precisión decimal declarada. Registro: [AUDITORIA_TABLAS_MD.json](evidencia/AUDITORIA_TABLAS_MD.json)."]
out='\n'.join(lines)+'\n'
for a,b in [('durante1s','durante 1 s'),('de0.083s','de 0.083 s'),('superar100%','superar 100 %'),
            ('resume18ángulos','resume 18 ángulos'),('tienen8réplicas','tienen 8 réplicas'),
            ('complementario4','complementario 4'),('tauW=4s','tauW=4 s'),('horizonte15.8s','horizonte 15.8 s'),
            ('tauW=1s','tauW=1 s'),('separado0','separado 0'),
            ('dt=0.01s','dt=0.01 s'),('nuevas911000','nuevas 911000'),
            ('es12.048','es 12.048'),
            ('de0.5','de 0.5'),('diseño:0.05 y0.173','diseño: 0.05 y 0.173'),
            ('dentro de±0.01','dentro de ±0.01'),('y±0.015','y ±0.015'),
            ('memoria1s','memoria 1 s'),('percentil95','percentil 95'),
            ('remuestrean8réplicas','remuestrean 8 réplicas'),('pareadas10000veces','pareadas 10 000 veces'),
            ('tauW4s','tauW 4 s'),('de1s','de 1 s'),('ganancia12.048','ganancia 12.048'),
            ('memorias0,0.25,1y4s','memorias 0, 0.25, 1 y 4 s'),
            ('factores1,12.048y48','factores 1, 12.048 y 48'),('del5%','del 5 %'),
            ('comparó12y24s','comparó 12 y 24 s'),('tau4','tau de 4 s'),
            ('de24s','de 24 s'),('cada0.1s','cada 0.1 s'),('cada0.01s','cada 0.01 s')]:out=out.replace(a,b)
import re
spaced=[]
def prose(line):return bool(line.strip()) and not re.match(r'^(#|\||[-*+]\s|!\[|```)',line)
for line in out.splitlines():
    if spaced and prose(spaced[-1]) and prose(line):spaced.append('')
    spaced.append(line)
out='\n'.join(spaced)+'\n'
(ROOT/'INFORME_PRIORIDAD_3.md').write_text(out)
print(ROOT/'INFORME_PRIORIDAD_3.md')
