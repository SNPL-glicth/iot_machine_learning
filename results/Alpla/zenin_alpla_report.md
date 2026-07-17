# ZENIN sobre ALPLA — Resultados por Parámetro

**Config:** threshold=0.5, window=50
**Parámetros:** 47 (Chiller 18, CA 29)

## 1. Resumen por Tipo de Señal

### Contadores monotónicos (7 params)

|Métrica|Valor|
|---|---|
|Parámetros|7|
|Detección promedio (fixed 0.5)|236 / ~273 pts (87%)|
|Rango p50 scores|[0.272, 0.481]|
|Rango p95 scores|[0.501, 0.648]|
|Señal útil para ZENIN?|No — señal siempre creciente, ZENIN siempre detecta como anómalo porque cada nuevo valor supera el rango histórico.|
|Alternativa|Usar detección de saltos discretos o cambios en pendiente (diff logarítmico o % change).|

### Señales estacionarias/cíclicas (37 params)

|Métrica|Valor|
|---|---|
|Parámetros|37|
|Detección promedio (fixed 0.5)|30 / ~288 pts (10%)|
|Rango p50 scores|[0.076, 0.324]|
|Rango p95 scores|[0.104, 0.431]|
|Señal útil para ZENIN?|Sí — varían alrededor de un punto de operación. Scores bajos en operación normal, altos en desviaciones.|
|Alternativa|Threshold fijo 0.5 o calibración por percentil con supuesto de contaminación a priori.|

### Señales constantes (3 params, presión CA, 1 valor único)

|Métrica|Valor|
|---|---|
|Parámetros|3|
|Detección promedio (fixed 0.5)|0 / ~303 pts (0%)|
|Rango p50 scores|[0.104, 0.104]|
|Rango p95 scores|[0.104, 0.104]|
|Señal útil para ZENIN?|No — señal plana, scores fijos ~0.104, el detector no puede distinguir nada.|
|Alternativa|Excluir del pipeline de detección. Monitorear solo cambio de estado binario.|

## 2. Resultados por Parámetro

### Chiller (18 params)

|Parámetro|Tipo|n|vals|p50|p95|p99|det@0.5|det%|
|---|---|---|---|---|---|---|---|---|
|Cto.1 Tiempo de operación del compresor|contador|312|306|0.481|0.648|0.648|262|100%|
|Cto.2 Tiempo de operación del compresor|contador|312|309|0.473|0.640|0.640|262|100%|
|Consumo de energía sin restabecim. RTAE 5|contador|312|308|0.334|0.501|0.584|235|90%|
|Cto.2 Presión del refrigerante del condens|estacionaria|312|263|0.324|0.387|0.488|174|66%|
|Cto.1 Presión del refrigerante del condens|estacionaria|312|265|0.317|0.388|0.594|173|66%|
|Cto.2 Número de arranques del compresor|contador|312|45|0.351|0.603|0.634|169|65%|
|Cto.1 Número de arranques del compresor|contador|312|74|0.272|0.523|0.685|148|56%|
|Cto.1 Presión del refrigerante del evapora|estacionaria|312|50|0.282|0.318|0.367|42|16%|
|Cto.1 Temperatura de saturación del refrig|estacionaria|312|107|0.283|0.303|0.407|26|10%|
|Temperatura del medio ambiente|estacionaria|312|30|0.171|0.306|0.310|24|9%|
|Punto de ajuste activo de agua helada|estacionaria|312|5|0.127|0.371|0.600|23|9%|
|Cto.2 Temperatura de saturación de refrige|estacionaria|312|103|0.239|0.300|0.420|18|7%|
|Caudalímetro 5|estacionaria|312|95|0.284|0.300|0.453|16|6%|
|Temperatura de salida de agua|estacionaria|312|45|0.095|0.271|0.405|14|5%|
|Cto.2 Presión del refrigerante del evapora|estacionaria|312|50|0.232|0.298|0.454|8|3%|
|Cto.1 Temperatura de saturación del refrig|estacionaria|312|39|0.081|0.251|0.312|7|3%|
|Temperatura de entrada de agua|estacionaria|312|50|0.076|0.162|0.353|6|2%|
|Cto.2 Temperatura de saturación de refrige|estacionaria|312|39|0.079|0.126|0.297|3|1%|

### CA (29 params)

|Parámetro|Tipo|n|vals|p50|p95|p99|det@0.5|det%|
|---|---|---|---|---|---|---|---|---|
|Horas de servicio|contador|353|335|0.478|0.645|0.646|302|100%|
|Horas de carga|contador|353|332|0.459|0.625|0.625|276|91%|
|Temperatura del tanque pulmón|estacionaria|353|22|0.098|0.311|0.579|55|18%|
|Temperatura de cruceta vertical|estacionaria|353|18|0.101|0.312|0.386|53|17%|
|Temperatura del aceite|estacionaria|353|16|0.101|0.310|0.473|52|17%|
|Temperatura de chumacera lado libre|estacionaria|353|34|0.097|0.431|0.488|42|14%|
|Temperatura de cruceta horizontal|estacionaria|353|17|0.096|0.309|0.480|42|14%|
|Temperatura de aire a la descarga 2a etapa|estacionaria|353|34|0.092|0.312|0.350|35|12%|
|Temperatura de chumacera lado motor|estacionaria|353|32|0.090|0.375|0.583|30|10%|
|Temperatura de aire a la descarga 3a etapa|estacionaria|353|33|0.100|0.314|0.476|29|10%|
|Temperatura de aire a la descarga 1a etapa|estacionaria|353|38|0.095|0.303|0.361|26|9%|
|Temperatura de aspiración de 2a etapa|estacionaria|353|13|0.100|0.306|0.444|23|8%|
|Presión del tanque pulmón|estacionaria|353|9|0.086|0.351|0.599|21|7%|
|Temperatura de agua a la entrada del compr|estacionaria|353|22|0.094|0.303|0.303|21|7%|
|Temperatura de agua a la salida del compre|estacionaria|353|23|0.093|0.301|0.304|21|7%|
|Temperatura ambiente|estacionaria|353|22|0.088|0.299|0.482|18|6%|
|Presión de regulación|estacionaria|353|11|0.090|0.274|0.418|15|5%|
|Temperatura de aspiración de 3a etapa|estacionaria|353|10|0.080|0.296|0.308|14|5%|
|Temperatura de chumacera lado polea|estacionaria|353|24|0.085|0.294|0.366|12|4%|
|Temperatura del rodamiento del motor lado |estacionaria|353|27|0.087|0.297|0.328|12|4%|
|Temperatura del rodamiento del motor lado |estacionaria|353|33|0.085|0.293|0.480|12|4%|
|Temperatura del punto de rocío del secador|estacionaria|353|29|0.089|0.210|0.559|10|3%|
|Temperatura del estator|estacionaria|353|32|0.089|0.297|0.306|6|2%|
|Presión de aire a la descarga de 3a etapa|estacionaria|353|5|0.091|0.104|0.368|5|2%|
|Presión del aceite|estacionaria|353|10|0.090|0.264|0.312|5|2%|
|Presión de aire a la descarga de 2a etapa|estacionaria|353|2|0.104|0.271|0.271|3|1%|
|Presión de agua de entrada de la bomba|constante|353|1|0.104|0.104|0.104|0|0%|
|Presión de agua de salida de la bomba|constante|353|1|0.104|0.104|0.104|0|0%|
|Presión de aire a la descarga de 1a etapa|constante|353|1|0.104|0.104|0.104|0|0%|

## 3. Separación de Scores (Ranking Quality)

Sin ground truth, no se puede calcular AUC-PR. La única métrica de calidad del ranking es la separación entre scores bajos (normales) y altos (anómalos). Se mide como `(p95 - p50) / σ` — cuántas desviaciones estándar separan el percentil 95 de la mediana.

### Mejor separación (top 10)

|Parámetro|Equipo|p50|p95|p95-p50|(p95-p50)/σ|
|---|---|---|---|---|---|
|Cto.1 Temperatura de saturación del refr|Chiller|0.081|0.251|0.170|2.58|
|Presión del aceite|CA|0.090|0.264|0.174|2.55|
|Temperatura de chumacera lado libre|CA|0.097|0.431|0.333|2.53|
|Temperatura de aspiración de 3a etapa|CA|0.080|0.296|0.216|2.31|
|Temperatura de agua a la salida del comp|CA|0.093|0.301|0.208|2.27|
|Presión del tanque pulmón|CA|0.086|0.351|0.266|2.25|
|Temperatura de chumacera lado motor|CA|0.090|0.375|0.285|2.22|
|Temperatura de agua a la entrada del com|CA|0.094|0.303|0.209|2.19|
|Temperatura de chumacera lado polea|CA|0.085|0.294|0.209|2.15|
|Temperatura de aire a la descarga 2a eta|CA|0.092|0.312|0.220|2.09|

### Peor separación (últimos 10)

|Parámetro|Equipo|p50|p95|p95-p50|(p95-p50)/σ|
|---|---|---|---|---|---|
|Cto.1 Tiempo de operación del compresor|Chiller|0.481|0.648|0.167|0.85|
|Horas de servicio|CA|0.478|0.645|0.167|0.84|
|Cto.2 Presión del refrigerante del evapo|Chiller|0.232|0.298|0.066|0.57|
|Cto.2 Temperatura de saturación de refri|Chiller|0.239|0.300|0.062|0.53|
|Cto.1 Presión del refrigerante del conde|Chiller|0.317|0.388|0.071|0.51|
|Cto.2 Presión del refrigerante del conde|Chiller|0.324|0.387|0.063|0.47|
|Cto.1 Presión del refrigerante del evapo|Chiller|0.282|0.318|0.036|0.30|
|Presión de aire a la descarga de 3a etap|CA|0.091|0.104|0.014|0.24|
|Cto.1 Temperatura de saturación del refr|Chiller|0.283|0.303|0.020|0.17|
|Caudalímetro 5|Chiller|0.284|0.300|0.016|0.14|

## 4. Recomendaciones para ALPLA

#### 1. Excluir contadores monotónicos del pipeline ZENIN
7 parámetros (horas, arranques, consumo, tiempos de operación) tienen ~90% de detección porque ZENIN no maneja señales monotónicas crecientes. Cada nuevo valor es estadísticamente anómalo respecto al historial. Para contadores, usar detección de cambios en pendiente o saltos discretos.

#### 2. Excluir señales constantes
3 parámetros de presión (CA) tienen exactamente 1 valor. Score constante ~0.104. Sin variación, no hay anomalías que detectar.

#### 3. Threshold fijo 0.5 funciona para ~60% de parámetros estacionarios
Los 37 parámetros estacionarios tienen p50 en 0.076—0.324 y p95 en 0.104—0.431. Threshold 0.5 es conservador pero razonable como default. Para mayor sensibilidad, calibrar por percentil con supuesto a priori.

#### 4. Score ranking es la métrica honesta
Sin etiquetas reales no se puede reportar F1. La separación p95-p50 (mejor: 0.17—0.33, peor: 0.01—0.07) muestra que el ranking tiene poder de separación en ~60% de parámetros.

#### 5. Validar con eventos reales de mantenimiento
El paso más importante para ALPLA es cruzar los scores ZENIN con registros históricos de fallas o paradas. Si existen timestamps de mantenimiento correctivo, se puede calcular AUC-PR real.

## Archivos

- `zenin_alpla_results.json` — Resultados detallados (47 parámetros)
- `chiller_with_anomalies.csv` — Datos Chiller (18 parámetros, 312 días)
- `ca_with_anomalies.csv` — Datos CA (29 parámetros, 353 días)
- `run_zenin_alpla.py` — Script de ejecución