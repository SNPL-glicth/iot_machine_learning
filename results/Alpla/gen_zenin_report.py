import json, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "zenin_alpla_results.json")) as f:
    data = json.load(f)

params = data["parameters"]

counter_keys = {"Horas de carga","Horas de servicio","Consumo de energía sin restabecim. RTAE 5",
    "Cto.1 Número de arranques del compresor","Cto.2 Número de arranques del compresor",
    "Cto.1 Tiempo de operación del compresor","Cto.2 Tiempo de operación del compresor"}
constant_keys = {"Presión de agua de entrada de la bomba","Presión de agua de salida de la bomba",
    "Presión de aire a la descarga de 1a etapa"}

for p in params:
    n = p["parameter"]
    if n in constant_keys: p["_class"] = "constante"
    elif n in counter_keys: p["_class"] = "contador"
    else: p["_class"] = "estacionaria"

lines = []
def L(s=""): lines.append(s)
def H(n, t): L(f"{'#'*n} {t}")
def T(h, rows):
    L("|" + "|".join(h) + "|")
    L("|" + "|".join(["---"]*len(h)) + "|")
    for r in rows:
        L("|" + "|".join(str(c) for c in r) + "|")

H(1, "ZENIN sobre ALPLA — Resultados por Parámetro")
L()
L(f"**Config:** threshold={data['metadata']['voting_threshold']}, window={data['metadata']['window_size']}")
L(f"**Parámetros:** {len(params)} (Chiller 18, CA 29)")
L()

H(2, "1. Resumen por Tipo de Señal")
L()

for cls, label, desc_util, desc_alt in [
    ("contador",
     "Contadores monotónicos (7 params)",
     "No — señal siempre creciente, ZENIN siempre detecta como anómalo porque cada nuevo valor supera el rango histórico.",
     "Usar detección de saltos discretos o cambios en pendiente (diff logarítmico o % change)."),
    ("estacionaria",
     "Señales estacionarias/cíclicas (37 params)",
     "Sí — varían alrededor de un punto de operación. Scores bajos en operación normal, altos en desviaciones.",
     "Threshold fijo 0.5 o calibración por percentil con supuesto de contaminación a priori."),
    ("constante",
     "Señales constantes (3 params, presión CA, 1 valor único)",
     "No — señal plana, scores fijos ~0.104, el detector no puede distinguir nada.",
     "Excluir del pipeline de detección. Monitorear solo cambio de estado binario."),
]:
    pp = [p for p in params if p["_class"] == cls]
    dets = [p["det_fixed_0.5"] for p in pp]
    n_detect = int(np.mean([p["n_points"] for p in pp])) - 50
    p50s = [p["score_p50"] for p in pp]
    p95s = [p["score_p95"] for p in pp]

    H(3, label)
    L()
    T(["Métrica", "Valor"], [
        ["Parámetros", str(len(pp))],
        ["Detección promedio (fixed 0.5)", f"{np.mean(dets):.0f} / ~{n_detect} pts ({np.mean(dets)/n_detect*100:.0f}%)"],
        ["Rango p50 scores", f"[{min(p50s):.3f}, {max(p50s):.3f}]"],
        ["Rango p95 scores", f"[{min(p95s):.3f}, {max(p95s):.3f}]"],
        ["Señal útil para ZENIN?", desc_util],
        ["Alternativa", desc_alt],
    ])
    L()

H(2, "2. Resultados por Parámetro")
L()
for eq in ["Chiller", "CA"]:
    pp = sorted([p for p in params if p["equipment"]==eq], key=lambda x: -x["det_fixed_0.5"])
    H(3, f"{eq} ({len(pp)} params)")
    L()
    T(["Parámetro", "Tipo", "n", "vals", "p50", "p95", "p99", "det@0.5", "det%"],
        [(p["parameter"][:42], p["_class"], p["n_points"], p["n_unique_values"],
          f"{p['score_p50']:.3f}", f"{p['score_p95']:.3f}", f"{p['score_p99']:.3f}",
          p["det_fixed_0.5"],
          f"{p['det_fixed_0.5']/(p['n_points']-50)*100:.0f}%")
         for p in pp])
    L()

H(2, "3. Separación de Scores (Ranking Quality)")
L()
L("Sin ground truth, no se puede calcular AUC-PR. La única métrica de calidad del ranking "
  "es la separación entre scores bajos (normales) y altos (anómalos). Se mide como "
  "`(p95 - p50) / σ` — cuántas desviaciones estándar separan el percentil 95 de la mediana.")
L()
H(3, "Mejor separación (top 10)")
L()
sep = sorted(
    [p for p in params if p["_class"] != "constante"],
    key=lambda p: -(p["score_p95"] - p["score_p50"]) / max(p["score_std"], 0.001))
T(["Parámetro", "Equipo", "p50", "p95", "p95-p50", "(p95-p50)/σ"],
    [(s["parameter"][:40], s["equipment"], f"{s['score_p50']:.3f}", f"{s['score_p95']:.3f}",
      f"{s['score_p95']-s['score_p50']:.3f}",
      f"{(s['score_p95']-s['score_p50'])/max(s['score_std'],0.001):.2f}")
     for s in sep[:10]])
L()
H(3, "Peor separación (últimos 10)")
L()
T(["Parámetro", "Equipo", "p50", "p95", "p95-p50", "(p95-p50)/σ"],
    [(s["parameter"][:40], s["equipment"], f"{s['score_p50']:.3f}", f"{s['score_p95']:.3f}",
      f"{s['score_p95']-s['score_p50']:.3f}",
      f"{(s['score_p95']-s['score_p50'])/max(s['score_std'],0.001):.2f}")
     for s in sep[-10:]])
L()

H(2, "4. Recomendaciones para ALPLA")
L()
recs = [
    ("Excluir contadores monotónicos del pipeline ZENIN",
     "7 parámetros (horas, arranques, consumo, tiempos de operación) tienen ~90% "
     "de detección porque ZENIN no maneja señales monotónicas crecientes. "
     "Cada nuevo valor es estadísticamente anómalo respecto al historial. "
     "Para contadores, usar detección de cambios en pendiente o saltos discretos."),
    ("Excluir señales constantes",
     "3 parámetros de presión (CA) tienen exactamente 1 valor. Score constante "
     "~0.104. Sin variación, no hay anomalías que detectar."),
    ("Threshold fijo 0.5 funciona para ~60% de parámetros estacionarios",
     "Los 37 parámetros estacionarios tienen p50 en 0.076—0.324 y p95 en "
     "0.104—0.431. Threshold 0.5 es conservador pero razonable como default. "
     "Para mayor sensibilidad, calibrar por percentil con supuesto a priori."),
    ("Score ranking es la métrica honesta",
     "Sin etiquetas reales no se puede reportar F1. La separación p95-p50 "
     "(mejor: 0.17—0.33, peor: 0.01—0.07) muestra que el ranking tiene "
     "poder de separación en ~60% de parámetros."),
    ("Validar con eventos reales de mantenimiento",
     "El paso más importante para ALPLA es cruzar los scores ZENIN con "
     "registros históricos de fallas o paradas. Si existen timestamps de "
     "mantenimiento correctivo, se puede calcular AUC-PR real."),
]
for i, (title, body) in enumerate(recs, 1):
    H(4, f"{i}. {title}")
    L(body)
    L()

H(2, "Archivos")
L()
L(f"- `zenin_alpla_results.json` — Resultados detallados ({len(params)} parámetros)")
L("- `chiller_with_anomalies.csv` — Datos Chiller (18 parámetros, 312 días)")
L("- `ca_with_anomalies.csv` — Datos CA (29 parámetros, 353 días)")
L("- `run_zenin_alpla.py` — Script de ejecución")

output = "\n".join(lines)
path = os.path.join(SCRIPT_DIR, "zenin_alpla_report.md")
with open(path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"Written to {path}, {len(lines)} lines")
