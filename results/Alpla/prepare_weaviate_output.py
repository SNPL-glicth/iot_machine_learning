#!/usr/bin/env python3
"""
Preparación de salida cognitiva para integración con Weaviate.
Genera resúmenes semánticos por anomalía y patrón, listos para embeddings.

v2 — Median+MAD para parámetros tipo contador + bypass de Isolation Forest.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detección de distribuciones donde mean+std es inestable y se prefiere
# median+MAD. Dos criterios, cualquiera activa:
#   1. Cola estrecha: MAD < 1% de la mediana (pero MAD > 0, porque MAD=0
#      ocurre cuando >50% de valores son exactamente la mediana, y en ese
#      caso mean+std funciona correctamente al capturar la dispersión real).
#   2. Outliers extremos: max/median > 5 (típico de contadores con reset o
#      valor anómalo que infla artificialmente mean+std).
def _needs_robust_stats(values) -> bool:
    vals = values.dropna()
    if len(vals) < 10:
        return False
    median = float(np.median(vals))
    if median == 0:
        return False
    mad = float(np.median(np.abs(vals - median)))
    max_val = float(vals.max())
    # Criterio 1: cola estrecha (MAD pequeño relativo a mediana)
    narrow_tail = mad > 0 and mad < 0.01 * median
    # Criterio 2: outlier extremo (max >> mediana, típico de contadores).
    # Requiere MAD > 0: si MAD=0, la distribución es discreta (>50% en la
    # mediana) y mean+std captura correctamente la dispersión real.
    # Ratio 8×: calibrado para excluir sensores con picos esporádicos
    # (ej. punto de rocío 7.7×) pero capturar contadores reales
    # (Consumo 10.5×, Horas de carga 9.95×, Temp salida agua 8.8×).
    extreme_outlier = mad > 0 and max_val > 8 * median
    return narrow_tail or extreme_outlier

def _mad(values):
    """Median Absolute Deviation (robust scale estimator)."""
    median = np.median(values)
    return float(np.median(np.abs(values - median)))

def _counter_threshold(median, mad):
    """Umbral robusto para contadores: MAD con piso relativo 5%.
    
    MAD puro es peligroso cuando la distribución es muy estrecha
    (ej. MAD=8 con mediana=4512): valores con 1% de deriva normal
    activarían falsos positivos. El piso relativo evita esto.
    
    Caso especial mad==0: CUALQUIER valor != mediana es anómalo.
    """
    if mad == 0:
        return median
    scale = mad
    # Use el mayor entre: 5*MAD absoluto o 5% de la mediana
    effective_scale = max(5.0 * scale, 0.05 * median)
    return median + effective_scale


def _detect_counter_extremes(anomalies_df, equipment_id, existing_anomalies):
    """Detecta valores extremos en parámetros contador usando median+MAD.
    
    Bypass del Isolation Forest: evalúa TODAS las filas porque un contador
    con un valor físicamente imposible es una anomalía por sí mismo,
    independientemente de si otros parámetros se correlacionan.
    
    Si existe counter_anomalies.json (del stage independiente), lo usa.
    """
    counter_json = os.path.join(_SCRIPT_DIR, 'counter_anomalies.json')
    if os.path.exists(counter_json):
        with open(counter_json, encoding='utf-8') as f:
            stage_data = json.load(f)
        new_entries = []
        existing_keys = set()
        for entry in existing_anomalies:
            if entry.get("type") != "anomaly":
                continue
            ts = entry.get("timestamp", "")
            for p in entry.get("anomalous_parameters", []):
                existing_keys.add((ts, p))
        for a in stage_data.get("anomalies", []):
            if a.get("equipment_id") != equipment_id:
                continue
            ts_key = a["timestamp"]
            col = a["parameter"]
            if (ts_key, col) in existing_keys:
                continue
            new_entries.append({
                "type": "anomaly",
                "equipment_id": equipment_id,
                "timestamp": ts_key,
                "anomalous_parameters": [col],
                "counter_extreme": True,
                "summary": a["summary"],
            })
        return new_entries

    # Fallback inline: detectar columnas de cola estrecha automáticamente
    counter_cols = [c for c in anomalies_df.columns
                    if c != 'iso_anomaly' and _needs_robust_stats(anomalies_df[c])]
    new_entries = []
    existing_keys = set()

    for entry in existing_anomalies:
        if entry.get("type") != "anomaly":
            continue
        ts = entry.get("timestamp", "")
        for p in entry.get("anomalous_parameters", []):
            existing_keys.add((ts, p))

    for col in counter_cols:
        vals = anomalies_df[col].dropna()
        if len(vals) < 10:
            continue
        col_median = float(np.median(vals))
        col_mad = _mad(vals)
        threshold = _counter_threshold(col_median, col_mad)

        extreme_mask = vals > threshold
        extreme_indices = vals[extreme_mask].index

        for idx in extreme_indices:
            ts_key = idx.strftime('%Y-%m-%d %H:%M:%S')
            if (ts_key, col) in existing_keys:
                continue
            row_val = float(vals.loc[idx])
            ratio = row_val / col_median if col_median > 0 else float('inf')
            new_entries.append({
                "type": "anomaly",
                "equipment_id": equipment_id,
                "timestamp": ts_key,
                "anomalous_parameters": [col],
                "counter_extreme": True,
                "summary": (
                    f"⚠️ Anomalía en contador '{col}': valor {row_val:,.0f} "
                    f"excede umbral robusto {threshold:,.0f} "
                    f"(mediana={col_median:,.0f}, MAD={col_mad:,.0f}). "
                    f"Esto es {ratio:.0f}× la mediana de operación normal. "
                    f"Revisar integridad del sensor o reset del contador."
                ),
            })

    return new_entries


# Cargar dataset
file_path = os.path.join(_SCRIPT_DIR, 'Información Chiller y CA - ZENIN.xlsx')
xl = pd.ExcelFile(file_path)
df_chiller = pd.read_excel(xl, sheet_name='Chiller')
df_ca = pd.read_excel(xl, sheet_name='CA')

# Cargar anomalías detectadas
chiller_anomalies = pd.read_csv(os.path.join(_SCRIPT_DIR, 'chiller_with_anomalies.csv'), index_col=0, parse_dates=True)
ca_anomalies = pd.read_csv(os.path.join(_SCRIPT_DIR, 'ca_with_anomalies.csv'), index_col=0, parse_dates=True)

print("\n" + "="*80)
print("PREPARACIÓN DE SALIDA COGNITIVA PARA WEAVIATE v2 (con MAD)")
print("="*80)


def generate_semantic_summaries(df, anomalies_df, sheet_name):
    """Genera resúmenes semánticos listos para embeddings."""
    print(f"\n--- {sheet_name} ---")

    summaries = []

    # ── Resumen general del equipo ──
    equipment_id = df['Equipo'].iloc[0]
    date_range = f"{df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}"
    total_records = len(df)

    general_summary = {
        "type": "equipment_profile",
        "equipment_id": equipment_id,
        "sheet": sheet_name,
        "date_range": date_range,
        "total_records": total_records,
        "parameters_count": df['Parámetro'].nunique(),
        "summary": f"Equipo {equipment_id} con datos de {sheet_name} desde {date_range}. "
                   f"Registra {total_records} mediciones de {df['Parámetro'].nunique()} parámetros diferentes. "
                   f"Los datos incluyen mediciones de temperatura, presión, consumo energético y horas de operación. "
                   f"El dataset es adecuado para análisis predictivo y detección de anomalías operativas."
    }
    summaries.append(general_summary)

    # ── Resúmenes por parámetro ──
    for param in df['Parámetro'].unique():
        param_data = df[df['Parámetro'] == param]
        param_values = param_data['Valor']

        param_summary = {
            "type": "parameter_profile",
            "equipment_id": equipment_id,
            "parameter": param,
            "unit": param_data['UM'].iloc[0],
            "records": len(param_data),
            "mean": float(param_values.mean()),
            "std": float(param_values.std()),
            "min": float(param_values.min()),
            "max": float(param_values.max()),
            "summary": f"Parámetro '{param}' medido en {param_data['UM'].iloc[0]} con {len(param_data)} registros. "
                       f"Rango de operación: {param_values.min():.2f} a {param_values.max():.2f}. "
                       f"Promedio: {param_values.mean():.2f} con desviación estándar de {param_values.std():.2f}. "
                       f"Indicador de variabilidad: {param_values.std()/param_values.mean()*100 if param_values.mean() != 0 else 0:.1f}%."
        }
        summaries.append(param_summary)

    # ── Resúmenes de anomalías (Etapa 1 + 2 mejorada) ──
    # Etapa 1: Isolation Forest (solo para parámetros NO contador se respeta el filtro)
    # Etapa 2: mean+2*std para normales, median+3*MAD para contadores
    anomaly_records = anomalies_df[anomalies_df['iso_anomaly'] == True]
    if len(anomaly_records) > 0:
        for idx, row in anomaly_records.iterrows():
            anomalous_params = []
            for col in anomalies_df.columns:
                if col == 'iso_anomaly' or pd.isna(row[col]):
                    continue
                if _needs_robust_stats(anomalies_df[col]):
                    # Robust statistics para distribuciones de cola estrecha
                    vals = anomalies_df[col].dropna()
                    col_median = float(np.median(vals))
                    col_mad = _mad(vals)
                    scale = col_mad if col_mad > 0 else float(np.std(vals) * 0.01)
                    threshold = col_median + 3.0 * scale
                    if row[col] > threshold:
                        anomalous_params.append(col)
                else:
                    # Mean+std tradicional para distribuciones normales
                    col_mean = anomalies_df[col].mean()
                    col_std = anomalies_df[col].std()
                    if abs(row[col] - col_mean) > 2 * col_std:
                        anomalous_params.append(col)

            anomaly_summary = {
                "type": "anomaly",
                "equipment_id": equipment_id,
                "timestamp": idx.strftime('%Y-%m-%d %H:%M:%S'),
                "anomalous_parameters": anomalous_params,
                "summary": f"Anomalía detectada en equipo {equipment_id} el {idx.strftime('%Y-%m-%d %H:%M:%S')}. "
                           f"Parámetros fuera de rango: {', '.join(anomalous_params) if anomalous_params else 'múltiples parámetros'}. "
                           f"Esta anomalía puede indicar comportamiento operativo anormal, falla de sensor o condición de mantenimiento requerida. "
                           f"Se recomienda revisión manual de los registros operativos en ese período."
            }
            summaries.append(anomaly_summary)

    # ── Pasada extra: contadores con valores extremos (bypass IF) ──
    # El Isolation Forest usa StandardScaler (mean/std) que falla con
    # outliers extremos en contadores. Esta pasada detecta esos casos
    # usando median+MAD en TODAS las filas, sin filtro IF.
    counter_extremes = _detect_counter_extremes(
        anomalies_df, equipment_id, summaries
    )
    summaries.extend(counter_extremes)

    # ── Resúmenes de patrones temporales ──
    df_sorted = df.sort_values('Fecha')
    for param in df['Parámetro'].unique()[:5]:  # Solo primeros 5 para no saturar
        param_data = df[df['Parámetro'] == param].sort_values('Fecha')
        if len(param_data) > 50:
            values = param_data['Valor'].values
            if len(values) > 10:
                mid = len(values) // 2
                first_half_mean = values[:mid].mean()
                second_half_mean = values[mid:].mean()
                trend = "creciente" if second_half_mean > first_half_mean else "decreciente"
                trend_pct = abs(second_half_mean - first_half_mean) / first_half_mean * 100 if first_half_mean != 0 else 0

                if trend_pct > 5:
                    pattern_summary = {
                        "type": "temporal_pattern",
                        "equipment_id": equipment_id,
                        "parameter": param,
                        "trend": trend,
                        "trend_percentage": float(trend_pct),
                        "summary": f"El parámetro '{param}' muestra una tendencia {trend} del {trend_pct:.1f}% "
                                   f"entre la primera y segunda mitad del período analizado. "
                                   f"Esto puede indicar degradación de equipo, cambio de condiciones operativas "
                                   f"o ajuste de setpoints. Monitoreo continuo recomendado."
                    }
                    summaries.append(pattern_summary)

    return summaries


# Generar resúmenes
chiller_summaries = generate_semantic_summaries(df_chiller, chiller_anomalies, 'Chiller')
ca_summaries = generate_semantic_summaries(df_ca, ca_anomalies, 'CA')

# Combinar todos los resúmenes
all_summaries = chiller_summaries + ca_summaries

# Guardar en formato JSON para Weaviate
weaviate_output = {
    "metadata": {
        "source": "ALPLA Industrial Dataset",
        "date_generated": datetime.now().isoformat(),
        "total_summaries": len(all_summaries),
        "sheets": ["Chiller", "CA"]
    },
    "summaries": all_summaries
}

with open(os.path.join(_SCRIPT_DIR, 'weaviate_ready_output.json'), 'w', encoding='utf-8') as f:
    json.dump(weaviate_output, f, ensure_ascii=False, indent=2)

print(f"\nTotal de resúmenes generados: {len(all_summaries)}")
print(f"  - Perfiles de equipo: {sum(1 for s in all_summaries if s['type'] == 'equipment_profile')}")
print(f"  - Perfiles de parámetros: {sum(1 for s in all_summaries if s['type'] == 'parameter_profile')}")
print(f"  - Anomalías: {sum(1 for s in all_summaries if s['type'] == 'anomaly')}")
print(f"  - Patrones temporales: {sum(1 for s in all_summaries if s['type'] == 'temporal_pattern')}")

# Mostrar las nuevas detecciones de contadores
counter_anomalies = [s for s in all_summaries if s.get('counter_extreme')]
if counter_anomalies:
    print(f"\n⚠️  Anomalías detectadas por median+MAD (bypass IF): {len(counter_anomalies)}")
    for a in counter_anomalies:
        print(f"    {a['timestamp']} — {a['anomalous_parameters'][0]}")
        print(f"      {a['summary']}")

print(f"\nSalida guardada en: weaviate_ready_output.json")

# Guardar también versión simplificada para referencia rápida
with open(os.path.join(_SCRIPT_DIR, 'weaviate_summaries_readable.txt'), 'w', encoding='utf-8') as f:
    f.write("RESÚMENES SEMÁNTICOS PARA WEAVIATE v2 (MAD)\n")
    f.write("="*80 + "\n\n")

    for summary in all_summaries:
        f.write(f"Tipo: {summary['type']}\n")
        f.write(f"Equipo: {summary.get('equipment_id', 'N/A')}\n")
        f.write(f"Resumen: {summary['summary']}\n")
        f.write("-"*80 + "\n\n")

print("Versión legible guardada en: weaviate_summaries_readable.txt")
