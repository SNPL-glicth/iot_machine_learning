#!/usr/bin/env python3
"""
Preparación de salida cognitiva para integración con Weaviate.
Genera resúmenes semánticos por anomalía y patrón, listos para embeddings.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Cargar dataset
file_path = '/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/Información Chiller y CA - ZENIN.xlsx'
xl = pd.ExcelFile(file_path)
df_chiller = pd.read_excel(xl, sheet_name='Chiller')
df_ca = pd.read_excel(xl, sheet_name='CA')

# Cargar anomalías detectadas
chiller_anomalies = pd.read_csv('/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/chiller_with_anomalies.csv', index_col=0, parse_dates=True)
ca_anomalies = pd.read_csv('/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/ca_with_anomalies.csv', index_col=0, parse_dates=True)

print("\n" + "="*80)
print("PREPARACIÓN DE SALIDA COGNITIVA PARA WEAVIATE")
print("="*80)

def generate_semantic_summaries(df, anomalies_df, sheet_name):
    """Genera resúmenes semánticos listos para embeddings."""
    print(f"\n--- {sheet_name} ---")
    
    summaries = []
    
    # Resumen general del equipo
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
    
    # Resúmenes por parámetro
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
    
    # Resúmenes de anomalías
    anomaly_records = anomalies_df[anomalies_df['iso_anomaly'] == True]
    if len(anomaly_records) > 0:
        for idx, row in anomaly_records.iterrows():
            # Encontrar qué parámetros causaron la anomalía
            anomalous_params = []
            for col in anomalies_df.columns:
                if col != 'iso_anomaly' and pd.notna(row[col]):
                    # Comparar con media global
                    col_mean = anomalies_df[col].mean()
                    if abs(row[col] - col_mean) > 2 * anomalies_df[col].std():
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
    
    # Resúmenes de patrones temporales
    df_sorted = df.sort_values('Fecha')
    for param in df['Parámetro'].unique()[:5]:  # Solo primeros 5 para no saturar
        param_data = df[df['Parámetro'] == param].sort_values('Fecha')
        if len(param_data) > 50:
            # Detectar tendencias
            values = param_data['Valor'].values
            if len(values) > 10:
                # Tendencia simple (comparar primera vs última mitad)
                mid = len(values) // 2
                first_half_mean = values[:mid].mean()
                second_half_mean = values[mid:].mean()
                trend = "creciente" if second_half_mean > first_half_mean else "decreciente"
                trend_pct = abs(second_half_mean - first_half_mean) / first_half_mean * 100 if first_half_mean != 0 else 0
                
                if trend_pct > 5:  # Solo si hay tendencia significativa
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

with open('/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/weaviate_ready_output.json', 'w', encoding='utf-8') as f:
    json.dump(weaviate_output, f, ensure_ascii=False, indent=2)

print(f"\nTotal de resúmenes generados: {len(all_summaries)}")
print(f"  - Perfiles de equipo: {sum(1 for s in all_summaries if s['type'] == 'equipment_profile')}")
print(f"  - Perfiles de parámetros: {sum(1 for s in all_summaries if s['type'] == 'parameter_profile')}")
print(f"  - Anomalías: {sum(1 for s in all_summaries if s['type'] == 'anomaly')}")
print(f"  - Patrones temporales: {sum(1 for s in all_summaries if s['type'] == 'temporal_pattern')}")

print(f"\nSalida guardada en: weaviate_ready_output.json")
print("Formato listo para ingestión en Weaviate con embeddings.")

# Guardar también versión simplificada para referencia rápida
with open('/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/weaviate_summaries_readable.txt', 'w', encoding='utf-8') as f:
    f.write("RESÚMENES SEMÁNTICOS PARA WEAVIATE\n")
    f.write("="*80 + "\n\n")
    
    for summary in all_summaries:
        f.write(f"Tipo: {summary['type']}\n")
        f.write(f"Equipo: {summary.get('equipment_id', 'N/A')}\n")
        f.write(f"Resumen: {summary['summary']}\n")
        f.write("-"*80 + "\n\n")

print("Versión legible guardada en: weaviate_summaries_readable.txt")
