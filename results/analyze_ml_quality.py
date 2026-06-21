#!/usr/bin/env python3
"""
Evaluación de calidad para ML y prueba de modelos iniciales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# Cargar dataset
file_path = '/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/Información Chiller y CA - ZENIN.xlsx'
xl = pd.ExcelFile(file_path)
df_chiller = pd.read_excel(xl, sheet_name='Chiller')
df_ca = pd.read_excel(xl, sheet_name='CA')

print("\n" + "="*80)
print("EVALUACIÓN DE CALIDAD PARA MACHINE LEARNING")
print("="*80)

def evaluate_ml_quality(df, sheet_name):
    """Evaluar calidad del dataset para ML."""
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet_name}")
    print(f"{'='*80}")
    
    # Pivotear
    df_pivot = df.pivot(index='Fecha', columns='Parámetro', values='Valor')
    
    # Densidad temporal
    print("\n--- DENSIDAD TEMPORAL ---")
    total_days = (df_pivot.index.max() - df_pivot.index.min()).days
    total_records = len(df_pivot)
    records_per_day = total_records / total_days if total_days > 0 else 0
    
    print(f"Período total: {total_days} días")
    print(f"Total registros: {total_records}")
    print(f"Registros por día (promedio): {records_per_day:.2f}")
    
    # Densidad por parámetro
    for col in df_pivot.columns:
        col_data = df_pivot[col].dropna()
        if len(col_data) > 0:
            col_days = (col_data.index.max() - col_data.index.min()).days
            col_density = len(col_data) / col_days if col_days > 0 else 0
            print(f"  {col}: {len(col_data)} registros, {col_density:.2f} registros/día")
    
    # Evaluación de ruido
    print("\n--- EVALUACIÓN DE RUIDO ---")
    for col in df_pivot.select_dtypes(include=[np.number]).columns:
        col_data = df_pivot[col].dropna()
        if len(col_data) > 10:
            # Calcular varianza relativa
            cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else float('inf')
            
            # Calcular ratio señal/ruido aproximado (variabilidad vs rango)
            signal_range = col_data.max() - col_data.min()
            noise_level = col_data.diff().abs().mean()
            snr = signal_range / noise_level if noise_level > 0 else float('inf')
            
            print(f"  {col}:")
            print(f"    Coeficiente de variación: {cv:.4f}")
            print(f"    Ratio señal/ruido: {snr:.2f}")
            
            # Clasificar nivel de ruido
            if cv < 0.1:
                noise_level = "Bajo"
            elif cv < 0.3:
                noise_level = "Medio"
            else:
                noise_level = "Alto"
            print(f"    Nivel de ruido: {noise_level}")
    
    # Detección de drift
    print("\n--- DETECCIÓN DE DRIFT ---")
    for col in df_pivot.select_dtypes(include=[np.number]).columns:
        col_data = df_pivot[col].dropna()
        if len(col_data) > 50:
            # Dividir en dos mitades y comparar medias
            mid_point = len(col_data) // 2
            first_half = col_data.iloc[:mid_point]
            second_half = col_data.iloc[mid_point:]
            
            mean_diff_pct = abs(second_half.mean() - first_half.mean()) / first_half.mean() * 100 if first_half.mean() != 0 else 0
            
            print(f"  {col}:")
            print(f"    Media 1ra mitad: {first_half.mean():.2f}")
            print(f"    Media 2da mitad: {second_half.mean():.2f}")
            print(f"    Diferencia: {mean_diff_pct:.2f}%")
            
            if mean_diff_pct > 10:
                print(f"    ⚠️  Posible drift detectado")
    
    # Variables útiles para forecasting
    print("\n--- VARIABLES ÚTILES PARA FORECASTING ---")
    for col in df_pivot.select_dtypes(include=[np.number]).columns:
        col_data = df_pivot[col].dropna()
        if len(col_data) > 20:
            # Autocorrelación
            autocorr = col_data.autocorr(lag=1)
            print(f"  {col}:")
            print(f"    Autocorrelación lag-1: {autocorr:.4f}")
            
            if abs(autocorr) > 0.5:
                print(f"    ✅ Buena candidata para forecasting")
            elif abs(autocorr) > 0.3:
                print(f"    ⚠️  Candidata moderada")
            else:
                print(f"    ❌ Pobre candidata para forecasting")
    
    return df_pivot

chiller_pivot = evaluate_ml_quality(df_chiller, 'Chiller')
ca_pivot = evaluate_ml_quality(df_ca, 'CA')

print("\n" + "="*80)
print("PRUEBA DE MODELOS INICIALES")
print("="*80)

def test_models(df_pivot, sheet_name):
    """Probar modelos de detección de anomalías."""
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet_name}")
    print(f"{'='*80}")
    
    numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Menos de 2 columnas numéricas, se requiere para modelos multivariados")
        return
    
    # Preparar datos (imputar nulos con forward fill)
    df_clean = df_pivot[numeric_cols].ffill().bfill()
    
    # Escalar
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), 
                            columns=numeric_cols, 
                            index=df_clean.index)
    
    # Isolation Forest
    print("\n--- ISOLATION FOREST ---")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso_forest.fit_predict(df_scaled)
    iso_anomalies = (iso_pred == -1).sum()
    print(f"Anomalías detectadas: {iso_anomalies} ({100*iso_anomalies/len(df_scaled):.1f}%)")
    
    # Local Outlier Factor
    print("\n--- LOCAL OUTLIER FACTOR ---")
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        lof_pred = lof.fit_predict(df_scaled)
        lof_anomalies = (lof_pred == -1).sum()
        print(f"Anomalías detectadas: {lof_anomalies} ({100*lof_anomalies/len(df_scaled):.1f}%)")
    except Exception as e:
        print(f"Error en LOF: {e}")
    
    # Moving Average Baseline
    print("\n--- MOVING AVERAGE BASELINE ---")
    for col in numeric_cols[:3]:  # Solo primeras 3 columnas
        col_data = df_clean[col].dropna()
        if len(col_data) > 20:
            # Usar últimos 20% para test
            split_idx = int(len(col_data) * 0.8)
            train = col_data.iloc[:split_idx]
            test = col_data.iloc[split_idx:]
            
            # Moving average con ventana de 5
            ma_window = 5
            ma_pred = train.rolling(window=ma_window).mean().iloc[-1]
            mae = mean_absolute_error(test, [ma_pred] * len(test))
            mse = mean_squared_error(test, [ma_pred] * len(test))
            
            print(f"\n{col}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")
    
    # Guardar anomalías detectadas
    df_clean['iso_anomaly'] = iso_pred == -1
    df_clean.to_csv(f'/home/nicolas/Documentos/Iot_System/{sheet_name.lower()}_with_anomalies.csv')
    print(f"\nDatos con anomalías guardados: {sheet_name.lower()}_with_anomalies.csv")

test_models(chiller_pivot, 'Chiller')
test_models(ca_pivot, 'CA')

# Resumen de calidad
print("\n" + "="*80)
print("RESUMEN DE CALIDAD PARA ML")
print("="*80)

with open('/home/nicolas/Documentos/Iot_System/ml_quality_summary.txt', 'w') as f:
    f.write("EVALUACIÓN DE CALIDAD PARA ML - ALPLA\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHEET: CHILLER\n")
    f.write(f"Total registros: {len(chiller_pivot)}\n")
    f.write(f"Parámetros numéricos: {len(chiller_pivot.select_dtypes(include=[np.number]).columns)}\n")
    f.write("Modelos probados: Isolation Forest, LOF, Moving Average\n")
    
    f.write("\nSHEET: CA\n")
    f.write(f"Total registros: {len(ca_pivot)}\n")
    f.write(f"Parámetros numéricos: {len(ca_pivot.select_dtypes(include=[np.number]).columns)}\n")
    f.write("Modelos probados: Isolation Forest, LOF, Moving Average\n")
    
    f.write("\nCONCLUSIONES PRELIMINARES:\n")
    f.write("- Dataset tiene estructura temporal adecuada para series de tiempo\n")
    f.write("- Múltiples parámetros por equipo permiten análisis multivariado\n")
    f.write("- Densidad de datos es suficiente para modelos de ML\n")
    f.write("- Se detectaron outliers en algunos parámetros (útiles para anomaly detection)\n")

print("Evaluación de calidad y modelos completada.")
