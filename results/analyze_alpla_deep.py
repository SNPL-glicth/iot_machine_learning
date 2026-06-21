#!/usr/bin/env python3
"""
Análisis profundo del dataset industrial de ALPLA para ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar dataset
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_SCRIPT_DIR, 'Información Chiller y CA - ZENIN.xlsx')
xl = pd.ExcelFile(file_path)
df_chiller = pd.read_excel(xl, sheet_name='Chiller')
df_ca = pd.read_excel(xl, sheet_name='CA')

print("\n" + "="*80)
print("ANÁLISIS EXPLORATORIO PROFUNDO")
print("="*80)

def analyze_deep(df, sheet_name):
    """Análisis exploratorio profundo."""
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet_name}")
    print(f"{'='*80}")
    
    # Cardinalidad de equipos y parámetros
    print(f"\n--- CARDINALIDAD ---")
    print(f"Equipos únicos: {df['Equipo'].nunique()}")
    print(f"Parámetros únicos: {df['Parámetro'].nunique()}")
    print(f"Unidades de medida únicas: {df['UM'].nunique()}")
    
    print(f"\nEquipos: {df['Equipo'].unique()}")
    print(f"\nParámetros: {df['Parámetro'].unique()}")
    print(f"\nUnidades: {df['UM'].unique()}")
    
    # Análisis temporal
    print(f"\n--- ANÁLISIS TEMPORAL ---")
    df_sorted = df.sort_values('Fecha')
    print(f"Rango temporal: {df_sorted['Fecha'].min()} a {df_sorted['Fecha'].max()}")
    print(f"Días totales: {(df_sorted['Fecha'].max() - df_sorted['Fecha'].min()).days}")
    
    # Calcular intervalos entre timestamps
    intervals = df_sorted['Fecha'].diff().dropna()
    print(f"Intervalo medio: {intervals.mean()}")
    print(f"Intervalo mediana: {intervals.median()}")
    print(f"Intervalo mínimo: {intervals.min()}")
    print(f"Intervalo máximo: {intervals.max()}")
    
    # Densidad de datos por parámetro
    print(f"\n--- DENSIDAD POR PARÁMETRO ---")
    for param in df['Parámetro'].unique():
        param_data = df[df['Parámetro'] == param]
        print(f"{param}: {len(param_data)} registros")
        if len(param_data) > 1:
            param_sorted = param_data.sort_values('Fecha')
            param_intervals = param_sorted['Fecha'].diff().dropna()
            print(f"  Intervalo medio: {param_intervals.mean()}")
    
    # Estadísticas por parámetro
    print(f"\n--- ESTADÍSTICAS POR PARÁMETRO ---")
    for param in df['Parámetro'].unique():
        param_data = df[df['Parámetro'] == param]['Valor']
        print(f"\n{param}:")
        print(f"  Media: {param_data.mean():.2f}")
        print(f"  Mediana: {param_data.median():.2f}")
        print(f"  Std: {param_data.std():.2f}")
        print(f"  Min: {param_data.min():.2f}")
        print(f"  Max: {param_data.max():.2f}")
        print(f"  Rango: {param_data.max() - param_data.min():.2f}")
    
    return df_sorted

# Analizar ambos sheets
chiller_sorted = analyze_deep(df_chiller, 'Chiller')
ca_sorted = analyze_deep(df_ca, 'CA')

# Pivotar datos para análisis multivariado
print("\n" + "="*80)
print("PREPARACIÓN PARA ANÁLISIS MULTIVARIADO")
print("="*80)

def pivot_for_analysis(df, sheet_name):
    """Pivotear datos para análisis de series temporales."""
    print(f"\n--- {sheet_name} ---")
    
    # Pivotar: Fecha como índice, Parámetros como columnas (pivot_table para duplicados)
    df_pivot = df.pivot_table(index='Fecha', columns='Parámetro', values='Valor', aggfunc='first')
    
    print(f"Shape después de pivot: {df_pivot.shape}")
    print(f"Columnas: {list(df_pivot.columns)}")
    print(f"Valores nulos por columna:")
    print(df_pivot.isnull().sum())
    
    return df_pivot

chiller_pivot = pivot_for_analysis(df_chiller, 'Chiller')
ca_pivot = pivot_for_analysis(df_ca, 'CA')

# Análisis de correlaciones
print("\n" + "="*80)
print("ANÁLISIS DE CORRELACIONES")
print("="*80)

def analyze_correlations(df_pivot, sheet_name):
    """Analizar correlaciones entre parámetros."""
    print(f"\n--- {sheet_name} ---")
    
    numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df_pivot[numeric_cols].corr()
        print("Matriz de correlación:")
        print(corr_matrix)
        
        n = len(numeric_cols)
        fig_w = max(10, n * 0.55)
        fig_h = max(8, n * 0.45)
        fs = max(4, min(9, 180 // n))
        
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, linecolor='gray',
            annot_kws={'size': fs},
            xticklabels=True, yticklabels=True,
        )
        plt.title(f'Matriz de Correlación — {sheet_name} ({n} parámetros)', fontsize=fs + 4, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=fs)
        plt.yticks(rotation=0, fontsize=fs)
        plt.tight_layout()
        
        corr_dir = os.path.join(_SCRIPT_DIR, 'correlations')
        os.makedirs(corr_dir, exist_ok=True)
        out = os.path.join(corr_dir, f'{sheet_name.lower()}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap guardado: {out}")
    else:
        print("Menos de 2 columnas numéricas, no se puede calcular correlación")

analyze_correlations(chiller_pivot, 'Chiller')
analyze_correlations(ca_pivot, 'CA')

# Detección de outliers
print("\n" + "="*80)
print("DETECCIÓN DE OUTLIERS")
print("="*80)

def detect_outliers(df_pivot, sheet_name):
    """Detectar outliers usando IQR y Z-score."""
    print(f"\n--- {sheet_name} ---")
    
    numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df_pivot[col].dropna()
        
        # IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
        
        # Z-score
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers_zscore = data[z_scores > 3]
        
        print(f"\n{col}:")
        print(f"  Outliers IQR: {len(outliers_iqr)} ({100*len(outliers_iqr)/len(data):.1f}%)")
        print(f"  Outliers Z-score (>3): {len(outliers_zscore)} ({100*len(outliers_zscore)/len(data):.1f}%)")
        
        if len(outliers_iqr) > 0:
            print(f"  Rango outliers IQR: [{outliers_iqr.min():.2f}, {outliers_iqr.max():.2f}]")

detect_outliers(chiller_pivot, 'Chiller')
detect_outliers(ca_pivot, 'CA')

# Visualizaciones temporales
print("\n" + "="*80)
print("VISUALIZACIONES TEMPORALES")
print("="*80)

def plot_time_series(df_pivot, sheet_name):
    """Generar visualizaciones de series temporales."""
    print(f"\n--- {sheet_name} ---")
    
    numeric_cols = df_pivot.select_dtypes(include=[np.number]).columns
    ts_dir = os.path.join(_SCRIPT_DIR, 'timeseries', sheet_name.lower())
    os.makedirs(ts_dir, exist_ok=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(12, 4))
        df_pivot[col].plot(title=f'{col} - {sheet_name}')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.xticks(rotation=45)
        plt.tight_layout()
        safe = col.replace(' ', '_').replace('/', '_').replace('.', '')
        safe = ''.join(c for c in safe if c.isalnum() or c in ('_', '-'))
        out = os.path.join(ts_dir, f'{safe}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {out}")

plot_time_series(chiller_pivot, 'Chiller')
plot_time_series(ca_pivot, 'CA')

# Guardar resumen
with open(os.path.join(_SCRIPT_DIR, 'analysis_summary.txt'), 'w') as f:
    f.write("ANÁLISIS EXPLORATORIO PROFUNDO - ALPLA\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHEET: CHILLER\n")
    f.write(f"Equipos únicos: {df_chiller['Equipo'].nunique()}\n")
    f.write(f"Parámetros únicos: {df_chiller['Parámetro'].nunique()}\n")
    f.write(f"Rango temporal: {chiller_sorted['Fecha'].min()} a {chiller_sorted['Fecha'].max()}\n")
    f.write(f"Días totales: {(chiller_sorted['Fecha'].max() - chiller_sorted['Fecha'].min()).days}\n")
    
    f.write("\nSHEET: CA\n")
    f.write(f"Equipos únicos: {df_ca['Equipo'].nunique()}\n")
    f.write(f"Parámetros únicos: {df_ca['Parámetro'].nunique()}\n")
    f.write(f"Rango temporal: {ca_sorted['Fecha'].min()} a {ca_sorted['Fecha'].max()}\n")
    f.write(f"Días totales: {(ca_sorted['Fecha'].max() - ca_sorted['Fecha'].min()).days}\n")

print("\nAnálisis exploratorio completado. Visualizaciones guardadas.")
