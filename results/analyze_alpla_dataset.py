#!/usr/bin/env python3
"""
Análisis del dataset industrial de ALPLA para determinar idoneidad para ML.
Dataset: Información Chiller y CA - ZENIN.xlsx
Sheets: Chiller, CA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar dataset
file_path = '/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/Información Chiller y CA - ZENIN.xlsx'
xl = pd.ExcelFile(file_path)
print(f"Sheet names: {xl.sheet_names}")

# Cargar ambos sheets
df_chiller = pd.read_excel(xl, sheet_name='Chiller')
df_ca = pd.read_excel(xl, sheet_name='CA')

print("\n" + "="*80)
print("PERFILADO DEL DATASET")
print("="*80)

print("\n--- SHEET: CHILLER ---")
print(f"Shape: {df_chiller.shape}")
print(f"Columnas: {list(df_chiller.columns)}")
print(f"\nTipos de datos:")
print(df_chiller.dtypes)
print(f"\nValores nulos:")
print(df_chiller.isnull().sum())
print(f"\nPrimeras filas:")
print(df_chiller.head())
print(f"\nEstadísticas descriptivas:")
print(df_chiller.describe())

print("\n--- SHEET: CA ---")
print(f"Shape: {df_ca.shape}")
print(f"Columnas: {list(df_ca.columns)}")
print(f"\nTipos de datos:")
print(df_ca.dtypes)
print(f"\nValores nulos:")
print(df_ca.isnull().sum())
print(f"\nPrimeras filas:")
print(df_ca.head())
print(f"\nEstadísticas descriptivas:")
print(df_ca.describe())

# Detectar columnas temporales
print("\n" + "="*80)
print("ANÁLISIS DE FRECUENCIA TEMPORAL")
print("="*80)

def analyze_temporal_frequency(df, sheet_name):
    """Analiza frecuencia temporal y continuidad de series."""
    print(f"\n--- {sheet_name} ---")
    
    # Buscar columnas que podrían ser temporales
    temporal_cols = []
    for col in df.columns:
        if df[col].dtype in ['datetime64[ns]', 'object']:
            # Intentar parsear como fecha
            try:
                pd.to_datetime(df[col], errors='coerce')
                temporal_cols.append(col)
            except:
                pass
    
    print(f"Columnas temporales detectadas: {temporal_cols}")
    
    if temporal_cols:
        for col in temporal_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    print(f"\nColumna: {col}")
                    print(f"  Rango: {df[col].min()} a {df[col].max()}")
                    print(f"  Valores nulos: {df[col].isnull().sum()}")
                    
                    # Calcular intervalos entre timestamps
                    valid_dates = df[col].dropna().sort_values()
                    if len(valid_dates) > 1:
                        intervals = valid_dates.diff().dropna()
                        print(f"  Intervalo medio: {intervals.mean()}")
                        print(f"  Intervalo mínimo: {intervals.min()}")
                        print(f"  Intervalo máximo: {intervals.max()}")
            except Exception as e:
                print(f"  Error procesando {col}: {e}")
    else:
        print("No se detectaron columnas temporales explícitas")

analyze_temporal_frequency(df_chiller, 'Chiller')
analyze_temporal_frequency(df_ca, 'CA')

# Cardinalidad de equipos y parámetros
print("\n" + "="*80)
print("CARDINALIDAD DE EQUIPOS Y PARÁMETROS")
print("="*80)

def analyze_cardinality(df, sheet_name):
    """Analiza cardinalidad de columnas categóricas."""
    print(f"\n--- {sheet_name} ---")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} valores únicos")
            if unique_count <= 20:
                print(f"    Valores: {df[col].unique()[:10]}")

analyze_cardinality(df_chiller, 'Chiller')
analyze_cardinality(df_ca, 'CA')

# Guardar resultados básicos
print("\n" + "="*80)
print("GUARDANDO INFORMACIÓN DEL DATASET")
print("="*80)

with open('/home/nicolas/Documentos/Iot_System/iot_machine_learning/Datos/dataset_profile.txt', 'w') as f:
    f.write("DATASET ALPLA - PERFILADO\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHEET: CHILLER\n")
    f.write(f"Shape: {df_chiller.shape}\n")
    f.write(f"Columnas: {list(df_chiller.columns)}\n")
    f.write(f"Tipos:\n{df_chiller.dtypes}\n")
    f.write(f"Valores nulos:\n{df_chiller.isnull().sum()}\n")
    f.write(f"Estadísticas:\n{df_chiller.describe()}\n\n")
    
    f.write("SHEET: CA\n")
    f.write(f"Shape: {df_ca.shape}\n")
    f.write(f"Columnas: {list(df_ca.columns)}\n")
    f.write(f"Tipos:\n{df_ca.dtypes}\n")
    f.write(f"Valores nulos:\n{df_ca.isnull().sum()}\n")
    f.write(f"Estadísticas:\n{df_ca.describe()}\n")

print("Perfilado guardado en dataset_profile.txt")
