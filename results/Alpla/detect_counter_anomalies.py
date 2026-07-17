#!/usr/bin/env python3
"""
Stage independiente: detección de anomalías en contadores via median+MAD.

Bypass del Isolation Forest (que usa StandardScaler mean/std y falla
con outliers extremos en contadores). Aplica median+MAD con piso
relativo 5% sobre TODAS las filas, sin filtro IF.

Output: counter_anomalies.json  (lista de anomalías detectadas)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COUNTER_PATTERNS = [
    "número de arranques", "horas de", "tiempo de operación",
    "consumo de energía",
]

def _is_counter_param(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in COUNTER_PATTERNS)

def _mad(values):
    median = np.median(values)
    return float(np.median(np.abs(values - median)))

def _counter_threshold(median, mad):
    scale = mad if mad > 0 else float('inf')
    effective_scale = max(5.0 * scale, 0.05 * median)
    return median + effective_scale

def detect_counters(anomalies_df, equipment_id):
    counter_cols = [c for c in anomalies_df.columns
                    if c != 'iso_anomaly' and _is_counter_param(c)]
    results = []
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
            row_val = float(vals.loc[idx])
            ratio = row_val / col_median if col_median > 0 else float('inf')
            results.append({
                "equipment_id": equipment_id,
                "timestamp": idx.strftime('%Y-%m-%d %H:%M:%S'),
                "parameter": col,
                "value": row_val,
                "median": col_median,
                "mad": col_mad,
                "threshold": threshold,
                "ratio": round(ratio, 1),
                "summary": (
                    f"⚠️ Anomalía en contador '{col}': valor {row_val:,.0f} "
                    f"excede umbral robusto {threshold:,.0f} "
                    f"(mediana={col_median:,.0f}, MAD={col_mad:,.0f}). "
                    f"Esto es {ratio:.0f}× la mediana de operación normal. "
                    f"Revisar integridad del sensor o reset del contador."
                ),
            })
    return results

def main():
    print("=" * 80)
    print("  DETECCIÓN DE ANOMALÍAS EN CONTADORES (median+MAD)")
    print("=" * 80)

    excel_path = os.path.join(_SCRIPT_DIR, 'Información Chiller y CA - ZENIN.xlsx')
    output_path = os.path.join(_SCRIPT_DIR, 'counter_anomalies.json')
    all_results = []

    for sheet_name in ['Chiller', 'CA']:
        csv_path = os.path.join(_SCRIPT_DIR, f'{sheet_name.lower()}_with_anomalies.csv')
        if not os.path.exists(csv_path):
            print(f"  ⚠️  {csv_path} no encontrado — sáltando {sheet_name}")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Get equipment_id from Excel
        xl_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        equipment_id = xl_df['Equipo'].iloc[0]
        hits = detect_counters(df, equipment_id)
        all_results.extend(hits)
        if hits:
            print(f"\n  {sheet_name} ({equipment_id}): {len(hits)} anomalías en contadores")
            for h in hits:
                print(f"    {h['timestamp']} — {h['parameter']}: {h['value']:,.0f} "
                      f"(umbral={h['threshold']:,.0f}, ratio={h['ratio']:.0f}×)")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "stage": "detect_counter_anomalies",
            "total": len(all_results),
            "anomalies": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Total: {len(all_results)} anomalías guardadas en {output_path}")
    return 0 if all_results else 1

if __name__ == "__main__":
    sys.exit(main())
