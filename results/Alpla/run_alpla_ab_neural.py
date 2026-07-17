"""A/B test — NeuralExpert standalone vs MoE fused, CON SPLIT TEMPORAL 80/20.

╔══════════════════════════════════════════════════════════════════════╗
║  DATA LEAKAGE FIXED: neural se entrena SOLO en el 80% inicial      ║
║  de cada serie y se evalúa SOLO en el 20% final (nunca visto).     ║
║                                                                    ║
║  Este script es un wrapper que ejecuta el runner limpio y          ║
║  reporta los resultados corregidos.                                ║
╚══════════════════════════════════════════════════════════════════════╝

Resultados completos en run_alpla_ab_clean.py → ab_clean_results.json

Resumen:

  PASO A — NeuralExpert standalone (80% train / 20% test cronológico):
    MoE 4 baseline (solo test):  median nMAE = 9.63%
    Neural standalone (solo test): median nMAE = 9.00%
    Δ = -0.63pp  (mejora genuina, ya sin leakage)
    31/47 series mejoran, 13/47 empeoran, 3/47 neutrales

  PASO B — MoE 5 expertos (neural peso 0.05) vs MoE 4:
    Con k=2: Neural NUNCA seleccionado (0/2047 preds) → resultado idéntico
    Con k=3: Neural tampoco entra en top-3 (peso normalizado ~0.006-0.041)
    → El peso 0.05 es insuficiente para que neural compita con baseline/
      statistical/taylor/kalman bajo el gating actual + top-k
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_script = os.path.join(script_dir, "run_alpla_ab_clean.py")
    if os.path.exists(clean_script):
        print("Ejecutando runner limpio (80/20 split)...")
        subprocess.run([sys.executable, clean_script], cwd=os.path.join(script_dir, ".."))
    else:
        print(f"ERROR: {clean_script} no encontrado. Resultados en ab_clean_results.json")
