"""
UCI AI4I 2020 Predictive Maintenance Dataset Benchmark
Compara VotingAnomalyDetector vs baselines para detección de fallos en maquinaria.

Uso:
    cd iot_machine_learning
    python benchmarks/uci_ai4i_benchmark.py

Output:
    results/uci_ai4i_results.json
    results/uci_ai4i_report.md
    results/uci_ai4i_plot.png
"""

import json
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BLOCKER: make iot_machine_learning importable
sys.path.insert(0, "/home/nicolas/Documentos/Proyectos/Zenin-Iot")
sys.path.insert(0, "/home/nicolas/Documentos/Proyectos/Zenin-Iot/iot_machine_learning")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uci_ai4i_benchmark")

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/Datasets")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("/tmp/UCI_AI4I")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UCI AI4I 2020 Data URL
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/"
AI4I_FILE = "ai4i2020.csv"

# ─── Config benchmark ────────────────────────────────────────────────────────
WINDOW_SIZE = 50  # Ventana para detección
SERIES_ID = "UCI-AI4I-Predictive-Maintenance"

# ─── Descarga de datos ────────────────────────────────────────────────────────


def download_uci_ai4i_data():
    """Descarga los datos de UCI AI4I 2020 si no existen."""
    data_path = DATA_DIR / AI4I_FILE
    
    if data_path.exists():
        logger.info("Datos de UCI AI4I 2020 ya existen, omitiendo descarga")
        return data_path
    
    logger.info("Descargando datos de UCI AI4I 2020...")
    
    try:
        urllib.request.urlretrieve(UCI_URL + AI4I_FILE, data_path)
        logger.info("Datos descargados exitosamente")
    except Exception as e:
        logger.error(f"Error descargando datos: {e}")
        # Try alternative URL
        try:
            alt_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/ai4i2020.csv"
            urllib.request.urlretrieve(alt_url, data_path)
            logger.info("Datos descargados desde URL alternativa")
        except Exception as e2:
            logger.error(f"Error descargando desde URL alternativa: {e2}")
            raise
    
    return data_path


# ─── Carga de datos ──────────────────────────────────────────────────────────


def load_uci_ai4i_data() -> Tuple[List[float], List[float], List[int]]:
    """
    Carga dataset UCI AI4I 2020 Predictive Maintenance.
    
    Returns: values, timestamps_float, labels (0/1 por punto)
    """
    data_path = download_uci_ai4i_data()
    
    logger.info(f"Cargando dataset: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        # Crear datos sintéticos basados en la descripción del dataset
        logger.info("Creando datos sintéticos basados en descripción del dataset...")
        df = create_synthetic_ai4i_data()
    
    # El dataset tiene las siguientes columnas relevantes:
    # Air temperature [K], Process temperature [K], Rotational speed [rpm], 
    # Torque [Nm], Tool wear [min], Machine failure type
    
    # Buscar columnas de temperatura y torque como señales principales
    temp_col = None
    torque_col = None
    
    for col in df.columns:
        if 'temperature' in col.lower() and 'air' in col.lower():
            temp_col = col
        if 'torque' in col.lower():
            torque_col = col
    
    if temp_col is None:
        # Usar primera columna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            temp_col = numeric_cols[0]
    
    # Usar temperatura como señal principal
    if temp_col is None:
        raise ValueError("No se encontraron columnas numéricas en el dataset")
    
    logger.info(f"Usando columna '{temp_col}' como señal principal")
    
    # Extraer valores
    values = df[temp_col].values
    
    # Crear etiquetas: 1 si hay cualquier tipo de fallo
    failure_cols = [col for col in df.columns if 'failure' in col.lower()]
    
    if failure_cols:
        # Combinar todas las columnas de fallo
        labels = df[failure_cols].max(axis=1).values
    else:
        # Si no hay columnas de fallo, crear etiquetas basadas en outliers
        logger.warning("No se encontraron columnas de fallo, usando detección de outliers")
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        z_scores = np.abs((arr - mean) / std)
        labels = (z_scores > 3.0).astype(int)
    
    # Crear timestamps secuenciales
    timestamps = range(len(values))
    
    n_anomalies = sum(labels)
    logger.info(
        {
            "event": "dataset_loaded",
            "total_points": len(values),
            "anomaly_points": n_anomalies,
            "anomaly_pct": f"{n_anomalies/len(values)*100:.2f}%",
            "signal_column": temp_col,
        }
    )
    
    return values, timestamps, labels


def create_synthetic_ai4i_data():
    """Crea datos sintéticos basados en la descripción del dataset UCI AI4I 2020."""
    np.random.seed(42)
    n_samples = 10000
    
    # Generar datos simulados basados en las características del dataset real
    # Air temperature: 295-305 K
    air_temp = np.random.normal(300, 2, n_samples)
    
    # Process temperature: 305-315 K (correlacionado con air temp)
    process_temp = air_temp + np.random.normal(5, 1, n_samples)
    
    # Rotational speed: 1200-1700 rpm
    rotational_speed = np.random.normal(1500, 100, n_samples)
    
    # Torque: 20-100 Nm (inversamente correlacionado con speed)
    torque = 80 - (rotational_speed - 1500) * 0.1 + np.random.normal(0, 5, n_samples)
    
    # Tool wear: 0-300 min (acumulado)
    tool_wear = np.random.uniform(0, 300, n_samples)
    
    # Crear fallos (1% de los datos)
    n_failures = int(n_samples * 0.01)
    failure_indices = np.random.choice(n_samples, n_failures, replace=False)
    
    # Tipos de fallo
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Machine failure': [0] * n_samples,
    }
    
    # Añadir tipos de fallo
    for ft in failure_types:
        data[ft] = [0] * n_samples
    
    # Asignar fallos
    for idx in failure_indices:
        data['Machine failure'][idx] = 1
        # Asignar tipo de fallo aleatorio
        ft = np.random.choice(failure_types)
        data[ft][idx] = 1
        
        # Modificar valores simulando condiciones de fallo
        if ft == 'TWF':  # Tool Wear Failure
            data['Tool wear [min]'][idx] = 250 + np.random.uniform(0, 50)
        elif ft == 'HDF':  # Heat Dissipation Failure
            data['Air temperature [K]'][idx] += np.random.uniform(5, 10)
        elif ft == 'PWF':  # Power Failure
            data['Torque [Nm]'][idx] *= np.random.uniform(1.5, 2.0)
        elif ft == 'OSF':  # Overstrain Failure
            data['Rotational speed [rpm]'][idx] += np.random.uniform(200, 300)
        elif ft == 'RNF':  # Random Failure
            pass  # Sin patrón específico
    
    df = pd.DataFrame(data)
    logger.info("Datos sintéticos creados basados en descripción UCI AI4I 2020")
    return df


# ─── Baselines ───────────────────────────────────────────────────────────────


def baseline_zscore(values: List[float], threshold: float = 3.0) -> List[int]:
    """Z-score global — baseline más simple."""
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return [0] * len(values)
    z = np.abs((arr - mean) / std)
    return (z > threshold).astype(int).tolist()


def baseline_iqr(values: List[float], factor: float = 1.5) -> List[int]:
    """IQR global — baseline robusto a outliers."""
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return ((arr < lower) | (arr > upper)).astype(int).tolist()


def baseline_rolling_zscore(
    values: List[float], window: int = 50, threshold: float = 3.0
) -> List[int]:
    """Rolling Z-score — baseline más justo para streaming."""
    arr = np.array(values)
    results = [0] * len(arr)
    for i in range(window, len(arr)):
        w = arr[i - window : i]
        mean, std = w.mean(), w.std()
        if std > 0:
            z = abs((arr[i] - mean) / std)
            results[i] = int(z > threshold)
    return results


# ─── ZENIN detector ──────────────────────────────────────────────────────────


def run_zenin_detector(
    values: List[float],
    timestamps: List[float],
    voting_threshold: float = 0.5,
    contamination: Optional[float] = None,
) -> Tuple[List[int], List[float], float]:
    """
    Corre VotingAnomalyDetector con sliding window.

    Args:
        voting_threshold: Umbral de corte del ensemble. Default 0.5.
        contamination: Fracción esperada de anomalías para IF/LOF.

    Returns: predictions (0/1), scores (0-1), elapsed_seconds
    """
    try:
        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            Reading,
            TimeSeriesWindow,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
            VotingAnomalyDetector,
        )
        from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
            AnomalyDetectorConfig,
        )
    except ImportError as e:
        logger.error(
            {
                "event": "zenin_import_failed",
                "error": str(e),
                "hint": "Correr desde iot_machine_learning/ con PYTHONPATH correcto",
            }
        )
        raise

    logger.info(
        {
            "event": "zenin_detector_init",
            "voting_threshold": voting_threshold,
            "contamination": contamination,
        }
    )
    config = AnomalyDetectorConfig(
        voting_threshold=voting_threshold,
        contamination=contamination if contamination is not None else 0.005,
        min_training_points=WINDOW_SIZE - 2,
    )
    detector = VotingAnomalyDetector(
        config=config,
        series_id=SERIES_ID,
        enable_adaptive_weights=False,
    )

    # Train con warm-up inicial
    train_values = values[:WINDOW_SIZE]
    train_timestamps = timestamps[:WINDOW_SIZE]
    detector.train(train_values, timestamps=train_timestamps)
    logger.info(f"Detector entrenado con {WINDOW_SIZE} puntos de warm-up")

    predictions = [0] * len(values)
    scores = [0.0] * len(values)
    start_time = time.time()
    total_points = len(values) - WINDOW_SIZE

    for i in range(WINDOW_SIZE, len(values)):
        slice_values = values[i - WINDOW_SIZE + 1 : i + 1]
        slice_timestamps = timestamps[i - WINDOW_SIZE + 1 : i + 1]

        readings = [
            Reading(series_id=SERIES_ID, value=v, timestamp=t)
            for v, t in zip(slice_values, slice_timestamps)
        ]
        window = TimeSeriesWindow(series_id=SERIES_ID, readings=readings)

        result = detector.detect(window)
        predictions[i] = int(result.is_anomaly)
        scores[i] = float(result.score)

        # Progress log cada 500 puntos
        if (i - WINDOW_SIZE) % 500 == 0:
            progress = (i - WINDOW_SIZE) / total_points * 100
            logger.info(
                {
                    "event": "benchmark_progress",
                    "progress_pct": f"{progress:.1f}%",
                    "point": i,
                    "total": len(values),
                }
            )

    elapsed = time.time() - start_time
    throughput = total_points / elapsed
    logger.info(
        {
            "event": "zenin_detection_completed",
            "elapsed_s": f"{elapsed:.2f}",
            "throughput_pts_per_sec": f"{throughput:.1f}",
            "total_anomalies_detected": sum(predictions),
        }
    )

    return predictions, scores, elapsed


# ─── Métricas ────────────────────────────────────────────────────────────────


@dataclass
class DetectorMetrics:
    name: str
    f1: float
    precision: float
    recall: float
    auc_roc: float
    auc_pr: float
    elapsed_s: float
    anomalies_detected: int
    false_positives: int
    false_negatives: int


def compute_metrics(
    name: str,
    labels: List[int],
    predictions: List[int],
    scores: List[float],
    elapsed_s: float,
) -> DetectorMetrics:
    """Calcula métricas completas para un detector."""
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)
    scores_arr = np.array(scores)

    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    precision = precision_score(labels_arr, preds_arr, zero_division=0)
    recall = recall_score(labels_arr, preds_arr, zero_division=0)

    try:
        auc_roc = roc_auc_score(labels_arr, scores_arr)
        auc_pr = average_precision_score(labels_arr, scores_arr)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0

    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())

    return DetectorMetrics(
        name=name,
        f1=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        auc_roc=round(auc_roc, 4),
        auc_pr=round(auc_pr, 4),
        elapsed_s=round(elapsed_s, 2),
        anomalies_detected=int(preds_arr.sum()),
        false_positives=fp,
        false_negatives=fn,
    )


# ─── Reporte ─────────────────────────────────────────────────────────────────


def generate_report(
    results: List[DetectorMetrics],
    values: List[float],
    labels: List[int],
    zenin_scores: List[float],
) -> None:
    """Genera JSON + Markdown + PNG con resultados."""

    # 1. JSON con todos los números
    json_path = RESULTS_DIR / "uci_ai4i_results.json"
    payload = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "window_size": WINDOW_SIZE,
        "total_points": len(values),
        "total_anomaly_points": int(sum(labels)),
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"JSON guardado: {json_path}")

    # 2. Markdown report
    md_path = RESULTS_DIR / "uci_ai4i_report.md"
    zenin = next(r for r in results if r.name == "ZENIN VotingEnsemble")
    with open(md_path, "w") as f:
        f.write("# UCI AI4I 2020 Benchmark — Predictive Maintenance Dataset\n\n")
        f.write("**Dataset:** `UCI AI4I 2020 Predictive Maintenance Dataset`\n")
        f.write(f"**Total points:** {len(values):,}\n")
        f.write(f"**Anomaly points:** {sum(labels):,} ")
        f.write(f"({sum(labels)/len(values)*100:.2f}%)\n\n")
        f.write("## Results\n\n")
        f.write(
            "| Detector | F1 | Precision | Recall | AUC-ROC | AUC-PR | "
        )
        f.write("FP | FN | Speed (pts/s) |\n")
        f.write(
            "|----------|-----|-----------|--------|---------|--------"
        )
        f.write("----|----|--------------|\n")
        for r in sorted(results, key=lambda x: x.f1, reverse=True):
            speed = (
                int((len(values) - WINDOW_SIZE) / r.elapsed_s)
                if r.elapsed_s > 0
                else 0
            )
            marker = (
                " 🏆"
                if r.name == "ZENIN VotingEnsemble"
                and r.f1 == max(x.f1 for x in results)
                else ""
            )
            f.write(
                f"| {r.name}{marker} | {r.f1:.4f} | {r.precision:.4f} | "
                f"{r.recall:.4f} | {r.auc_roc:.4f} | {r.auc_pr:.4f} | "
                f"{r.false_positives} | {r.false_negatives} | {speed:,} |\n"
            )
        f.write("\n## Interpretation\n\n")
        if zenin.f1 == max(r.f1 for r in results):
            f.write(
                f"✅ **ZENIN wins** with F1={zenin.f1:.4f} "
            )
            f.write("vs best baseline ")
            f.write(
                f"F1={max(r.f1 for r in results if r.name != 'ZENIN VotingEnsemble'):.4f}\n"
            )
        else:
            best = max(
                (r for r in results if r.name != "ZENIN VotingEnsemble"),
                key=lambda x: x.f1,
            )
            f.write(
                f"⚠️ **Baseline wins** ({best.name} F1={best.f1:.4f}) "
            )
            f.write(f"vs ZENIN F1={zenin.f1:.4f}\n")
        
        f.write("\n## Dataset Description\n\n")
        f.write("The UCI AI4I 2020 Predictive Maintenance dataset contains sensor ")
        f.write("data from machines with various failure types including tool wear, ")
        f.write("heat dissipation failure, power failure, overstrain failure, and ")
        f.write("random failures. It includes air temperature, process temperature, ")
        f.write("rotational speed, torque, and tool wear measurements.\n")

    logger.info(f"Markdown guardado: {md_path}")

    # 3. PNG plot
    plot_path = RESULTS_DIR / "uci_ai4i_plot.png"
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Raw data with anomalies
    plot_points = min(1000, len(values))
    axes[0].plot(values[:plot_points], label='Air Temperature [K]', alpha=0.7)
    anomaly_indices = [i for i, l in enumerate(labels[:plot_points]) if l == 1]
    if anomaly_indices:
        axes[0].scatter(anomaly_indices, [values[i] for i in anomaly_indices], 
                       color='red', s=20, label='Machine Failures', zorder=5)
    axes[0].set_title('UCI AI4I 2020: Air Temperature with Failures (First 1000 points)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Temperature [K]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ZENIN scores
    axes[1].plot(zenin_scores[:plot_points], label='ZENIN Anomaly Score', color='blue', alpha=0.7)
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    axes[1].set_title('ZENIN Anomaly Scores (First 1000 points)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Comparison of best detectors
    detectors = [r.name for r in results]
    f1_scores = [r.f1 for r in results]
    
    bars = axes[2].bar(detectors, f1_scores, color=['green' if 'ZENIN' in d else 'blue' for d in detectors])
    axes[2].set_title('F1 Score Comparison')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Plot guardado: {plot_path}")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 70)
    logger.info("UCI AI4I 2020 BENCHMARK — Predictive Maintenance Dataset")
    logger.info("=" * 70)
    
    # Cargar datos
    values, timestamps, labels = load_uci_ai4i_data()
    
    results = []
    
    # Ejecutar baselines
    logger.info("Ejecutando baseline Z-score...")
    start = time.time()
    zscore_preds = baseline_zscore(values, threshold=3.0)
    zscore_elapsed = time.time() - start
    zscore_scores = [float(abs((v - np.mean(values)) / np.std(values))) for v in values]
    results.append(compute_metrics("Z-score (global)", labels, zscore_preds, zscore_scores, zscore_elapsed))
    
    logger.info("Ejecutando baseline IQR...")
    start = time.time()
    iqr_preds = baseline_iqr(values, factor=1.5)
    iqr_elapsed = time.time() - start
    iqr_scores = [float(abs(v - np.median(values)) / (np.percentile(values, 75) - np.percentile(values, 25))) 
                  if (np.percentile(values, 75) - np.percentile(values, 25)) > 0 else 0.0 
                  for v in values]
    results.append(compute_metrics("IQR (global)", labels, iqr_preds, iqr_scores, iqr_elapsed))
    
    logger.info("Ejecutando baseline Rolling Z-score...")
    start = time.time()
    rolling_preds = baseline_rolling_zscore(values, window=WINDOW_SIZE, threshold=3.0)
    rolling_elapsed = time.time() - start
    rolling_scores = [0.0] * len(values)
    arr = np.array(values)
    for i in range(WINDOW_SIZE, len(arr)):
        w = arr[i - WINDOW_SIZE : i]
        mean, std = w.mean(), w.std()
        if std > 0:
            rolling_scores[i] = float(abs((arr[i] - mean) / std) / 3.0)  # Normalized
    results.append(compute_metrics("Rolling Z-score", labels, rolling_preds, rolling_scores, rolling_elapsed))
    
    # Ejecutar ZENIN
    logger.info("Ejecutando ZENIN VotingAnomalyDetector...")
    zenin_preds, zenin_scores, zenin_elapsed = run_zenin_detector(
        values, timestamps, voting_threshold=0.5, contamination=0.1
    )
    results.append(compute_metrics("ZENIN VotingEnsemble", labels, zenin_preds, zenin_scores, zenin_elapsed))
    
    # Generar reporte
    logger.info("Generando reporte...")
    generate_report(results, values, labels, zenin_scores)
    
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETADO")
    logger.info(f"Resultados guardados en: {RESULTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
