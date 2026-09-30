"""Harness de Evaluación Estándar para NAB y Series Temporales Industriales.

Implementa:
1. Métricas oficiales de NAB (NAB Standard Score con perfiles Standard, Low-FP, Low-FN).
2. Métricas de Evento / Ventana (Event-Level Windowed Precision, Recall, F1).
3. Agrupamiento de falsas alarmas contiguas (False Alarm Clusters / Incidents).
4. Métricas de Rango (Range-Based Precision & Recall, Tatbul et al. NeurIPS 2018).
5. Métricas de Punto estándar (Point-wise F1, Precision, Recall, AUC-ROC, AUC-PR).

Permite auditar algoritmos de streaming sin los sesgos patológicos de evaluación puntual.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class AnomalyWindow:
    """Ventana de anomalía temporal."""

    window_id: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    center_time: Optional[float] = None

    def contains_idx(self, idx: int) -> bool:
        return self.start_idx <= idx <= self.end_idx

    def contains_time(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time

    @property
    def length(self) -> int:
        return max(1, self.end_idx - self.start_idx + 1)


@dataclass
class PointwiseMetrics:
    """Métricas clásicas punto a punto."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None


@dataclass
class EventLevelMetrics:
    """Métricas a nivel de evento/ventana."""

    total_events: int
    tp_events: int
    fn_events: int
    fp_points: int
    fp_clusters: int  # Incidentes de falsa alarma contiguos
    recall_event: float
    precision_event_pointwise: float  # TP_event / (TP_event + FP_points)
    precision_event_cluster: float  # TP_event / (TP_event + FP_clusters)
    f1_event_cluster: float  # F1 armónico con clusters de falsa alarma
    f1_event_pointwise: float  # F1 armónico con puntos individuales de falsa alarma
    detection_delays_points: List[int] = field(default_factory=list)
    detection_delays_relative: List[float] = field(default_factory=list)  # delay / window_len


@dataclass
class NABScoreResult:
    """Resultado del cálculo de la métrica oficial de NAB."""

    standard_score: float
    low_fp_score: float
    low_fn_score: float
    raw_score: float
    null_score: float
    perfect_score: float


@dataclass
class RangeBasedMetrics:
    """Métricas basadas en rango (Tatbul et al., NeurIPS 2018)."""

    range_recall: float
    range_precision: float
    range_f1: float


@dataclass
class ComprehensiveNABReport:
    """Informe consolidado de evaluación."""

    dataset_name: str
    total_points: int
    anomalous_points: int
    anomalous_events: int
    pointwise: PointwiseMetrics
    event_level: EventLevelMetrics
    nab_scoring: NABScoreResult
    range_based: RangeBasedMetrics

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        p = self.pointwise
        e = self.event_level
        n = self.nab_scoring
        r = self.range_based

        delays_str = (
            f"Mean: {np.mean(e.detection_delays_points):.1f} pts, P50: {np.median(e.detection_delays_points):.1f} pts"
            if e.detection_delays_points
            else "N/A"
        )

        return f"""### Resumen de Evaluación Multimétrica: {self.dataset_name}

| Paradigma de Evaluación | Métrica | Valor | Interpretación Operativa |
|---|---|---|---|
| **Evento / Ventana (NAB)** | **Event Recall** | **{e.recall_event*100:.2f}%** ({e.tp_events}/{e.total_events} eventos) | Cobertura de incidentes críticos |
| | **Event Precision (Cluster)** | **{e.precision_event_cluster*100:.2f}%** | 1 cluster continuo = 1 falsa alarma |
| | **Event F1 (Cluster)** | **{e.f1_event_cluster:.4f}** | Métrica industrial primaria |
| | **Retraso de Detección** | {delays_str} | Velocidad de respuesta ante fallas |
| **Oficial NAB Score** | **NAB Standard Profile** | **{n.standard_score:.2f} / 100** | Puntuación oficial de Numenta |
| | **NAB Low-FP Profile** | **{n.low_fp_score:.2f} / 100** | Ponderación estricta anti-falsos positivos |
| | **NAB Low-FN Profile** | **{n.low_fn_score:.2f} / 100** | Ponderación estricta anti-omisión |
| **Rango (Tatbul et al.)** | **Range Recall** | **{r.range_recall*100:.2f}%** | Solapamiento y existencia de rango |
| | **Range Precision** | **{r.range_precision*100:.2f}%** | Precisión ajustada por fragmentación |
| | **Range F1** | **{r.range_f1:.4f}** | Balance continuo de eventos |
| **Puntual Estricto (Local)**| **Pointwise Recall** | {p.recall*100:.2f}% ({p.tp}/{p.tp+p.fn} pts) | Cota inferior estricta punto a punto |
| | **Pointwise Precision** | {p.precision*100:.2f}% ({p.fp} FP) | Penaliza cualquier retraso de punto |
| | **Pointwise F1** | **{p.f1:.4f}** | Métrica clásica (altamente sesgada en IoT) |
| | **AUC-ROC / AUC-PR** | {p.auc_roc or 0.0:.4f} / {p.auc_pr or 0.0:.4f} | Capacidad de ranking de anomalías |
"""


class NABEvaluator:
    """Evaluador exhaustivo multimétrica para series temporales y NAB."""

    def __init__(
        self,
        window_size_points: int = 10,
        default_prob_threshold: float = 0.5,
    ) -> None:
        """Inicializa evaluador.

        Args:
            window_size_points: Radio N de tolerancia (ventana = [-N, +N] puntos, total 2N+1).
            default_prob_threshold: Umbral por defecto para binarizar scores continuos.
        """
        self.window_radius = window_size_points
        self.threshold = default_prob_threshold

    def build_windows_from_timestamps(
        self,
        series_timestamps: List[float],
        anomaly_timestamps: List[float],
    ) -> List[AnomalyWindow]:
        """Construye ventanas de detección centradas en las marcas de anomalía."""
        ts_arr = np.array(series_timestamps)
        windows: List[AnomalyWindow] = []

        for w_id, a_ts in enumerate(anomaly_timestamps):
            closest_idx = int(np.argmin(np.abs(ts_arr - a_ts)))
            start_idx = max(0, closest_idx - self.window_radius)
            end_idx = min(len(series_timestamps) - 1, closest_idx + self.window_radius)

            windows.append(
                AnomalyWindow(
                    window_id=w_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=series_timestamps[start_idx],
                    end_time=series_timestamps[end_idx],
                    center_time=a_ts,
                )
            )

        return windows

    def evaluate(
        self,
        predictions: List[int],
        scores: Optional[List[float]],
        series_timestamps: List[float],
        anomaly_timestamps: List[float],
        dataset_name: str = "NAB Dataset",
    ) -> ComprehensiveNABReport:
        """Calcula todas las métricas de forma integrada."""
        n_points = len(predictions)
        windows = self.build_windows_from_timestamps(series_timestamps, anomaly_timestamps)

        # 1. Crear máscara de verdad fundamental
        y_true = np.zeros(n_points, dtype=int)
        for w in windows:
            y_true[w.start_idx : w.end_idx + 1] = 1

        y_pred = np.array(predictions, dtype=int)

        # 2. Métricas Punto a Punto
        pointwise = self._compute_pointwise(y_true, y_pred, scores)

        # 3. Métricas de Evento / Ventana
        event_metrics = self._compute_event_metrics(y_pred, windows, n_points)

        # 4. Métrica Oficial de NAB
        nab_scoring = self._compute_official_nab(y_pred, windows)

        # 5. Métricas de Rango (Tatbul et al.)
        range_metrics = self._compute_range_metrics(y_true, y_pred)

        return ComprehensiveNABReport(
            dataset_name=dataset_name,
            total_points=n_points,
            anomalous_points=int(np.sum(y_true)),
            anomalous_events=len(windows),
            pointwise=pointwise,
            event_level=event_metrics,
            nab_scoring=nab_scoring,
            range_based=range_metrics,
        )

    def _compute_pointwise(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: Optional[List[float]],
    ) -> PointwiseMetrics:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        auc_roc: Optional[float] = None
        auc_pr: Optional[float] = None
        if scores is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = float(roc_auc_score(y_true, scores))
                auc_pr = float(average_precision_score(y_true, scores))
            except Exception:
                pass

        return PointwiseMetrics(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=prec,
            recall=rec,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
        )

    def _compute_event_metrics(
        self,
        y_pred: np.ndarray,
        windows: List[AnomalyWindow],
        n_points: int,
    ) -> EventLevelMetrics:
        total_events = len(windows)
        tp_events = 0
        delays_pts: List[int] = []
        delays_rel: List[float] = []

        # Verificar qué ventanas fueron detectadas
        detected_windows: set[int] = set()
        for w in windows:
            pred_slice = y_pred[w.start_idx : w.end_idx + 1]
            if np.any(pred_slice == 1):
                tp_events += 1
                detected_windows.add(w.window_id)
                # Primer índice de detección dentro de la ventana
                first_det_rel = int(np.argmax(pred_slice == 1))
                delays_pts.append(first_det_rel)
                delays_rel.append(first_det_rel / w.length)

        fn_events = total_events - tp_events
        recall_event = tp_events / max(1, total_events)

        # Calcular máscara de ventanas para aislar Falsos Positivos
        in_window_mask = np.zeros(n_points, dtype=bool)
        for w in windows:
            in_window_mask[w.start_idx : w.end_idx + 1] = True

        fp_indices = np.where((y_pred == 1) & (~in_window_mask))[0]
        fp_points = len(fp_indices)

        # Agrupar falsos positivos contiguos en clusters
        fp_clusters = 0
        if fp_points > 0:
            diffs = np.diff(fp_indices)
            # Un cluster nuevo comienza si la diferencia entre índices consecutivos es > 1
            fp_clusters = 1 + int(np.sum(diffs > 1))

        # Precisiones
        prec_pointwise = tp_events / max(1, tp_events + fp_points)
        prec_cluster = tp_events / max(1, tp_events + fp_clusters)

        # F1s
        f1_cluster = (
            2 * (prec_cluster * recall_event) / max(1e-12, prec_cluster + recall_event)
            if (prec_cluster + recall_event) > 0
            else 0.0
        )
        f1_pointwise = (
            2 * (prec_pointwise * recall_event) / max(1e-12, prec_pointwise + recall_event)
            if (prec_pointwise + recall_event) > 0
            else 0.0
        )

        return EventLevelMetrics(
            total_events=total_events,
            tp_events=tp_events,
            fn_events=fn_events,
            fp_points=fp_points,
            fp_clusters=fp_clusters,
            recall_event=recall_event,
            precision_event_pointwise=prec_pointwise,
            precision_event_cluster=prec_cluster,
            f1_event_cluster=f1_cluster,
            f1_event_pointwise=f1_pointwise,
            detection_delays_points=delays_pts,
            detection_delays_relative=delays_rel,
        )

    def _compute_official_nab(
        self,
        y_pred: np.ndarray,
        windows: List[AnomalyWindow],
    ) -> NABScoreResult:
        """Implementa la formulación de puntuación exacta de Numenta NAB."""
        # Perfiles de costo oficiales de NAB:
        # Standard: A_tp=1.0, A_fp=0.11, A_fn=1.0
        # Low FP:   A_tp=1.0, A_fp=0.22, A_fn=1.0
        # Low FN:   A_tp=1.0, A_fp=0.055, A_fn=2.0
        profiles = {
            "standard": {"a_tp": 1.0, "a_fp": 0.11, "a_fn": 1.0},
            "low_fp": {"a_tp": 1.0, "a_fp": 0.22, "a_fn": 1.0},
            "low_fn": {"a_tp": 1.0, "a_fp": 0.055, "a_fn": 2.0},
        }

        # Máscara de ventanas
        n_points = len(y_pred)
        in_window_mask = np.zeros(n_points, dtype=bool)
        for w in windows:
            in_window_mask[w.start_idx : w.end_idx + 1] = True

        fp_count = int(np.sum((y_pred == 1) & (~in_window_mask)))

        # Detecciones en ventana
        tp_detections_y: List[float] = []
        fn_window_count = 0

        for w in windows:
            pred_slice = y_pred[w.start_idx : w.end_idx + 1]
            if np.any(pred_slice == 1):
                first_rel = int(np.argmax(pred_slice == 1))
                # Posición y normalizada a [-1, 1] dentro de la ventana: y = 2 * (idx / length) - 1
                # En NAB: y = (first_idx - window_start) / window_len
                # La función sigmoide oficial premia detecciones en el primer tercio
                # En NAB: y escala en [-1.0, 1.0] dentro de la ventana
                # y = -1.0 al inicio de ventana (máxima recompensa: sigmoide ~ +1.0)
                # y = 0.0 en el centro de la ventana
                # y = +1.0 al final de la ventana (penalización por tardanza: sigmoide ~ -1.0)
                if w.length > 1:
                    y_pos = 2.0 * (first_rel / (w.length - 1)) - 1.0
                else:
                    y_pos = -1.0
                sig_val = (2.0 / (1.0 + math.exp(5.0 * y_pos))) - 1.0
                tp_detections_y.append(sig_val)
            else:
                fn_window_count += 1

        scores_out = {}
        # Perfect score alcanzado si se detecta al inicio de cada ventana (y = -1.0)
        perfect_sig = (2.0 / (1.0 + math.exp(-5.0))) - 1.0  # ~ 0.9866
        for pname, pparams in profiles.items():
            a_tp = pparams["a_tp"]
            a_fp = pparams["a_fp"]
            a_fn = pparams["a_fn"]

            # Score alcanzado
            raw_s = sum(a_tp * sig for sig in tp_detections_y) - (a_fp * fp_count) - (a_fn * fn_window_count)

            # Null score (no predice nada: todos los eventos son FN, 0 FP, 0 TP)
            null_s = -a_fn * len(windows)

            # Perfect score (detecta al inicio de cada ventana con perfect_sig, 0 FP, 0 FN)
            perfect_s = a_tp * len(windows) * perfect_sig

            # Normalización NAB a escala [0, 100]
            if perfect_s - null_s > 0:
                norm_s = 100.0 * (raw_s - null_s) / (perfect_s - null_s)
            else:
                norm_s = 0.0

            scores_out[pname] = max(-100.0, min(100.0, norm_s))

        return NABScoreResult(
            standard_score=scores_out["standard"],
            low_fp_score=scores_out["low_fp"],
            low_fn_score=scores_out["low_fn"],
            raw_score=raw_s,
            null_score=null_s,
            perfect_score=perfect_s,
        )

    def _compute_range_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RangeBasedMetrics:
        """Cálculo aproximado de métricas de rango según Tatbul et al. (NeurIPS 2018)."""
        # Identificar segmentos continuos de y_true
        true_diffs = np.diff(np.pad(y_true, (1, 1), "constant"))
        true_starts = np.where(true_diffs == 1)[0]
        true_ends = np.where(true_diffs == -1)[0]

        pred_diffs = np.diff(np.pad(y_pred, (1, 1), "constant"))
        pred_starts = np.where(pred_diffs == 1)[0]
        pred_ends = np.where(pred_diffs == -1)[0]

        if len(true_starts) == 0:
            return RangeBasedMetrics(range_recall=1.0, range_precision=0.0, range_f1=0.0)
        if len(pred_starts) == 0:
            return RangeBasedMetrics(range_recall=0.0, range_precision=1.0, range_f1=0.0)

        # Range Recall: fracción de cada rango real cubierto por predicciones
        recalls = []
        for ts, te in zip(true_starts, true_ends):
            range_len = te - ts
            overlap = np.sum(y_pred[ts:te] == 1)
            # Recompensa por existencia + cobertura proporcional
            exist_reward = 1.0 if overlap > 0 else 0.0
            overlap_fraction = overlap / max(1, range_len)
            recalls.append(0.5 * exist_reward + 0.5 * overlap_fraction)

        range_recall = float(np.mean(recalls))

        # Range Precision: fracción de cada rango predicho que solapa con verdad fundamental
        precisions = []
        for ps, pe in zip(pred_starts, pred_ends):
            range_len = pe - ps
            overlap = np.sum(y_true[ps:pe] == 1)
            precisions.append(overlap / max(1, range_len))

        range_precision = float(np.mean(precisions)) if precisions else 0.0

        if range_recall + range_precision > 0:
            range_f1 = 2 * (range_recall * range_precision) / (range_recall + range_precision)
        else:
            range_f1 = 0.0

        return RangeBasedMetrics(
            range_recall=range_recall,
            range_precision=range_precision,
            range_f1=range_f1,
        )
