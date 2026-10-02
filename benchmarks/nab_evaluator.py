"""Harness de Evaluación Estándar para NAB y Series Temporales Industriales.

Implementa:
1. Métricas oficiales de NAB con equivalencia canónica Numenta (numenta/NAB):
   - Ventanas oficiales completas (combined_windows.json)
   - Sigmoide escalada canónica y atenuación de FP
   - Período probatorio del 15% (probationary period)
   - Perfiles canónicos: Standard, Low-FP, Low-FN
   - Barrido de umbrales (threshold sweeper)
2. Métricas de Evento / Ventana (Event-Level Windowed Precision, Recall, F1).
3. Agrupamiento de falsas alarmas contiguas (False Alarm Clusters / Incidents).
4. Métricas de Rango (Range-Based Precision & Recall, Tatbul et al. NeurIPS 2018).
5. Métricas de Punto estándar (Point-wise F1, Precision, Recall, AUC-ROC, AUC-PR).

Permite auditar algoritmos de streaming con rigor científico y reproducibilidad total.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ─── Funciones Sigmoides Canónicas de Numenta NAB ─────────────────────────────


def canonical_sigmoid(x: float) -> float:
    """Función sigmoide logística estándar."""
    return 1.0 / (1.0 + math.exp(-x))


def canonical_scaled_sigmoid(relative_position: float) -> float:
    """Función sigmoide escalada canónica de Numenta NAB (numenta/NAB/nab/sweeper.py).

    Calcula la recompensa / penalización temporal:
    - y = -1.0 (inicio exacto de ventana): recompensa máxima TP = 2*sig(5) - 1 ~ 0.98661
    - y = -0.5 (mitad de ventana): recompensa retrasada TP = 2*sig(2.5) - 1 ~ 0.84828
    - y = 0.0 (fin exacto de ventana): recompensa neutra = 0.0
    - y = 1.0 (después de ventana): penalización FP = 2*sig(-5) - 1 ~ -0.98661
    - y > 3.0 (lejos de ventana): penalización máxima FP = -1.0

    Args:
        relative_position: Posición relativa al evento o ventana.

    Returns:
        Valor escalar en [-1.0, 0.986614].
    """
    if relative_position > 3.0:
        return -1.0
    return 2.0 * canonical_sigmoid(-5.0 * relative_position) - 1.0


# ─── Dataclasses de Métricas ──────────────────────────────────────────────────


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
    detection_delays_relative: List[float] = field(default_factory=list)


@dataclass
class NABScoreResult:
    """Resultado del cálculo de la métrica oficial de NAB."""

    standard_score: float
    low_fp_score: float
    low_fn_score: float
    raw_score: float
    null_score: float
    perfect_score: float
    optimal_threshold: Optional[float] = None
    optimal_standard_score: Optional[float] = None


@dataclass
class RangeBasedMetrics:
    """Métricas basadas en rango (Tatbul et al., NeurIPS 2018)."""

    range_recall: float
    range_precision: float
    range_f1: float


@dataclass
class CanonicalThresholdScore:
    """Puntuación canónica de NAB para un umbral específico."""

    threshold: float
    score: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int


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
| | **Event F1 (Cluster)** | **{e.f1_event_cluster:.4f}** | Desempeño agrupando detecciones contiguas por incidente |
| | **Retraso de Detección** | {delays_str} | Velocidad de respuesta ante fallas |
| **Oficial NAB Score** | **NAB Standard Profile** | **{n.standard_score:.2f} / 100** | Puntuación canónica de Numenta |
| | **NAB Low-FP Profile** | **{n.low_fp_score:.2f} / 100** | Ponderación estricta anti-falsos positivos |
| | **NAB Low-FN Profile** | **{n.low_fn_score:.2f} / 100** | Ponderación estricta anti-omisión |
| | **NAB Optimal (Threshold Sweeper)** | **{n.optimal_standard_score or n.standard_score:.2f} / 100** | Máximo teórico discriminativo |
| **Rango (Tatbul et al.)** | **Range Recall** | **{r.range_recall*100:.2f}%** | Solapamiento y existencia de rango |
| | **Range Precision** | **{r.range_precision*100:.2f}%** | Precisión ajustada por fragmentación |
| | **Range F1** | **{r.range_f1:.4f}** | Balance continuo de eventos |
| **Puntual Estricto (Local)**| **Pointwise Recall** | {p.recall*100:.2f}% ({p.tp}/{p.tp+p.fn} pts) | Cota inferior estricta punto a punto |
| | **Pointwise Precision** | {p.precision*100:.2f}% ({p.fp} FP) | Penaliza cualquier retraso de punto |
| | **Pointwise F1** | **{p.f1:.4f}** | Métrica clásica (altamente sesgada en IoT) |
| | **AUC-ROC / AUC-PR** | {p.auc_roc or 0.0:.4f} / {p.auc_pr or 0.0:.4f} | Capacidad de ranking de anomalías |
"""


# ─── Sweeper Canónico de Numenta NAB ──────────────────────────────────────────


@dataclass
class _AnomalyPoint:
    timestamp: Any
    anomaly_score: float
    sweep_score: float
    window_name: Optional[str]


def score_dataset_canonical(
    timestamps: List[Any],
    anomaly_scores: List[float],
    window_limits: List[Tuple[Any, Any]],
    dataset_name: str = "dataset",
    threshold: Optional[float] = 0.5,
    profile_name: str = "standard",
    cost_matrix: Optional[Dict[str, float]] = None,
    probation_percent: float = 0.15,
) -> CanonicalThresholdScore:
    """Implementa exactamente la función scoreDataSet de Numenta NAB.

    Args:
        timestamps: Lista de timestamps (float unix o pandas Timestamp).
        anomaly_scores: Puntuaciones continuas o binarias en [0.0, 1.0].
        window_limits: Lista de tuplas (start_time, end_time) para cada ventana.
        dataset_name: Nombre único del dataset para identificar ventanas.
        threshold: Umbral para binarizar detecciones. Si es None, busca el umbral óptimo.
        profile_name: Perfil de ponderación ('standard', 'low_fp', 'low_fn').
        cost_matrix: Matriz opcional con tpWeight, fpWeight, fnWeight.
        probation_percent: Porcentaje inicial descartado como período probatorio (default 0.15).

    Returns:
        CanonicalThresholdScore con score bruto, tp, fp, fn, tn y total.
    """
    profiles = {
        "standard": {"tpWeight": 1.0, "fpWeight": 0.11, "fnWeight": 1.0},
        "low_fp": {"tpWeight": 1.0, "fpWeight": 0.22, "fnWeight": 1.0},
        "low_fn": {"tpWeight": 1.0, "fpWeight": 0.055, "fnWeight": 2.0},
    }
    if cost_matrix is not None:
        cm = cost_matrix
    else:
        cm = profiles.get(profile_name, profiles["standard"])

    tp_weight = cm["tpWeight"]
    fp_weight = cm["fpWeight"]
    fn_weight = cm["fnWeight"]

    max_tp = canonical_scaled_sigmoid(-1.0)
    num_rows = len(timestamps)
    probationary_length = min(
        math.floor(probation_percent * num_rows),
        int(probation_percent * 5000),
    )

    remaining_windows = list(window_limits)
    anomaly_list: List[_AnomalyPoint] = []

    cur_window_limits: Optional[Tuple[Any, Any]] = None
    cur_window_name: Optional[str] = None
    cur_window_width: Optional[float] = None
    cur_window_right_idx: Optional[int] = None
    prev_window_width: Optional[float] = None
    prev_window_right_idx: Optional[int] = None

    ts_list = list(timestamps)

    for i, (cur_time, cur_score) in enumerate(zip(ts_list, anomaly_scores, strict=False)):
        # Entrada en ventana
        if remaining_windows and cur_time == remaining_windows[0][0]:
            cur_window_limits = remaining_windows.pop(0)
            cur_window_name = f"{dataset_name}|{cur_window_limits[0]}"
            cur_window_right_idx = ts_list.index(cur_window_limits[1])
            left_idx = ts_list.index(cur_window_limits[0])
            cur_window_width = float(cur_window_right_idx - left_idx + 1)

        # Cálculo de weightedScore según posición
        if cur_window_limits is not None and cur_window_right_idx is not None and cur_window_width is not None:
            position_in_window = -(cur_window_right_idx - i + 1) / cur_window_width
            unweighted_score = canonical_scaled_sigmoid(position_in_window)
            weighted_score = unweighted_score * tp_weight / max_tp
        else:
            if prev_window_right_idx is None or prev_window_width is None:
                unweighted_score = -1.0
            else:
                numerator = abs(prev_window_right_idx - i)
                denominator = float(prev_window_width - 1)
                position_past_window = numerator / max(1.0, denominator)
                unweighted_score = canonical_scaled_sigmoid(position_past_window)
            weighted_score = unweighted_score * fp_weight

        # Marcado de probation
        point_window_name = cur_window_name if i >= probationary_length else "probationary"
        anomaly_list.append(_AnomalyPoint(cur_time, float(cur_score), float(weighted_score), point_window_name))

        # Salida de ventana
        if cur_window_limits is not None and cur_time == cur_window_limits[1]:
            prev_window_right_idx = i
            prev_window_width = cur_window_width
            cur_window_limits = None
            cur_window_name = None
            cur_window_width = None
            cur_window_right_idx = None

    # Filtrar probation y ordenar descendentemente por score
    scorable_list = [p for p in anomaly_list if p.window_name != "probationary"]
    scorable_list.sort(key=lambda x: x.anomaly_score, reverse=True)

    # Inicializar diccionarios de partes
    score_parts: Dict[str, float] = {"fp": 0.0}
    for row in scorable_list:
        if row.window_name is not None and row.window_name != "probationary":
            score_parts[row.window_name] = -fn_weight

    scores_by_threshold: List[CanonicalThresholdScore] = []
    cur_thresh = 1.1

    tn = sum(1 for x in scorable_list if x.window_name is None)
    fn = sum(1 for x in scorable_list if x.window_name is not None)
    tp = 0
    fp = 0

    for data_point in scorable_list:
        if data_point.anomaly_score != cur_thresh:
            cur_s = sum(score_parts.values())
            scores_by_threshold.append(
                CanonicalThresholdScore(cur_thresh, cur_s, tp, tn, fp, fn, tp + tn + fp + fn)
            )
            cur_thresh = data_point.anomaly_score

        if data_point.window_name is not None:
            tp += 1
            fn -= 1
            score_parts[data_point.window_name] = max(
                score_parts[data_point.window_name],
                data_point.sweep_score,
            )
        else:
            fp += 1
            tn -= 1
            score_parts["fp"] += data_point.sweep_score

    # Guardar último corte
    cur_s = sum(score_parts.values())
    scores_by_threshold.append(
        CanonicalThresholdScore(cur_thresh, cur_s, tp, tn, fp, fn, tp + tn + fp + fn)
    )

    if threshold is None:
        # Búsqueda del mejor umbral
        best_row = max(scores_by_threshold, key=lambda s: s.score)
        return best_row

    # Búsqueda del corte más cercano al umbral indicado
    matching_row = scores_by_threshold[-1]
    prev_row = scores_by_threshold[0]
    for ts_score in scores_by_threshold:
        if ts_score.threshold == threshold:
            matching_row = ts_score
            break
        elif ts_score.threshold < threshold:
            matching_row = prev_row
            break
        prev_row = ts_score

    return matching_row


# ─── Evaluador Multimétrica Integral ──────────────────────────────────────────


class NABEvaluator:
    """Evaluador exhaustivo multimétrica con soporte canónico oficial de Numenta NAB."""

    def __init__(
        self,
        window_size_points: int = 10,
        default_prob_threshold: float = 0.5,
        probation_percent: float = 0.15,
    ) -> None:
        """Inicializa evaluador.

        Args:
            window_size_points: Radio N de tolerancia (usado como fallback si no hay combined_windows).
            default_prob_threshold: Umbral por defecto para binarizar scores continuos.
            probation_percent: Proporción de la serie descartada como período probatorio (default 0.15).
        """
        self.window_radius = window_size_points
        self.threshold = default_prob_threshold
        self.probation_percent = probation_percent

    def build_windows_from_ranges(
        self,
        series_timestamps: List[Any],
        window_ranges: List[Tuple[Any, Any]],
    ) -> List[AnomalyWindow]:
        """Construye ventanas exactas a partir de intervalos [start, end] (combined_windows.json)."""
        ts_series = pd.Series([pd.to_datetime(t) for t in series_timestamps])
        windows: List[AnomalyWindow] = []

        for w_id, (s_time, e_time) in enumerate(window_ranges):
            s_dt = pd.to_datetime(s_time)
            e_dt = pd.to_datetime(e_time)

            s_idx = int((ts_series - s_dt).abs().idxmin())
            e_idx = int((ts_series - e_dt).abs().idxmin())
            if s_idx > e_idx:
                s_idx, e_idx = e_idx, s_idx

            c_idx = (s_idx + e_idx) // 2

            windows.append(
                AnomalyWindow(
                    window_id=w_id,
                    start_idx=s_idx,
                    end_idx=e_idx,
                    start_time=float(pd.to_datetime(series_timestamps[s_idx]).timestamp()),
                    end_time=float(pd.to_datetime(series_timestamps[e_idx]).timestamp()),
                    center_time=float(pd.to_datetime(series_timestamps[c_idx]).timestamp()),
                )
            )

        return windows

    def build_windows_from_timestamps(
        self,
        series_timestamps: List[float],
        anomaly_timestamps: List[float],
    ) -> List[AnomalyWindow]:
        """Construye ventanas de tolerancia simétricas centradas en timestamps puntuales."""
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
        series_timestamps: List[Any],
        anomaly_timestamps: Optional[List[float]] = None,
        window_ranges: Optional[List[Tuple[Any, Any]]] = None,
        dataset_name: str = "NAB Dataset",
    ) -> ComprehensiveNABReport:
        """Calcula todas las métricas de forma integrada."""
        n_points = len(predictions)

        # Construir ventanas oficiales o sintéticas
        if window_ranges is not None:
            windows = self.build_windows_from_ranges(series_timestamps, window_ranges) if len(window_ranges) > 0 else []
            window_limits = window_ranges
        elif anomaly_timestamps is not None:
            # Detectar si series_timestamps ya son números flotantes o timestamps de fecha
            first_ts = series_timestamps[0] if len(series_timestamps) > 0 else 0.0
            if isinstance(first_ts, (int, float, np.number)):
                ts_floats = [float(t) for t in series_timestamps]
                anom_floats = [float(t) for t in anomaly_timestamps]
            else:
                ts_floats = [float(pd.to_datetime(t).timestamp()) for t in series_timestamps]
                anom_floats = [
                    float(t) if isinstance(t, (int, float, np.number)) else float(pd.to_datetime(t).timestamp())
                    for t in anomaly_timestamps
                ]

            windows = self.build_windows_from_timestamps(ts_floats, anom_floats)
            window_limits = [
                (series_timestamps[w.start_idx], series_timestamps[w.end_idx])
                for w in windows
            ]
        else:
            raise ValueError("Se debe proveer window_ranges o anomaly_timestamps")

        # 1. Máscara de verdad fundamental
        y_true = np.zeros(n_points, dtype=int)
        for w in windows:
            y_true[w.start_idx : w.end_idx + 1] = 1

        y_pred = np.array(predictions, dtype=int)

        # 2. Métricas Punto a Punto
        pointwise = self._compute_pointwise(y_true, y_pred, scores)

        # 3. Métricas de Evento / Ventana
        event_metrics = self._compute_event_metrics(y_pred, windows, n_points)

        # 4. Métrica Oficial de NAB Canónica (Sweeper)
        nab_scoring = self._compute_canonical_nab_scoring(
            series_timestamps=series_timestamps,
            predictions=predictions,
            scores=scores,
            window_limits=window_limits,
            windows=windows,
            dataset_name=dataset_name,
        )

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

        detected_windows: set[int] = set()
        for w in windows:
            pred_slice = y_pred[w.start_idx : w.end_idx + 1]
            if np.any(pred_slice == 1):
                tp_events += 1
                detected_windows.add(w.window_id)
                first_det_rel = int(np.argmax(pred_slice == 1))
                delays_pts.append(first_det_rel)
                delays_rel.append(first_det_rel / w.length)

        fn_events = total_events - tp_events
        recall_event = tp_events / max(1, total_events)

        in_window_mask = np.zeros(n_points, dtype=bool)
        for w in windows:
            in_window_mask[w.start_idx : w.end_idx + 1] = True

        fp_indices = np.where((y_pred == 1) & (~in_window_mask))[0]
        fp_points = len(fp_indices)

        fp_clusters = 0
        if fp_points > 0:
            diffs = np.diff(fp_indices)
            fp_clusters = 1 + int(np.sum(diffs > 1))

        prec_pointwise = tp_events / max(1, tp_events + fp_points)
        prec_cluster = tp_events / max(1, tp_events + fp_clusters)

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

    def _compute_canonical_nab_scoring(
        self,
        series_timestamps: List[Any],
        predictions: List[int],
        scores: Optional[List[float]],
        window_limits: List[Tuple[Any, Any]],
        windows: List[AnomalyWindow],
        dataset_name: str,
    ) -> NABScoreResult:
        """Ejecuta el cálculo oficial canónico de NAB para Standard, Low-FP y Low-FN."""
        input_scores = [float(s) for s in scores] if scores is not None else [float(p) for p in predictions]
        thresh_to_eval = self.threshold if scores is not None else 0.5

        profiles_norm: Dict[str, float] = {}
        raw_scores: Dict[str, float] = {}

        # Pesos oficiales:
        profile_params = {
            "standard": {"tpWeight": 1.0, "fpWeight": 0.11, "fnWeight": 1.0},
            "low_fp": {"tpWeight": 1.0, "fpWeight": 0.22, "fnWeight": 1.0},
            "low_fn": {"tpWeight": 1.0, "fpWeight": 0.055, "fnWeight": 2.0},
        }

        num_windows = len(windows)

        for pname, pparams in profile_params.items():
            cost_matrix = pparams
            tp_w = cost_matrix["tpWeight"]
            fn_w = cost_matrix["fnWeight"]

            row = score_dataset_canonical(
                timestamps=series_timestamps,
                anomaly_scores=input_scores,
                window_limits=window_limits,
                dataset_name=dataset_name,
                threshold=thresh_to_eval,
                cost_matrix=cost_matrix,
                probation_percent=self.probation_percent,
            )
            raw_s = row.score
            raw_scores[pname] = raw_s

            null_s = -fn_w * num_windows
            perfect_s = tp_w * num_windows
            denom = perfect_s - null_s

            if denom > 0:
                norm_s = 100.0 * (raw_s - null_s) / denom
            else:
                norm_s = 0.0

            profiles_norm[pname] = max(-100.0, min(100.0, norm_s))

        # Barrido óptimo si se proveen scores continuos
        opt_standard_score: Optional[float] = None
        opt_thresh: Optional[float] = None
        if scores is not None:
            opt_row = score_dataset_canonical(
                timestamps=series_timestamps,
                anomaly_scores=input_scores,
                window_limits=window_limits,
                dataset_name=dataset_name,
                threshold=None,  # Sweep
                cost_matrix=profile_params["standard"],
                probation_percent=self.probation_percent,
            )
            null_std = -1.0 * num_windows
            perf_std = 1.0 * num_windows
            opt_denom = perf_std - null_std
            if opt_denom > 0:
                opt_standard_score = 100.0 * (opt_row.score - null_std) / opt_denom
            else:
                opt_standard_score = 100.0 if opt_row.score == 0 else 0.0
            opt_thresh = opt_row.threshold

        return NABScoreResult(
            standard_score=profiles_norm["standard"],
            low_fp_score=profiles_norm["low_fp"],
            low_fn_score=profiles_norm["low_fn"],
            raw_score=raw_scores["standard"],
            null_score=-1.0 * num_windows,
            perfect_score=1.0 * num_windows,
            optimal_threshold=opt_thresh,
            optimal_standard_score=opt_standard_score,
        )

    def _compute_range_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RangeBasedMetrics:
        """Cálculo de métricas de rango según Tatbul et al. (NeurIPS 2018)."""
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

        recalls = []
        for ts, te in zip(true_starts, true_ends, strict=False):
            range_len = te - ts
            overlap = np.sum(y_pred[ts:te] == 1)
            exist_reward = 1.0 if overlap > 0 else 0.0
            overlap_fraction = overlap / max(1, range_len)
            recalls.append(0.5 * exist_reward + 0.5 * overlap_fraction)

        range_recall = float(np.mean(recalls))

        precisions = []
        for ps, pe in zip(pred_starts, pred_ends, strict=False):
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
