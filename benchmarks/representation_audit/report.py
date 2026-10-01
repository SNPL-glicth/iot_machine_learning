"""Generador de reportes estructurados para la auditoría de representaciones de ZENIN.

Produce:
- Tablas comparativas formateadas en Markdown / ASCII
- Matriz de pérdida de información L_R
- Desglose por evento canónico y retardo temporal
- Exportación a JSON estructurado en benchmarks/results/representation_audit.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .event_analysis import EventAuditSummary, PreCompressionGuardResult
from .metrics import DetectorSeparability, StatisticalSeparability
from .resource_measurement import ResourceBenchmarkResult


def format_markdown_report(
    resource_results: list[ResourceBenchmarkResult],
    stat_results: dict[str, StatisticalSeparability],
    det_results: dict[str, DetectorSeparability],
    event_summaries: dict[str, EventAuditSummary],
    guard_results: list[PreCompressionGuardResult],
) -> str:
    """Genera el reporte integral en formato Markdown."""
    lines: list[str] = []

    lines.append("# ZENIN: AUDITORÍA EXPERIMENTAL DE PÉRDIDA DE REPRESENTACIÓN")
    lines.append("Dataset: NAB machine_temperature_system_failure.csv (22,695 puntos, 4 fallas canónicas)\n")

    # 1. Recursos y compresión
    lines.append("## 1. Costo Computacional y Factor de Compresión")
    lines.append(
        "| Representación | Puntos Salida | Compresión | Tiempo (ms) | Memoria (KB) | Throughput (pts/s) | Latencia (μs/pt) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in resource_results:
        lines.append(
            f"| `{r.name}` | {r.n_output_points:,} | {r.compression_ratio:.1f}x | {r.elapsed_ms:.2f} | {r.peak_memory_kb:.1f} | {r.throughput_pts_per_sec:,.0f} | {r.latency_per_point_us:.3f} |"
        )
    lines.append("")

    # 2. Separabilidad Estadística y del Detector
    lines.append("## 2. Separabilidad Estadística y del Detector (S_stat & S_detector)")
    lines.append(
        "| Representación | Cohen's d | Wasserstein | SNR (dB) | KS Stat | ROC-AUC | PR-AUC | Margen Sep (P95) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in resource_results:
        st = stat_results.get(r.name)
        dt = det_results.get(r.name)
        if st and dt:
            lines.append(
                f"| `{r.name}` | {st.cohens_d:.3f} | {st.wasserstein_dist:.2f} | {st.snr_db:.1f} dB | {st.ks_statistic:.3f} | {dt.roc_auc:.4f} | {dt.pr_auc:.4f} | {dt.separation_margin_p95:+.3f} |"
            )
    lines.append("")

    # 3. Preservación por Evento Canónico y Desplazamiento Temporal
    lines.append("## 3. Preservación de Eventos Canónicos y Desplazamiento Temporal (S_event & S_temporal)")
    lines.append(
        "| Representación | Ev 1 (Variación) | Ev 2 (Shock 2°) | Ev 3 (Régimen 64°) | Ev 4 (Colapso 25°) | FP Puntos | FP Clusters |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in resource_results:
        ev = event_summaries.get(r.name)
        if not ev:
            continue

        def _cell(e_id: int) -> str:
            assert ev is not None
            p = ev.event_preservation.get(e_id)
            t = ev.temporal_preservation.get(e_id)
            if not p or not p.detected:
                return "❌ PERDIDO"
            delay_str = f"+{t.delay_points_equiv} pts" if t and t.delay_points_equiv is not None else "OK"
            return f"✅ ({delay_str})"

        lines.append(
            f"| `{r.name}` | {_cell(1)} | {_cell(2)} | {_cell(3)} | {_cell(4)} | {ev.fp_points_total} | {ev.fp_clusters_total} |"
        )
    lines.append("")

    # 4. Matriz de Retención de Evidencia y Pérdida L_R
    lines.append("## 4. Matriz de Retención de Evidencia de Pico (L_R = 1 - Peak_trans / Peak_raw)")
    lines.append(
        "| Representación | Ev 1 Retención | Ev 2 Retención | Ev 3 Retención | Ev 4 Retención | Estado Global |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    for r in resource_results:
        ev = event_summaries.get(r.name)
        if not ev:
            continue

        def _ret(e_id: int) -> str:
            assert ev is not None
            p = ev.event_preservation.get(e_id)
            if not p or not p.detected:
                return "0.0% (L_R=1.0)"
            pct = p.peak_retention_ratio * 100.0
            lr = 1.0 - p.peak_retention_ratio
            return f"{pct:.1f}% (L_R={lr:+.2f})"

        # Clasificación de seguridad
        all_detected = all(p.detected for p in ev.event_preservation.values())
        status = "🟢 SEGURA" if all_detected else "🔴 DESTRUCTIVA"

        lines.append(
            f"| `{r.name}` | {_ret(1)} | {_ret(2)} | {_ret(3)} | {_ret(4)} | {status} |"
        )
    lines.append("")

    # 5. Evaluación del Guardián Barato (Pregunta 6)
    lines.append("## 5. Pregunta 6: El Guardián Barato de No-Compresión (Pre-Compression Guard)")
    lines.append(
        "| Disparador O(1) | Eventos Protegidos | Falsos Frenados Normal (%) | Ahorro de Datos Efectivo (%) | Viabilidad |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: |"
    )
    for g in guard_results:
        prot_str = f"{g.true_event_protection_rate:.0f}% ({sum(1 for v in g.events_protected.values() if v)}/4)"
        viab = "🟢 EXCELENTE" if g.true_event_protection_rate == 100.0 and g.data_savings_achieved_pct > 80.0 else "🟡 PARCIAL"
        lines.append(
            f"| `{g.trigger_name}` | {prot_str} | {g.false_guard_rate_normal:.1f}% | {g.data_savings_achieved_pct:.1f}% | {viab} |"
        )
    lines.append("")

    return "\n".join(lines)


def save_audit_json(
    out_path: Path,
    resource_results: list[ResourceBenchmarkResult],
    stat_results: dict[str, StatisticalSeparability],
    det_results: dict[str, DetectorSeparability],
    event_summaries: dict[str, EventAuditSummary],
    guard_results: list[PreCompressionGuardResult],
) -> None:
    """Guarda el resultado completo en formato JSON para consumo downstream."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "resources": [asdict(r) for r in resource_results],
        "statistical_separability": {k: asdict(v) for k, v in stat_results.items()},
        "detector_separability": {k: asdict(v) for k, v in det_results.items()},
        "event_summaries": {
            k: {
                "name": v.name,
                "fp_points": v.fp_points_total,
                "fp_clusters": v.fp_clusters_total,
                "max_fp_score": v.max_fp_score,
                "event_preservation": {e_id: asdict(ep) for e_id, ep in v.event_preservation.items()},
                "temporal_preservation": {e_id: asdict(tp) for e_id, tp in v.temporal_preservation.items()},
            }
            for k, v in event_summaries.items()
        },
        "guards": [asdict(g) for g in guard_results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
