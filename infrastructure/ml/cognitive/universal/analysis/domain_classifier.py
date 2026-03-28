"""Domain classification from content analysis."""

from __future__ import annotations

from typing import Any, Dict, List

from .types import InputType
from iot_machine_learning.infrastructure.ml.inference.bayesian.naive_bayes import NaiveBayesClassifier


# NaiveBayes classifier for intelligent domain classification
_classifier = NaiveBayesClassifier(classes=["infrastructure", "security", "operations", "business", "trading", "general"])

def _extract_features(raw_data: Any, input_type: InputType) -> Dict[str, float]:
    """Extract features for NaiveBayes classification."""
    features = {}
    
    if input_type == InputType.TEXT and isinstance(raw_data, str):
        text = raw_data.lower()
        # Keyword density features
        for domain, keywords in DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            features[f"{domain}_keyword_density"] = count / max(len(text.split()), 1)
        
        # Text features
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["has_numbers"] = 1.0 if any(c.isdigit() for c in text) else 0.0
        features["has_exclamation"] = 1.0 if "!" in text else 0.0
        features["has_question"] = 1.0 if "?" in text else 0.0
    
    elif input_type == InputType.NUMERIC and isinstance(raw_data, list):
        if raw_data:
            import statistics
            features["numeric_length"] = len(raw_data)
            features["numeric_mean"] = statistics.mean(raw_data)
            features["numeric_variance"] = statistics.variance(raw_data) if len(raw_data) > 1 else 0.0
            features["numeric_cv"] = (statistics.stdev(raw_data) / abs(statistics.mean(raw_data))) if len(raw_data) > 1 and statistics.mean(raw_data) != 0 else 0.0
    
    return features

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "infrastructure": [
        "server", "cpu", "memory", "disk", "network", "node",
        "cluster", "deploy", "container", "kubernetes", "latency",
        "bandwidth", "throughput", "load", "capacity", "storage",
        # Spanish keywords
        "servidor", "memoria", "disco", "red", "nodo", "clúster",
        "infraestructura", "temperatura", "rack", "sistema", "base de datos",
        "crítico", "temperatura", "falla", "caída", "colapso",
    ],
    "security": [
        "vulnerability", "breach", "unauthorized", "firewall",
        "intrusion", "malware", "exploit", "authentication",
        "encryption", "certificate", "attack", "threat", "access",
    ],
    "operations": [
        "incident", "outage", "downtime", "maintenance",
        "escalation", "sla", "recovery", "alert", "ticket",
        "oncall", "runbook", "postmortem", "deploy",
        # Maintenance and repair keywords
        "preventive", "corrective", "repair", "service",
        "work order", "inspection", "overhaul", "replacement",
        "component", "compressor", "valve", "motor", "pump",
        "bearing", "seal", "filter", "lubrication",
        # Spanish keywords
        "mantenimiento", "preventivo", "correctivo", "reparación",
        "servicio", "inspección", "sobrecarga", "reemplazo",
        "componente", "compresor", "válvula", "motor", "bomba",
        "rodamiento", "sellado", "filtro", "lubricación",
    ],
    "business": [
        "revenue", "cost", "budget", "forecast", "margin",
        "growth", "kpi", "target", "profit", "contract",
        "customer", "sales", "market", "invoice",
    ],
    "trading": [
        "price", "volume", "volatility", "bid", "ask", "spread",
        "order", "execution", "liquidity", "risk", "position",
        "hedge", "derivative", "futures", "options", "market",
    ],
}


def classify_domain(
    raw_data: Any,
    input_type: InputType,
    metadata: Dict[str, Any],
    hint: str = "",
) -> str:
    """Classify document/data domain using NaiveBayes with keyword fallback.

    Args:
        raw_data: Original input
        input_type: Detected InputType
        metadata: From input_detector
        hint: Optional domain override

    Returns:
        Domain string (infrastructure, security, trading, operations, business, general)
    """
    if hint and hint in DOMAIN_KEYWORDS:
        return hint
    
    # Try NaiveBayes first
    features = _extract_features(raw_data, input_type)
    if features:  # Only use NaiveBayes if we have features
        proba = _classifier.predict_proba(features)
        if proba.confidence > 0.4:  # confident enough
            return proba.winner
    
    # Fallback to keyword matching
    if input_type == InputType.TEXT:
        return _classify_text_domain(str(raw_data))
    
    if input_type == InputType.NUMERIC:
        return _classify_numeric_domain(raw_data, metadata)
    
    if input_type == InputType.TABULAR:
        return _classify_tabular_domain(raw_data, metadata)
    
    if input_type == InputType.MIXED:
        if isinstance(raw_data, str):
            return _classify_text_domain(raw_data)
    
    return "general"


def _classify_text_domain(text: str) -> str:
    """Keyword-based domain classification for text."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    
    for domain_name, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[domain_name] = count
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _classify_numeric_domain(values: list, metadata: Dict[str, Any]) -> str:
    """Pattern-based domain classification for numeric data.
    
    Heuristics:
        - High variance (CV > 0.5) → trading
        - Low variance (CV < 0.1) → operations (stable metrics)
        - Otherwise → general
    """
    if not values or len(values) < 3:
        return "general"
    
    try:
        mean = sum(values) / len(values)
        if abs(mean) < 1e-9:
            return "general"
        
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        cv = std / abs(mean)
        
        if cv > 0.5:
            return "trading"
        if cv < 0.1:
            return "operations"
    except Exception:
        pass
    
    return "general"


def _classify_tabular_domain(data: dict, metadata: Dict[str, Any]) -> str:
    """Column-name-based domain classification."""
    column_names = metadata.get("column_names", [])
    
    if not column_names:
        return "general"
    
    col_text = " ".join(str(c).lower() for c in column_names)
    
    scores: Dict[str, int] = {}
    for domain_name, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in col_text)
        if count > 0:
            scores[domain_name] = count
    
    if not scores:
        return "general"
    
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def fit_online(
    pre_computed_scores: Dict[str, Any],
    label: str,
) -> None:
    """Update NaiveBayes classifier with confirmed domain label.
    
    Args:
        pre_computed_scores: Pre-computed analysis scores with features
        label: Confirmed domain label
    """
    try:
        # Extract features from pre_computed_scores
        features = {}
        
        # Add text features if available
        if "word_count" in pre_computed_scores:
            features["word_count"] = float(pre_computed_scores["word_count"])
        if "urgency_score" in pre_computed_scores:
            features["urgency_score"] = float(pre_computed_scores["urgency_score"])
        if "sentiment_score" in pre_computed_scores:
            features["sentiment_score"] = float(pre_computed_scores["sentiment_score"])
        
        # Add entity-based features
        entities = pre_computed_scores.get("entities", [])
        if entities:
            features["entity_count"] = float(len(entities))
            # Check for temperature patterns
            temp_entities = [e for e in entities if "°" in str(e) or "C" in str(e) or "F" in str(e)]
            features["has_temperature"] = 1.0 if temp_entities else 0.0
            # Check for device patterns
            device_entities = [e for e in entities if any(x in str(e).upper() for x in ["NODE", "TMP", "SERVER", "ROUTER"])]
            features["has_device"] = 1.0 if device_entities else 0.0
        
        # Only update if we have meaningful features
        if features and label in _classifier.classes:
            _classifier.fit_online(features, label)
    except Exception:
        # Graceful fail - don't break pipeline for online learning errors
        pass


# --- Attention-based enhancement (optional) ---
_ATTENTION_AVAILABLE = False
try:
    from ...neural.attention import AttentionContextCollector
    from ....text.analyzers.keyword_config import ATTENTION_CONFIG
    _ATTENTION_AVAILABLE = True
except Exception:
    pass

def _extract_attention_features(
    text: str, 
    attention_context: Optional[object] = None,
) -> Dict[str, float]:
    """Extract attention-based features for domain classification."""
    features: Dict[str, float] = {}
    
    if attention_context is None and _ATTENTION_AVAILABLE:
        try:
            vocab = {kw: i for i, kw in enumerate(
                [w for kws in DOMAIN_KEYWORDS.values() for w in kws][:ATTENTION_CONFIG["D_MODEL"]]
            )}
            collector = AttentionContextCollector(
                vocab=vocab, n_heads=ATTENTION_CONFIG["N_HEADS"], d_model=ATTENTION_CONFIG["D_MODEL"],
            )
            attention_context = collector.collect_context(text, budget_ms=50.0)
        except Exception:
            pass
    
    if attention_context:
        domain_scores = attention_context.multi_domain_scores
        if domain_scores:
            total = sum(domain_scores.values())
            if total > 0:
                probs = {k: v/total for k, v in domain_scores.items()}
                entropy = -sum(p * __import__('math').log(p) for p in probs.values() if p > 0)
                features["attention_domain_entropy"] = entropy
                features["attention_domain_confidence"] = 1.0 - entropy / 1.5
                
        temporal = attention_context.temporal_markers
        if temporal:
            features["attention_temporal_score"] = max(temporal.values())
            
        features["attention_confidence"] = attention_context.confidence
    
    return features
