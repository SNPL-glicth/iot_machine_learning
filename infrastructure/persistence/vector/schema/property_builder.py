"""Property builder helpers for Weaviate schema."""

from __future__ import annotations

from typing import Any, Dict


def build_property(
    name: str,
    data_type: str,
    description: str,
    *,
    skip_vectorization: bool = False,
    tokenization: str | None = None,
) -> Dict[str, Any]:
    """Build a Weaviate property dict.

    Args:
        name: Property name (camelCase).
        data_type: Weaviate data type (``"text"``, ``"number"``, ``"int"``,
            ``"boolean"``, ``"date"``, ``"text[]"``).
        description: Human-readable description.
        skip_vectorization: If ``True``, this property is excluded from
            the vectorization input.  Use for IDs, scores, and structured
            fields that should not influence semantic similarity.
        tokenization: Weaviate tokenization strategy for text fields.
            ``"field"`` = exact match (for IDs, enums).
            ``"word"`` = standard tokenization (default for text).
            ``None`` = use Weaviate default.
    """
    prop: Dict[str, Any] = {
        "name": name,
        "dataType": [data_type],
        "description": description,
    }

    module_config: Dict[str, Any] = {}

    if skip_vectorization:
        module_config["text2vec-transformers"] = {
            "skip": True,
            "vectorizePropertyName": False,
        }

    if module_config:
        prop["moduleConfig"] = module_config

    if tokenization is not None and data_type == "text":
        prop["tokenization"] = tokenization

    return prop
