"""Weaviate GraphQL query builder and result parser."""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

_WEAVIATE_CLASS = "ZeninDocument"


def build_graphql_query(
    query: str,
    limit: int,
    where_filter: Optional[Dict[str, Any]],
) -> str:
    """Build a Weaviate GraphQL nearText query."""
    where_clause = ""
    if where_filter:
        path = json.dumps(where_filter["path"])
        op = where_filter["operator"]
        val = json.dumps(where_filter.get("valueString", ""))
        where_clause = (
            f', where: {{path: {path}, operator: {op}, valueString: {val}}}'
        )
    return f"""
    {{
        Get {{
            {_WEAVIATE_CLASS}(
                nearText: {{concepts: [{json.dumps(query)}]}}
                limit: {limit}
                {where_clause}
            ) {{
                content
                source
                classification
                tenantId
                analysisResultId
                contextType
                _additional {{
                    id
                    certainty
                }}
            }}
        }}
    }}
    """


def parse_graphql_results(
    data: Dict[str, Any],
    tenant_id: str,
) -> List[Dict[str, Any]]:
    """Parse Weaviate GraphQL response into result dicts."""
    try:
        items = (
            data.get("data", {})
            .get("Get", {})
            .get(_WEAVIATE_CLASS, [])
        )
    except Exception:
        return []
    results = []
    for item in items or []:
        additional = item.get("_additional", {})
        results.append(
            {
                "doc_id": additional.get("id", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "classification": item.get("classification", ""),
                "score": float(additional.get("certainty", 0.0)),
                "tenant_id": item.get("tenantId", tenant_id),
                "analysis_result_id": item.get("analysisResultId") or None,
                "context_type": item.get("contextType") or None,
            }
        )
    return results
