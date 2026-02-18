"""Migration runner for Weaviate schema creation."""

from __future__ import annotations

from typing import Any, Dict


def create_class_v4(client: Any, class_def: Dict[str, Any]) -> None:
    """Create a single class using the weaviate v4 client API.

    Falls back to the raw REST endpoint if the high-level API
    does not support the full property config.
    """
    import weaviate.classes.config as wvc

    # Map data type strings to weaviate v4 DataType enums
    _type_map = {
        "text": wvc.DataType.TEXT,
        "text[]": wvc.DataType.TEXT_ARRAY,
        "int": wvc.DataType.INT,
        "number": wvc.DataType.NUMBER,
        "boolean": wvc.DataType.BOOL,
        "date": wvc.DataType.DATE,
    }

    properties = []
    for p in class_def["properties"]:
        dt_str = p["dataType"][0]
        dt = _type_map.get(dt_str)
        if dt is None:
            raise ValueError(f"Unknown data type: {dt_str}")

        skip = False
        vectorize_name = True
        mc = p.get("moduleConfig", {}).get("text2vec-transformers", {})
        if mc:
            skip = mc.get("skip", False)
            vectorize_name = mc.get("vectorizePropertyName", True)

        tokenization_val = None
        tok_str = p.get("tokenization")
        if tok_str == "field":
            tokenization_val = wvc.Tokenization.FIELD
        elif tok_str == "word":
            tokenization_val = wvc.Tokenization.WORD

        prop_kwargs: Dict[str, Any] = {
            "name": p["name"],
            "data_type": dt,
            "description": p.get("description", ""),
            "skip_vectorization": skip,
            "vectorize_property_name": vectorize_name,
        }
        if tokenization_val is not None:
            prop_kwargs["tokenization"] = tokenization_val

        properties.append(wvc.Property(**prop_kwargs))

    client.collections.create(
        name=class_def["class"],
        description=class_def.get("description", ""),
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(
            vectorize_collection_name=False,
        ),
        properties=properties,
    )
