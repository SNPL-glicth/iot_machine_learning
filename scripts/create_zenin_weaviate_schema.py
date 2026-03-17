"""Create Weaviate schema for ZENIN documents.

Creates a collection for document embeddings with metadata.
"""

import weaviate
import os
import sys

def create_zenin_schema():
    """Create ZENIN document schema in Weaviate."""
    
    # Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    client = weaviate.Client(weaviate_url)
    
    # Check if collection exists
    try:
        existing = client.schema.get("ZeninDocument")
        print(f"Collection ZeninDocument already exists")
        return
    except:
        pass
    
    # Define schema
    schema = {
        "class": "ZeninDocument",
        "description": "ZENIN document embeddings for semantic search",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "poolingStrategy": "masked_mean",
                "vectorizeClassName": False
            }
        },
        "properties": [
            {
                "name": "documentId",
                "dataType": ["text"],
                "description": "UUID del documento en PostgreSQL",
                "indexInverted": True
            },
            {
                "name": "tenantId",
                "dataType": ["text"],
                "description": "UUID del tenant",
                "indexInverted": True
            },
            {
                "name": "filename",
                "dataType": ["text"],
                "description": "Nombre original del archivo",
                "indexInverted": True
            },
            {
                "name": "contentType",
                "dataType": ["text"],
                "description": "Tipo de contenido: tabular, text, image, audio, binary",
                "indexInverted": True
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Contenido textual extraído del documento",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizePropertyName": False
                    }
                }
            },
            {
                "name": "rawText",
                "dataType": ["text"],
                "description": "Texto raw completo",
                "indexInverted": False
            },
            {
                "name": "conclusion",
                "dataType": ["text"],
                "description": "Conclusión generada por ML",
                "indexInverted": False
            },
            {
                "name": "uploadedAt",
                "dataType": ["date"],
                "description": "Timestamp de subida"
            },
            {
                "name": "metadata",
                "dataType": ["text"],
                "description": "Metadata adicional como JSON string",
                "indexInverted": False
            }
        ]
    }
    
    # Create schema
    client.schema.create_class(schema)
    print(f"✅ Collection ZeninDocument created successfully")
    print(f"   URL: {weaviate_url}")
    print(f"   Vectorizer: text2vec-transformers")
    print(f"   Properties: documentId, tenantId, filename, contentType, content, rawText, conclusion, uploadedAt, metadata")


if __name__ == "__main__":
    try:
        create_zenin_schema()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
