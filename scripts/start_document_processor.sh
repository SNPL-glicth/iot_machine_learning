#!/bin/bash
# Start ZENIN Document Processor
# Async worker que procesa documentos desde PostgreSQL

cd "$(dirname "$0")/.."

echo "🚀 Starting ZENIN Document Processor..."
echo "   Polling zenin_docs.documents every 5 seconds"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m ml_service.runners.document_processor
