#!/bin/bash
# Script para ejecutar benchmarks de mantenimiento predictivo y detección de anomalías

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================="
echo "Ejecutando benchmarks de ZENIN ML..."
echo "=================================="

cd "$PROJECT_ROOT"

# NAB Machine Temperature Benchmark (Canónico)
echo ""
echo "=================================="
echo "1. NAB Machine Temperature Benchmark"
echo "=================================="
python3 benchmarks/nab_machine_temp_benchmark.py

echo ""
echo "=================================="
echo "Benchmarks completados con éxito."
echo "Resultados consolidados en: benchmarks/results/"
echo "=================================="
