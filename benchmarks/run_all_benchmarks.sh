#!/bin/bash
# Script para ejecutar todos los benchmarks de datasets de mantenimiento predictivo

echo "=================================="
echo "Ejecutando todos los benchmarks..."
echo "=================================="

cd /home/nicolas/Documentos/Proyectos/Zenin-Iot/iot_machine_learning

# NASA C-MAPSS Benchmark
echo ""
echo "=================================="
echo "1. NASA C-MAPSS Turbofan Engine Degradation Benchmark"
echo "=================================="
python benchmarks/nasa_cmapss_benchmark.py

# SKAB Benchmark
echo ""
echo "=================================="
echo "2. SKAB (Skoltech Anomaly Benchmark)"
echo "=================================="
python benchmarks/skab_benchmark.py

# UCI AI4I 2020 Benchmark
echo ""
echo "=================================="
echo "3. UCI AI4I 2020 Predictive Maintenance Dataset"
echo "=================================="
python benchmarks/uci_ai4i_benchmark.py

echo ""
echo "=================================="
echo "Todos los benchmarks completados"
echo "Resultados guardados en: results/"
echo "=================================="
