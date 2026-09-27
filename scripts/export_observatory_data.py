#!/usr/bin/env python3
"""Export simulation data for ZENIN Modular Observatory."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_REPO_ROOT = (Path(__file__).resolve().parent.parent
              if Path(__file__).resolve().parent.name == "scripts"
              else Path(__file__).resolve().parent)
_ST_ROOT = _REPO_ROOT.parent
for p in (str(_ST_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.zenin_visualizer_2d_3d import generate_multiregime_simulation

def generate_and_export():
    records = generate_multiregime_simulation(220)

    # 3D Decision Surface mesh
    theta_grid = np.linspace(-np.pi, np.pi, 45).tolist()
    div_grid   = np.linspace(-1.0, 2.5, 45).tolist()
    T_g, D_g   = np.meshgrid(theta_grid, div_grid)
    s3_val     = 0.85**2 - 0.40**2
    s1_mesh    = 2.0 * 0.85 * 0.40 * np.cos(T_g)
    c_sov_mesh = np.abs(s3_val + s1_mesh)
    liouv_mesh = np.exp(-np.maximum(0.0, D_g))
    D_mesh     = np.clip(liouv_mesh * c_sov_mesh, 0.0, 1.5).tolist()
    surf_data  = {"theta": theta_grid, "div": div_grid, "z": D_mesh}

    # Stokes Unit Sphere mesh
    u_sp = np.linspace(0, 2 * np.pi, 36)
    v_sp = np.linspace(0, np.pi, 28)
    sphere_mesh = {
        "x": np.outer(np.cos(u_sp), np.sin(v_sp)).tolist(),
        "y": np.outer(np.sin(u_sp), np.sin(v_sp)).tolist(),
        "z": np.outer(np.ones_like(u_sp), np.cos(v_sp)).tolist(),
    }

    regime_bands = [
        {"x0": 0.0,  "x1": 2.5,  "label": "Deriva Estacionaria"},
        {"x0": 2.5,  "x1": 3.75, "label": "Inyeccion de Ruido"},
        {"x0": 3.75, "x1": 6.0,  "label": "Choque de Volatilidad"},
        {"x0": 6.0,  "x1": 8.25, "label": "Rebote Anti-fase (MRT)"},
        {"x0": 8.25, "x1": 11.0, "label": "Recuperacion del Atractor"},
    ]

    out_file = Path("results/visualizations/zenin_observatory/data/simulation_data.js")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Build window.Z precomputed arrays
    data_payload = {
        "records": records,
        "surfData": surf_data,
        "sphereMesh": sphere_mesh,
        "regimeBands": regime_bands,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("/* ZENIN Simulation Precomputed Data */\n")
        f.write("window.RAW_DATA = ")
        json.dump(data_payload, f, separators=(',', ':'))
        f.write(";\n\n")
        f.write("""
(function() {
  const R = window.RAW_DATA.records;
  const N = R.length;
  const Z = {
    records: R,
    N: N,
    surfData: window.RAW_DATA.surfData,
    sphereMesh: window.RAW_DATA.sphereMesh,
    regimeBands: window.RAW_DATA.regimeBands,
    times: R.map(r => r.time),
    xs: R.map(r => r.x),
    vels: R.map(r => r.v),
    accs: R.map(r => r.a),
    dMs: R.map(r => r.d_mahal),
    rets: R.map(r => r.ret),
    rTs: R.map(r => r.R_t),
    rTsN: R.map(r => -r.R_t),
    cvars: R.map(r => r.cvar_t),
    lams: R.map(r => r.lambda_crono),
    phis: R.map(r => r.phi_base),
    s0arr: R.map(r => r.s0),
    s1arr: R.map(r => r.s1),
    s2arr: R.map(r => r.s2),
    s3arr: R.map(r => r.s3),
    s1n: R.map(r => r.s1 / Math.max(1e-9, r.s0)),
    s2n: R.map(r => r.s2 / Math.max(1e-9, r.s0)),
    s3n: R.map(r => r.s3 / Math.max(1e-9, r.s0)),
    pds: R.map(r => r.phase_delta),
    dvs: R.map(r => r.div_flow),
    dests: R.map(r => r.dest_var),
    liovs: R.map(r => r.liouville_factor),
    cSovs: R.map(r => r.sovereign_certainty),
    execIdx: [],
    holdIdx: [],
    flushIdx: [],
    outIdx: [],
    vetoIdx: []
  };

  for (let i = 0; i < N; i++) {
    const r = R[i];
    if (r.action_code === 1) Z.execIdx.push(i);
    else if (r.action_code === -1) Z.flushIdx.push(i);
    else Z.holdIdx.push(i);

    if (r.is_outlier) Z.outIdx.push(i);
    if (r.veto_riesgo === 0) Z.vetoIdx.push(i);
  }

  window.Z = Z;
})();
""")

    print(f"[OK] Exported simulation_data.js ({out_file.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    generate_and_export()
