#!/usr/bin/env python3
"""Generator: ZENIN High-Tech Interactive Observatory (v3 - Complete Rewrite)."""
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


def build_rich_html_content(records: list[dict]) -> str:
    records_json = json.dumps(records)

    theta_grid = np.linspace(-np.pi, np.pi, 45).tolist()
    div_grid   = np.linspace(-1.0, 2.5, 45).tolist()
    T_g, D_g   = np.meshgrid(theta_grid, div_grid)
    s3_val     = 0.85**2 - 0.40**2
    s1_mesh    = 2.0 * 0.85 * 0.40 * np.cos(T_g)
    c_sov_mesh = np.abs(s3_val + s1_mesh)
    liouv_mesh = np.exp(-np.maximum(0.0, D_g))
    D_mesh     = np.clip(liouv_mesh * c_sov_mesh, 0.0, 1.5).tolist()
    surface_json = json.dumps({"theta": theta_grid, "div": div_grid, "z": D_mesh})

    u_sp = np.linspace(0, 2 * np.pi, 36)
    v_sp = np.linspace(0, np.pi, 28)
    sphere_json = json.dumps({
        "x": np.outer(np.cos(u_sp), np.sin(v_sp)).tolist(),
        "y": np.outer(np.sin(u_sp), np.sin(v_sp)).tolist(),
        "z": np.outer(np.ones_like(u_sp), np.cos(v_sp)).tolist(),
    })

    regime_boundaries_json = json.dumps([
        {"x0": 0.0,  "x1": 2.5,  "label": "Deriva Estacionaria"},
        {"x0": 2.5,  "x1": 3.75, "label": "Inyeccion de Ruido"},
        {"x0": 3.75, "x1": 6.0,  "label": "Choque de Volatilidad"},
        {"x0": 6.0,  "x1": 8.25, "label": "Rebote Anti-fase (MRT)"},
        {"x0": 8.25, "x1": 11.0, "label": "Recuperacion del Atractor"},
    ])

    # Load the HTML template
    tmpl_path = Path(__file__).resolve().parent / "zenin_observatory_template.html"
    with open(tmpl_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    return (html_template
            .replace("__RECORDS_JSON__", records_json)
            .replace("__SURFACE_JSON__", surface_json)
            .replace("__SPHERE_JSON__", sphere_json)
            .replace("__REGIME_BOUNDS__", regime_boundaries_json))


def main() -> None:
    from scripts.zenin_visualizer_2d_3d import generate_multiregime_simulation
    records  = generate_multiregime_simulation(220)
    out_html = Path("results/visualizations/zenin_behavior_interactive.html")
    content  = build_rich_html_content(records)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Observatory HTML: {out_html.resolve()} ({len(content)//1024} KB)")


if __name__ == "__main__":
    main()
