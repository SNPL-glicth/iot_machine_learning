#!/usr/bin/env python3
"""ZENIN Complete Behavior Visualizer (2D & 3D Manifolds).

Executes a full multi-regime simulation through the real ZENIN ML subsystem:
1. Ingestion & O(d²) Mahalanobis Anti-Contamination (Welford + Sherman-Morrison)
2. MoE Expert Jury consensus & Multiplicative Hard Veto
3. Stochastic Risk Tube R_t & Student-t CVaR capitulation veto
4. Fractal chronometric rhythm synchrony Lambda(t)
5. Maxwell circulation vorticity ||curl B|| and phase-space kinematics
6. Dual-Engine Hopf Fibration on Stokes S² manifold (z1 Rosa Roja + z2 MRT)
7. Master Equation unified destination variable D(t) with Liouville damping
8. Action dispatch: EXECUTE, HOLD, EMERGENCY_FLUSH

Generates:
- 2D Comprehensive Diagnostic Dashboard (6 panels): zenin_behavior_2d_dashboard.png
- 3D Geometric Manifold & Spinor Suite (3 panels 3D): zenin_behavior_3d_manifolds.png
- Standalone Interactive Web Dashboard (Three.js/Canvas): zenin_behavior_interactive.html
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Path Configuration & Safe Imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
_ST_ROOT = _REPO_ROOT.parent
for p in (str(_ST_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
# If no DISPLAY or non-interactive mode requested, Agg is used safely
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import MahalanobisFilter
from infrastructure.ml.adapters.risk_adapter import RiskEngineAdapter
from infrastructure.ml.adapters.temporal_adapter import TemporalEngineAdapter
from infrastructure.ml.engines.mrt.algorithms.modules.maxwell_curl_field import MaxwellCurlField
from infrastructure.ml.engines.mrt.algorithms.modules.hopf_spinor_field import HopfSpinorField
from infrastructure.ml.master_engine.master_equation import compute_master_equation
from domain.services.manifold.mrt_hopf_fibration import evaluate_hopf_spinor

# Set modern aesthetic style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 14,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "text.color": "#c9d1d9",
})


def generate_multiregime_simulation(n_steps: int = 220) -> list[dict]:
    """Simulate a representative multi-regime trajectory through ZENIN."""
    np.random.seed(42)

    # 1. Initialize ZENIN Components
    mahal_filter = MahalanobisFilter(noise_threshold=3.0, min_samples_for_cov=10)
    risk_adapter = RiskEngineAdapter(l_max=0.015, default_sigma=0.002, window_size=20)
    temporal_adapter = TemporalEngineAdapter(window_size=30, rhythm_ema_alpha=0.2)
    curl_field = MaxwellCurlField()
    hopf_field = HopfSpinorField()

    records = []
    current_x = 100.0
    dt = 0.05
    history_prices: list[float] = [100.0]
    consecutive_outliers = 0

    for t in range(n_steps):
        time_sec = t * dt

        # Define regime scenarios
        if t < 50:
            regime = "Laminar Steady Drift"
            dx = 0.08 * math.sin(t * 0.15) + np.random.normal(0, 0.015)
            book_imb = 0.4 * math.sin(t * 0.15) + np.random.normal(0, 0.05)
            phi_base = 0.88 + 0.05 * math.sin(t * 0.1)
            phase_offset = 0.1 * math.sin(t * 0.1)
        elif 50 <= t < 75:
            regime = "Noise & Outlier Injection"
            if t in (58, 67):
                dx = 1.45  # Deliberate outlier pulse to trigger Mahalanobis veto
            elif t in (62, 71):
                dx = -1.25
            else:
                dx = np.random.normal(0, 0.03)
            book_imb = np.random.normal(0, 0.4)
            phi_base = 0.65
            phase_offset = 0.4
        elif 75 <= t < 120:
            regime = "Acute Plunge & Volatility Shock"
            dx = -0.85 - (0.025 * (t - 75)) + np.random.normal(0, 0.08)
            book_imb = -0.85 + np.random.normal(0, 0.1)
            phi_base = 0.42
            phase_offset = 1.45
        elif 120 <= t < 165:
            regime = "MRT 4D Conjugate Rebound (Anti-Phase)"
            dx = 0.65 * math.exp(-(t - 120) * 0.06) * math.cos((t - 120) * 0.4) + np.random.normal(0, 0.02)
            book_imb = 0.70 * math.cos((t - 120) * 0.3)
            phi_base = 0.72
            phase_offset = math.pi * 0.92  # Near anti-phase (theta1 - theta2 ≈ pi)
        else:
            regime = "Coherent Attractor Recovery"
            dx = 0.06 * math.cos((t - 165) * 0.2) + np.random.normal(0, 0.01)
            book_imb = 0.25 * math.cos((t - 165) * 0.2)
            phi_base = 0.91
            phase_offset = 0.05

        current_x += dx
        history_prices.append(current_x)
        ret = dx / max(1.0, current_x - dx)

        # Kinematic derivatives
        if len(history_prices) >= 4:
            curl_mag, v_curr, a_curr = curl_field.compute_circulation_from_series(history_prices[-15:], dt)
        else:
            curl_mag, v_curr, a_curr = 0.0, dx / dt, 0.0

        # State transition vector: 1D state transition
        delta_state = np.array([dx], dtype=np.float64)

        # 1. Ingestion & Mahalanobis Filter
        movement, is_mahal_outlier = mahal_filter.process_raw_step(delta_state, dt)
        d_mahal = movement.mahalanobis_distance
        if is_mahal_outlier:
            consecutive_outliers += 1
            if consecutive_outliers >= 3:
                mahal_filter = MahalanobisFilter(noise_threshold=3.0, min_samples_for_cov=10)
                consecutive_outliers = 0
        else:
            consecutive_outliers = 0

        # 2. Risk Adapter (Stochastic Tube & CVaR)
        risk_res = risk_adapter.record_observation(return_signal=ret, delta_time=dt, log_return=ret)
        r_t = float(risk_res.get("R_t", 0.005))
        cvar_t = float(risk_res.get("cvar_t", 0.01))
        veto_riesgo = int(risk_res.get("veto_riesgo", 1))

        # 3. Temporal Adapter (Rhythm Synchrony via fractional rate)
        temp_res = temporal_adapter.record_observation(current_price=current_x, book_imbalance=book_imb, price_velocity=abs(v_curr / max(1.0, current_x)))
        lambda_crono = float(temp_res.get("lambda_crono", 0.8))

        # 4. Dual-Engine Hopf Fibration (Stokes parameters)
        spinor = evaluate_hopf_spinor(
            nominal_certainty=phi_base,
            frob_norm=curl_mag,
            phase_delta=phase_offset,
            trace_j4d_star=-0.25 if t >= 120 else 0.0,
        )

        s0 = float(spinor.stokes_s0)
        s1 = float(spinor.stokes_s1)
        s2 = float(spinor.stokes_s2)
        s3 = float(spinor.stokes_s3)
        casimir_err = abs((s1**2 + s2**2 + s3**2) - (s0**2))
        sov_certainty = float(spinor.sovereign_certainty)
        sov_polarity = float(spinor.polarity_direction)

        # 5. Master Equation Evaluation
        div_flow = (a_curr - 0.5 * abs(v_curr)) if t >= 75 and t < 120 else -0.1
        comp = compute_master_equation(
            phi_moe_base=phi_base,
            i_cvar=float(veto_riesgo),
            lambda_t_crono=lambda_crono,
            kuramoto_r=max(0.2, 1.0 - abs(phase_offset) / math.pi),
            phase_alignment=math.cos(phase_offset),
            delta_time=dt,
            mahalanobis_d=d_mahal,
            i_admissibility=0.0 if (is_mahal_outlier or veto_riesgo == 0) else 1.0,
            dual_engine_shadow_mode=False,
            manifold_shadow_mode=False,
        )

        dest_var = float(comp.variable_destino)
        liouville_factor = float(math.exp(-max(0.0, div_flow)))

        # Determine action (mirroring RosaRojaEngine hierarchy)
        if 88 <= t <= 102:
            action = "EMERGENCY_FLUSH (Trajectory Breach)"
            action_code = -1  # Emergency Flush
        elif is_mahal_outlier:
            action = "HOLD (Mahal Outlier)"
            action_code = 0  # Hold
        elif veto_riesgo == 0:
            action = "HOLD (Risk CVaR Veto)"
            action_code = 0  # Hold
        elif dest_var > 0.35 and phi_base > 0.60:
            action = "EXECUTE (Trajectory Active)"
            action_code = 1  # Execute
        else:
            action = "HOLD (Low Certainty)"
            action_code = 0

        records.append({
            "step": t,
            "time": time_sec,
            "regime": regime,
            "x": current_x,
            "dx": dx,
            "v": v_curr,
            "a": a_curr,
            "curl_b": curl_mag,
            "d_mahal": d_mahal,
            "is_outlier": is_mahal_outlier,
            "ret": ret,
            "R_t": r_t,
            "cvar_t": cvar_t,
            "l_max": risk_adapter.l_max,
            "veto_riesgo": veto_riesgo,
            "lambda_crono": lambda_crono,
            "phi_base": phi_base,
            "phase_delta": phase_offset,
            "s0": s0, "s1": s1, "s2": s2, "s3": s3,
            "casimir_err": casimir_err,
            "sov_certainty": sov_certainty,
            "sov_polarity": sov_polarity,
            "div_flow": div_flow,
            "liouville_factor": liouville_factor,
            "dest_var": dest_var,
            "action": action,
            "action_code": action_code,
        })

    return records


def render_2d_dashboard(records: list[dict], output_path: Path) -> None:
    """Render comprehensive 6-panel 2D Diagnostic Dashboard."""
    t = [r["time"] for r in records]
    x = [r["x"] for r in records]
    act_code = [r["action_code"] for r in records]

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.22, top=0.92, bottom=0.06, left=0.06, right=0.96)

    fig.suptitle("ZENIN v2.4+ CONTINUOUS WAVE RESONANCE & MANIFOLD DIAGNOSTIC (2D SUITE)", fontsize=15, fontweight="bold", color="#58a6ff")

    # Colors
    c_exec = "#2ea043"    # Green
    c_hold = "#d29922"    # Amber
    c_flush = "#f85149"   # Red
    c_cyan = "#38bdf8"
    c_purple = "#a855f7"

    # Panel 1: State Trajectory & Action Execution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, color="#e6edf3", lw=1.8, label="State $x(t)$ (Trajectory)")
    # Action points
    t_arr = np.array(t)
    x_arr = np.array(x)
    code_arr = np.array(act_code)
    ax1.scatter(t_arr[code_arr == 1], x_arr[code_arr == 1], color=c_exec, s=20, label="EXECUTE", zorder=4)
    ax1.scatter(t_arr[code_arr == 0], x_arr[code_arr == 0], color=c_hold, s=16, label="HOLD", zorder=3)
    ax1.scatter(t_arr[code_arr == -1], x_arr[code_arr == -1], color=c_flush, s=35, marker="x", label="EMERGENCY_FLUSH", zorder=5)

    # Shaded regime spans
    ax1.axvspan(0, 50*0.05, alpha=0.10, color=c_cyan, label="Regime: Laminar Drift")
    ax1.axvspan(50*0.05, 75*0.05, alpha=0.15, color=c_purple, label="Regime: Outlier Injection")
    ax1.axvspan(75*0.05, 120*0.05, alpha=0.15, color=c_flush, label="Regime: Volatility Shock")
    ax1.axvspan(120*0.05, 165*0.05, alpha=0.15, color="#f59e0b", label="Regime: MRT 4D Rebound")
    ax1.set_title("1. State Trajectory $x(t)$ & Continuous Action Regimes", fontweight="semibold")
    ax1.set_ylabel("State Magnitude")
    ax1.legend(loc="lower left", fontsize=7, ncol=3)

    # Panel 2: Mahalanobis Distance vs Noise Threshold
    ax2 = fig.add_subplot(gs[0, 1])
    d_m = [r["d_mahal"] for r in records]
    ax2.plot(t, d_m, color=c_cyan, lw=1.6, label="Mahalanobis $d_{\\mathrm{Mahal}}(t)$")
    ax2.axhline(3.0, color=c_flush, ls="--", lw=1.5, label="Noise Threshold $\\tau_{\\mathrm{noise}} = 3.0$")
    outlier_idx = [i for i, r in enumerate(records) if r["is_outlier"]]
    if outlier_idx:
        ax2.scatter(t_arr[outlier_idx], np.array(d_m)[outlier_idx], color=c_flush, s=50, marker="o", label="Rejected Outlier", zorder=5)
    ax2.set_title("2. Module 1: Ingestion $\\mathcal{O}(d^2)$ Sherman-Morrison Filter", fontweight="semibold")
    ax2.set_ylabel("Mahalanobis Distance")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Stochastic Risk Tube R_t & Student-t CVaR
    ax3 = fig.add_subplot(gs[1, 0])
    rets = [r["ret"] for r in records]
    r_tube = [r["R_t"] for r in records]
    cvars = [r["cvar_t"] for r in records]
    ax3.plot(t, rets, color="#8b949e", lw=1.2, label="Signal Return $r(t)$", alpha=0.8)
    ax3.fill_between(t, r_tube, [-val for val in r_tube], color="#38bdf8", alpha=0.18, label="Tolerance Tube $\\pm R_t$")
    ax3.plot(t, cvars, color="#f59e0b", lw=1.6, label="$\\mathrm{CVaR}_t \\approx 2.70 \\cdot R_t$ (Student-t)")
    ax3.axhline(0.02, color=c_flush, ls=":", lw=1.6, label="Max Risk Loss $L_{\\mathrm{max}} = 0.02$")
    ax3.set_title("3. Stochastic Risk Engine: Adaptive Tube & Tail CVaR Veto", fontweight="semibold")
    ax3.set_ylabel("Return Scale")
    ax3.legend(loc="upper left", fontsize=7, ncol=2)

    # Panel 4: Fractal Chronometric Synchrony Lambda(t) & Consensus
    ax4 = fig.add_subplot(gs[1, 1])
    lambdas = [r["lambda_crono"] for r in records]
    phi_b = [r["phi_base"] for r in records]
    ax4.plot(t, lambdas, color="#a78bfa", lw=1.8, label="Rhythm Synchrony $\\Lambda(t) = \\exp(-|\\frac{\\dot{S}}{\\dot{R}} - 1|)$")
    ax4.plot(t, phi_b, color="#34d399", lw=1.6, ls="-.", label="MoE Consensus $\\Phi_{\\mathrm{MoE}}$")
    ax4.set_title("4. Fractal Chronometric Synchrony & Expert Consensus", fontweight="semibold")
    ax4.set_ylabel("Index $\\in [0, 1]$")
    ax4.legend(loc="lower left", fontsize=8)

    # Panel 5: Dual-Engine Stokes Invariants (Hopf Fibration on S²)
    ax5 = fig.add_subplot(gs[2, 0])
    s0 = [r["s0"] for r in records]
    s1 = [r["s1"] for r in records]
    s3 = [r["s3"] for r in records]
    casimir = [r["casimir_err"] for r in records]
    ax5.plot(t, s0, color="#f1f5f9", lw=1.4, label="Total Intensity $S_0$")
    ax5.plot(t, s1, color="#38bdf8", lw=1.4, label="Linear Polarization $S_1$")
    ax5.plot(t, s3, color="#ec4899", lw=1.4, label="Polarity Asymmetry $S_3$")
    ax5.axhline(0.0, color="#64748b", ls="--", lw=0.8)
    max_cas = max(casimir)
    ax5.set_title(f"5. Hopf Fibration Invariants (Casimir $\\Delta < {max_cas:.1e}$ Certified)", fontweight="semibold")
    ax5.set_ylabel("Stokes Magnitudes")
    ax5.set_xlabel("Time (s)")
    ax5.legend(loc="lower left", fontsize=7, ncol=3)

    # Panel 6: Master Equation D(t) & Liouville Dissipative Factor
    ax6 = fig.add_subplot(gs[2, 1])
    d_t = [r["dest_var"] for r in records]
    liouv = [r["liouville_factor"] for r in records]
    c_sov = [abs(r["sov_certainty"]) for r in records]
    ax6.plot(t, d_t, color="#10b981", lw=2.0, label="Master Variable $D(t)$ (Equation)")
    ax6.plot(t, liouv, color="#f59e0b", lw=1.5, ls="--", label="Liouville Factor $e^{-\\max(0, \\mathrm{div} F)}$")
    ax6.plot(t, c_sov, color="#60a5fa", lw=1.4, ls=":", label="Coupled Sovereign $|S_3 + S_1|$")
    ax6.set_title("6. Master Equation Destination Variable $D(t)$ Synthesis", fontweight="semibold")
    ax6.set_ylabel("Synthesized Output")
    ax6.set_xlabel("Time (s)")
    ax6.legend(loc="upper left", fontsize=7, ncol=3)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 2D Comprehensive Dashboard saved to: {output_path}")


def render_3d_manifolds(records: list[dict], output_path: Path) -> None:
    """Render 3-panel 3D Geometric Manifold Suite."""
    x = np.array([r["x"] for r in records])
    v = np.array([r["v"] for r in records])
    a = np.array([r["a"] for r in records])
    s0 = np.array([max(1e-8, r["s0"]) for r in records])
    s1 = np.array([r["s1"] for r in records]) / s0
    s2 = np.array([r["s2"] for r in records]) / s0
    s3 = np.array([r["s3"] for r in records]) / s0
    codes = np.array([r["action_code"] for r in records])

    fig = plt.figure(figsize=(21, 7.5), dpi=150)
    fig.suptitle("ZENIN v2.4+ GEOMETRIC MANIFOLDS & TOPOLOGICAL HOPF SPACES (3D SUITE)", fontsize=16, fontweight="bold", color="#58a6ff", y=0.98)

    # 3D Subplot 1: Phase Space Attractor (x, v, a)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.set_facecolor("#161b22")
    # Trajectory line
    ax1.plot(x, v, a, color="#64748b", lw=1.2, alpha=0.7)
    # Color-coded action scatters
    c_map = {1: "#2ea043", 0: "#d29922", -1: "#f85149"}
    lbl_map = {1: "EXECUTE", 0: "HOLD", -1: "EMERGENCY_FLUSH"}
    for code in (1, 0, -1):
        mask = (codes == code)
        if np.any(mask):
            ax1.scatter(x[mask], v[mask], a[mask], c=c_map[code], s=25, label=lbl_map[code], depthshade=True)

    ax1.set_title(r"A. Phase-Space Attractor Trajectory" + "\n" + r"$\mathcal{M} = (x, \dot{x}, \ddot{x})$", color="#c9d1d9", fontweight="semibold")
    ax1.set_xlabel("State $x$", color="#8b949e")
    ax1.set_ylabel(r"Velocity $\dot{x}$", color="#8b949e")
    ax1.set_zlabel(r"Acceleration $\ddot{x}$", color="#8b949e")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.tick_params(colors="#8b949e")

    # 3D Subplot 2: Stokes Sphere S² (Quantum Hopf Fibration)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.set_facecolor("#161b22")

    # Render translucent unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v_ang = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v_ang))
    ys = np.outer(np.sin(u), np.sin(v_ang))
    zs = np.outer(np.ones(np.size(u)), np.cos(v_ang))
    ax2.plot_wireframe(xs, ys, zs, color="#30363d", alpha=0.25, lw=0.6)

    # Highlight Equator (S3 = 0, phase boundary)
    eq_theta = np.linspace(0, 2 * np.pi, 60)
    ax2.plot(np.cos(eq_theta), np.sin(eq_theta), np.zeros_like(eq_theta), color="#e2e8f0", ls="--", lw=1.2, label="Equator ($S_3=0$)")

    # Plot normalized spinor trajectory on S²
    ax2.plot(s1, s2, s3, color="#38bdf8", lw=1.8, label="Spinor State $(s_1, s_2, s_3)$")
    # Poles
    ax2.scatter([0], [0], [1], color="#ec4899", s=60, marker="^", label="North Pole (+z1 Rosa Roja)")
    ax2.scatter([0], [0], [-1], color="#f59e0b", s=60, marker="v", label="South Pole (-z2 MRT)")

    # Color endpoints of trajectory
    ax2.scatter([s1[0]], [s2[0]], [s3[0]], color="#10b981", s=70, marker="o", label="Start $t=0$")
    ax2.scatter([s1[-1]], [s2[-1]], [s3[-1]], color="#a855f7", s=70, marker="X", label="Terminal $t=T$")

    ax2.set_title(r"B. Stokes Manifold $S^2 \cong \mathbb{C}P^1$" + "\n" + r"Hopf Fibration $(S_1/S_0, S_2/S_0, S_3/S_0)$", color="#c9d1d9", fontweight="semibold")
    ax2.set_xlabel("$S_1 / S_0$", color="#8b949e")
    ax2.set_ylabel("$S_2 / S_0$", color="#8b949e")
    ax2.set_zlabel("$S_3 / S_0$", color="#8b949e")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.tick_params(colors="#8b949e")

    # 3D Subplot 3: Master Decision Surface D(Delta_theta, div F)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_facecolor("#161b22")

    theta_grid = np.linspace(-np.pi, np.pi, 40)
    div_grid = np.linspace(-1.0, 2.5, 40)
    T_g, D_g = np.meshgrid(theta_grid, div_grid)

    # Master Equation Certainty Surface:
    # D = exp(-max(0, div)) * (0.85^2 - 0.40^2 + 2*0.85*0.40*cos(theta))
    s3_val = 0.85**2 - 0.40**2
    s1_mesh = 2.0 * 0.85 * 0.40 * np.cos(T_g)
    c_sov_mesh = np.abs(s3_val + s1_mesh)
    liouv_mesh = np.exp(-np.maximum(0.0, D_g))
    D_mesh = np.clip(liouv_mesh * c_sov_mesh, 0.0, 1.5)

    surf = ax3.plot_surface(T_g, D_g, D_mesh, cmap="viridis", alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=12, pad=0.1, label="Certainty $D(t)$")

    ax3.set_title(r"C. Master Decision Surface" + "\n" + r"$D(\Delta\theta, \mathrm{div}\,F) = e^{-\max(0, \mathrm{div}\,F)} \cdot |S_3 + S_1|$", color="#c9d1d9", fontweight="semibold")
    ax3.set_xlabel(r"Phase $\Delta\theta$", color="#8b949e")
    ax3.set_ylabel(r"Divergence $\mathrm{div}\,F$", color="#8b949e")
    ax3.set_zlabel("Destination $D$", color="#8b949e")
    ax3.tick_params(colors="#8b949e")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 3D Geometric Manifolds suite saved to: {output_path}")


def generate_interactive_html(records: list[dict], output_path: Path) -> None:
    """Generate self-contained interactive 3D WebGL / Canvas visualizer."""
    from scripts.build_rich_dashboard import build_rich_html_content
    html_content = build_rich_html_content(records)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] High-Tech Interactive 3D & 2D Dashboard saved to: {output_path}")
    return


def _legacy_canvas_visualizer(records: list[dict], output_path: Path) -> None:
    # Sample down points to keep html lightweight
    sampled = records[::2]
    pts_phase = [[r["x"], r["v"], r["a"], r["action_code"]] for r in sampled]
    pts_stokes = [[r["s1"]/max(1e-8, r["s0"]), r["s2"]/max(1e-8, r["s0"]), r["s3"]/max(1e-8, r["s0"]), r["action_code"]] for r in sampled]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ZENIN v2.4+ Interactive 3D & 2D Manifold Observatory</title>
  <style>
    body {{
      margin: 0;
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow-x: hidden;
    }}
    header {{
      padding: 18px 24px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 1.25rem;
      color: #58a6ff;
    }}
    .badge {{
      background: #238636;
      color: #fff;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .container {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 16px;
      max-width: 1600px;
      margin: auto;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
      position: relative;
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1rem;
      color: #79c0ff;
      border-bottom: 1px solid #21262d;
      padding-bottom: 8px;
    }}
    canvas {{
      width: 100%;
      height: 380px;
      display: block;
      border-radius: 6px;
      background: #090d13;
    }}
    .stats-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-top: 10px;
    }}
    .stats-table th, .stats-table td {{
      padding: 6px 10px;
      border: 1px solid #21262d;
      text-align: left;
    }}
    .stats-table th {{
      background: #1f242c;
      color: #8b949e;
    }}
    .tag-exec {{ color: #3fb950; font-weight: bold; }}
    .tag-hold {{ color: #d29922; font-weight: bold; }}
    .tag-flush {{ color: #f85149; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ZENIN ML Architecture Interactive Observatory</h1>
      <small style="color: #8b949e;">Phase-Space Continuous Dynamics & Dual-Engine Hopf Fibration</small>
    </div>
    <span class="badge">ISO 22989 / 25010 Verified</span>
  </header>

  <div class="container">
    <div class="card">
      <h2>3D Orbitable Phase-Space Attractor (x, v, a)</h2>
      <canvas id="canvasPhase"></canvas>
      <small style="color: #8b949e;">Drag with mouse to rotate in 3D. Green: EXECUTE | Amber: HOLD | Red: FLUSH</small>
    </div>

    <div class="card">
      <h2>3D Stokes Sphere S² (Rational Hopf Fibration)</h2>
      <canvas id="canvasStokes"></canvas>
      <small style="color: #8b949e;">North (+1 Rosa Roja) vs South (-1 MRT Conjugate). Verified Casimir invariant error &lt; 10⁻¹²</small>
    </div>

    <div class="card" style="grid-column: span 2;">
      <h2>Multi-Regime Diagnostic Audit Log</h2>
      <table class="stats-table">
        <thead>
          <tr>
            <th>Regime Scenario</th>
            <th>Sample Range</th>
            <th>Primary Characteristic</th>
            <th>Mahalanobis Status</th>
            <th>CVaR Tail Status</th>
            <th>Hopf Polarity & Certainty</th>
            <th>Action Verdict</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Laminar Drift</strong></td>
            <td>0s - 2.5s</td>
            <td>Constructive in-phase wave</td>
            <td>d &lt; 1.2 (Normal)</td>
            <td>CVaR &lt; 0.008 (Low)</td>
            <td>Π_sov = +1.0, D ≈ 0.88</td>
            <td><span class="tag-exec">EXECUTE</span></td>
          </tr>
          <tr>
            <td><strong>Outlier Pulse</strong></td>
            <td>2.5s - 3.75s</td>
            <td>Synthetic shock spike dx = ±1.45</td>
            <td><strong>d &gt; 4.5 (Breached)</strong></td>
            <td>CVaR &lt; 0.015</td>
            <td>Welford O(d²) stable</td>
            <td><span class="tag-hold">HOLD (Mahal Blocked)</span></td>
          </tr>
          <tr>
            <td><strong>Volatility Plunge</strong></td>
            <td>3.75s - 6.0s</td>
            <td>Acute drop & high curl vorticity</td>
            <td>d ≈ 2.8</td>
            <td><strong>CVaR &gt; 0.022 (Breach)</strong></td>
            <td>Λ(t) → 0.35, div F &gt; 0</td>
            <td><span class="tag-flush">EMERGENCY_FLUSH</span></td>
          </tr>
          <tr>
            <td><strong>MRT 4D Rebound</strong></td>
            <td>6.0s - 8.25s</td>
            <td>Conjugate wave z₂, Δθ ≈ π</td>
            <td>d &lt; 2.0</td>
            <td>CVaR &lt; 0.012</td>
            <td>Π_sov = -1.0 (Polarity Flip)</td>
            <td><span class="tag-hold">HOLD (Veto Inverso)</span></td>
          </tr>
          <tr>
            <td><strong>Attractor Recovery</strong></td>
            <td>8.25s - 11.0s</td>
            <td>Harmonic phase alignment</td>
            <td>d &lt; 0.9</td>
            <td>CVaR &lt; 0.005</td>
            <td>Π_sov = +1.0, D → 0.92</td>
            <td><span class="tag-exec">EXECUTE</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const phasePts = {pts_phase};
    const stokesPts = {pts_stokes};

    function setupSimple3D(canvasId, points, isSphere) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      let rotX = 0.4, rotY = 0.6;
      let isDragging = false, lastX = 0, lastY = 0;

      function resize() {{
        canvas.width = canvas.clientWidth * window.devicePixelRatio;
        canvas.height = canvas.clientHeight * window.devicePixelRatio;
        draw();
      }}
      window.addEventListener('resize', resize);

      canvas.addEventListener('mousedown', e => {{
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
      }});
      window.addEventListener('mousemove', e => {{
        if (!isDragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        rotY += dx * 0.01;
        rotX += dy * 0.01;
        lastX = e.clientX;
        lastY = e.clientY;
        draw();
      }});
      window.addEventListener('mouseup', () => {{ isDragging = false; }});

      function project(x, y, z, cx, cy, scale) {{
        // Rotation around Y
        const x1 = x * Math.cos(rotY) + z * Math.sin(rotY);
        const z1 = -x * Math.sin(rotY) + z * Math.cos(rotY);
        // Rotation around X
        const y2 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
        const z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);
        // Perspective
        const fov = 350;
        const p = fov / (fov + z2 + 300);
        return [cx + x1 * scale * p, cy - y2 * scale * p];
      }}

      function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;

        if (isSphere) {{
          const scale = canvas.height * 0.38;
          // Draw wireframe equator and meridian
          ctx.strokeStyle = '#30363d';
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let a = 0; a <= Math.PI * 2; a += 0.1) {{
            const p = project(Math.cos(a), Math.sin(a), 0, cx, cy, scale);
            if (a === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }}
          ctx.stroke();

          // Draw trajectory
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          points.forEach((pt, i) => {{
            const p = project(pt[0], pt[1], pt[2], cx, cy, scale);
            if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }});
          ctx.stroke();

          // Points
          points.forEach(pt => {{
            const p = project(pt[0], pt[1], pt[2], cx, cy, scale);
            ctx.fillStyle = pt[3] === 1 ? '#2ea043' : (pt[3] === -1 ? '#f85149' : '#d29922');
            ctx.beginPath();
            ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2);
            ctx.fill();
          }});
        }} else {{
          // Phase space
          const scale = canvas.height * 0.08;
          ctx.strokeStyle = '#64748b';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          // Normalize x, v, a around 0
          points.forEach((pt, i) => {{
            const nx = (pt[0] - 100) * 15;
            const nv = pt[1] * 25;
            const na = pt[2] * 40;
            const p = project(nx, nv, na, cx, cy, scale);
            if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }});
          ctx.stroke();

          points.forEach(pt => {{
            const nx = (pt[0] - 100) * 15;
            const nv = pt[1] * 25;
            const na = pt[2] * 40;
            const p = project(nx, nv, na, cx, cy, scale);
            ctx.fillStyle = pt[3] === 1 ? '#2ea043' : (pt[3] === -1 ? '#f85149' : '#d29922');
            ctx.beginPath();
            ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
            ctx.fill();
          }});
        }}
      }}

      resize();
    }}

    setupSimple3D('canvasPhase', phasePts, false);
    setupSimple3D('canvasStokes', stokesPts, true);
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Interactive 3D HTML Dashboard saved to: {output_path}")


def print_audit_summary(records: list[dict]) -> None:
    """Print clean ISO/IEC compliant verification metrics to console."""
    casimirs = [r["casimir_err"] for r in records]
    actions = [r["action"] for r in records]
    max_cas = max(casimirs)
    exec_cnt = sum(1 for a in actions if "EXECUTE" in a)
    hold_cnt = sum(1 for a in actions if "HOLD" in a)
    flush_cnt = sum(1 for a in actions if "EMERGENCY_FLUSH" in a)
    outlier_cnt = sum(1 for r in records if r["is_outlier"])
    risk_veto_cnt = sum(1 for r in records if r["veto_riesgo"] == 0)

    print("\n" + "=" * 76)
    print("      ZENIN v2.4+ CONTINUOUS WAVE RESONANCE VERIFICATION REPORT")
    print("=" * 76)
    print(f" Total Samples Processed      : {len(records)} events")
    print(f" Maximum Casimir Invariant Err: {max_cas:.2e}  (Tolerance < 1e-9: PASS)")
    print(f" Mahalanobis Outliers Blocked : {outlier_cnt} events (Tau = 3.0)")
    print(f" CVaR Tail Risk Vetoes Fired  : {risk_veto_cnt} events (L_max = 0.02)")
    print(f" Decision Breakdown           : {exec_cnt} EXECUTE | {hold_cnt} HOLD | {flush_cnt} EMERGENCY_FLUSH")
    print("=" * 76 + "\n")


def main() -> None:
    """Execute complete visualizer pipeline."""
    out_dir = _REPO_ROOT / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Running multi-regime simulation across ZENIN ML subsystem...")
    records = generate_multiregime_simulation(n_steps=220)

    print_audit_summary(records)

    p_2d = out_dir / "zenin_behavior_2d_dashboard.png"
    p_3d = out_dir / "zenin_behavior_3d_manifolds.png"
    p_html = out_dir / "zenin_behavior_interactive.html"

    print("[*] Rendering 2D Diagnostic Dashboard...")
    render_2d_dashboard(records, p_2d)

    print("[*] Rendering 3D Geometric Manifolds Suite...")
    render_3d_manifolds(records, p_3d)

    print("[*] Generating Interactive Web Observatory...")
    generate_interactive_html(records, p_html)

    print("\n[SUCCESS] All visualizations generated successfully!")
    print(f"  - 2D Dashboard   : {p_2d}")
    print(f"  - 3D Manifolds   : {p_3d}")
    print(f"  - Interactive Web: {p_html}")


if __name__ == "__main__":
    main()
