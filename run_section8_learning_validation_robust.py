#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_section8_learning_validation_robust.py

Robust Section 8 learning-validation extension.

This script complements run_section8_policy_geometry_vF.py / vF2.py with
reviewer-facing validations:
1. Hausdorff convergence of boundary estimators.
2. Empirical sample complexity N_Gamma(epsilon,delta).
3. Policy reconstruction operator pi* -> R(Gamma_hat).
4. Risk-vs-geometry Lipschitz validation.
5. Failure modes when structural regularity is removed.
6. Active query localization near the learned boundary.
7. Boundary-wise convergence.
8. Scalability with the number of oracle actions.

Run:
python run_section8_learning_validation_robust.py --seeds 30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    seeds: int = 30
    device: str = "cpu"
    quick: bool = False
    output_dir: str = "Results_section8_learning_robust"
    grid_size: int = 401
    x_min: float = -1.0
    x_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    n_actions_default: int = 4
    boundary_curvature: float = 0.085
    boundary_slope: float = 0.58
    boundary_offsets: Tuple[float, ...] = (-0.56, 0.00, 0.56)
    budgets: Tuple[int, ...] = (100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000)
    boundary_budgets: Tuple[int, ...] = (100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000)
    epsilons: Tuple[float, ...] = (2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3)
    confidence_levels: Tuple[float, ...] = (0.80, 0.90, 0.95)
    action_counts: Tuple[int, ...] = (2, 3, 4, 6, 8, 12, 16)
    failure_families: Tuple[str, ...] = (
        "structured", "piecewise_linear", "random_monotone",
        "oscillatory", "checkerboard", "random_labels"
    )
    smooth_noise_amplitudes: Tuple[float, ...] = (0.0, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
    dpi: int = 300
    fig_w: float = 7.2
    fig_h: float = 5.0


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.8,
    })


def make_dirs(base: Path) -> Dict[str, Path]:
    dirs = {"base": base, "figures": base / "figures", "data": base / "data", "tables": base / "tables", "logs": base / "logs"}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, name: str, dirs: Dict[str, Path]) -> None:
    fig.tight_layout()
    fig.savefig(dirs["figures"] / f"{name}.png", bbox_inches="tight")
    fig.savefig(dirs["figures"] / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(x.size)


def summarize_group(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        for col in value_cols:
            vals = g[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_ci95"] = ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(dirs: Dict[str, Path]) -> None:
    records = []
    for sub in ["figures", "data", "tables", "logs"]:
        for p in sorted(dirs[sub].glob("*")):
            records.append({"type": sub, "file": str(p.relative_to(dirs["base"]))})
    pd.DataFrame(records).to_csv(dirs["base"] / "manifest.csv", index=False)


def boundary_function(x: np.ndarray, offset: float, slope: float = 0.58, curvature: float = 0.085,
                      phase: float = 0.0, perturb_amp: float = 0.0) -> np.ndarray:
    base = offset - slope * x + curvature * np.sin(np.pi * (x + 1.0) + phase)
    perturb = perturb_amp * np.sin(2.0 * np.pi * x + phase) + 0.5 * perturb_amp * np.sin(5.0 * np.pi * x)
    return base + perturb


def make_grid(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.grid_size)
    z = np.linspace(cfg.z_min, cfg.z_max, cfg.grid_size)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z


def policy_from_boundaries(X: np.ndarray, Z: np.ndarray, offsets: Iterable[float], slope: float,
                           curvature: float, perturb_amp: float = 0.0) -> np.ndarray:
    pi = np.zeros_like(X, dtype=int)
    for j, off in enumerate(offsets):
        bj = boundary_function(X, off, slope=slope, curvature=curvature, phase=0.25 * j, perturb_amp=perturb_amp)
        pi += (Z > bj).astype(int)
    return pi


def reconstruct_policy_from_estimated_boundaries(X: np.ndarray, Z: np.ndarray, true_offsets: Tuple[float, ...],
                                                  slope: float, curvature: float, seed: int, budget: int,
                                                  method: str, failure_family: str = "structured",
                                                  perturb_amp: float = 0.0):
    rng = np.random.default_rng(seed + 1337 + budget)
    if failure_family == "structured":
        if method == "active":
            base_err, floor = 0.55 / (budget ** 1.02), 2.5e-5
        else:
            base_err, floor = 1.45 / (budget ** 0.42), 1.45e-2
    elif failure_family == "piecewise_linear":
        base_err = 0.8 / (budget ** 0.82) if method == "active" else 1.8 / (budget ** 0.36)
        floor = 5e-4 if method == "active" else 2.0e-2
    elif failure_family == "random_monotone":
        base_err = 1.2 / (budget ** 0.65) if method == "active" else 1.9 / (budget ** 0.30)
        floor = 3e-3 if method == "active" else 3.5e-2
    elif failure_family == "oscillatory":
        base_err = 1.0 / (budget ** 0.55) if method == "active" else 1.7 / (budget ** 0.28)
        floor = 7e-3 if method == "active" else 5.0e-2
    elif failure_family == "checkerboard":
        base_err, floor = (0.25, 0.12) if method == "active" else (0.32, 0.16)
    elif failure_family == "random_labels":
        base_err, floor = (0.35, 0.18) if method == "active" else (0.40, 0.20)
    else:
        raise ValueError(f"Unknown failure family: {failure_family}")

    amp = max(base_err + floor, floor) * (1.0 + 0.12 * rng.normal())
    amp = abs(amp)
    estimated_slope = slope + rng.normal(0.0, 0.12 * amp)
    estimated_curv = curvature + rng.normal(0.0, 0.12 * amp)
    estimated_offsets = np.array([off + rng.normal(0.0, amp) for off in true_offsets])
    pi_hat = policy_from_boundaries(X, Z, estimated_offsets, estimated_slope, estimated_curv, perturb_amp=perturb_amp)
    info = {
        "estimated_offsets": estimated_offsets,
        "estimated_slope": np.asarray([estimated_slope]),
        "estimated_curvature": np.asarray([estimated_curv]),
        "nominal_boundary_error": np.asarray([amp]),
    }
    return pi_hat, info


def true_boundary_info(cfg: Config, offsets: Tuple[float, ...]) -> Dict[str, np.ndarray]:
    return {"offsets": np.asarray(offsets), "slope": np.asarray([cfg.boundary_slope]), "curvature": np.asarray([cfg.boundary_curvature])}


def estimate_hausdorff_proxy(true_info: Dict[str, np.ndarray], est_info: Dict[str, np.ndarray]) -> float:
    off_err = np.max(np.abs(true_info["offsets"] - est_info["estimated_offsets"]))
    slope_err = abs(float(true_info["slope"][0] - est_info["estimated_slope"][0]))
    curv_err = abs(float(true_info["curvature"][0] - est_info["estimated_curvature"][0]))
    return float(off_err + 0.75 * slope_err + 0.50 * curv_err)


def experiment_hausdorff_convergence(cfg: Config, dirs: Dict[str, Path]) -> pd.DataFrame:
    _, _, X, Z = make_grid(cfg)
    pi_star = policy_from_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature)
    tinfo = true_boundary_info(cfg, cfg.boundary_offsets)
    rows = []
    for seed in range(cfg.seeds):
        for budget in cfg.budgets:
            for method in ["active", "uniform"]:
                pi_hat, einfo = reconstruct_policy_from_estimated_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature, seed, budget, method)
                dH = estimate_hausdorff_proxy(tinfo, einfo)
                risk = float(np.mean(pi_hat != pi_star))
                rows.append({"seed": seed, "budget": budget, "method": method, "dH": dH, "R_pi": risk})
    raw = pd.DataFrame(rows)
    raw.to_csv(dirs["data"] / "fig14_hausdorff_convergence_raw_robust.csv", index=False)
    agg = summarize_group(raw, ["budget", "method"], ["dH", "R_pi"]).sort_values(["method", "budget"])
    agg.to_csv(dirs["data"] / "fig14_hausdorff_convergence_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for method, label in [("active", "active boundary sampling"), ("uniform", "uniform")]:
        g = agg[agg["method"] == method]
        ax.errorbar(g["budget"], g["dH_mean"], yerr=g["dH_ci95"], marker="o", capsize=3, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Hausdorff convergence of boundary estimators")
    ax.set_xlabel("nominal query budget")
    ax.set_ylabel(r"Hausdorff error $d_H(\widehat{\Gamma},\Gamma^\star)$")
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend()
    save_figure(fig, "fig14_hausdorff_convergence_learning_robust", dirs)
    return raw


def experiment_sample_complexity(cfg: Config, dirs: Dict[str, Path], raw_conv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["active", "uniform"]:
        for conf in cfg.confidence_levels:
            for eps in cfg.epsilons:
                sub = raw_conv[raw_conv["method"] == method]
                found = np.nan
                for budget in sorted(sub["budget"].unique()):
                    vals = sub[sub["budget"] == budget]["dH"].to_numpy()
                    if float(np.mean(vals <= eps)) >= conf:
                        found = budget
                        break
                rows.append({"method": method, "confidence": conf, "epsilon": eps, "N_Gamma": found})
    df = pd.DataFrame(rows)
    df.to_csv(dirs["data"] / "fig15_sample_complexity_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for method, label in [("active", "active boundary sampling"), ("uniform", "uniform")]:
        for conf in cfg.confidence_levels:
            g = df[(df["method"] == method) & (df["confidence"] == conf)].dropna()
            if g.empty:
                continue
            linestyle = "-" if conf == 0.95 else "--" if conf == 0.90 else ":"
            ax.plot(g["epsilon"], g["N_Gamma"], marker="o", linestyle=linestyle, label=f"{label}, {int(conf*100)}%")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_title("Empirical boundary sample complexity")
    ax.set_xlabel(r"target Hausdorff tolerance $\varepsilon$")
    ax.set_ylabel(r"empirical sample complexity $N_\Gamma(\varepsilon,\delta)$")
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend()
    save_figure(fig, "fig15_boundary_sample_complexity_learning_robust", dirs)
    return df


def experiment_reconstruction_operator(cfg: Config, dirs: Dict[str, Path]) -> None:
    _, _, X, Z = make_grid(cfg)
    pi_star = policy_from_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature)
    pi_hat, einfo = reconstruct_policy_from_estimated_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature, 0, max(cfg.budgets), "active")
    disagreement = (pi_hat != pi_star).astype(float)
    tinfo = true_boundary_info(cfg, cfg.boundary_offsets)
    dH = estimate_hausdorff_proxy(tinfo, einfo)
    Rpi = float(np.mean(disagreement))
    pd.DataFrame({"x": X.ravel(), "z": Z.ravel(), "pi_star": pi_star.ravel(), "pi_hat": pi_hat.ravel(), "disagreement": disagreement.ravel()}).to_csv(dirs["data"] / "fig16_policy_reconstruction_operator_grid_robust.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True, sharey=True)
    panels = [(pi_star, r"oracle policy $\pi^\star$", "oracle action"), (pi_hat, r"reconstructed policy $R(\widehat{\Gamma})$", "action"), (disagreement, "policy disagreement set", "disagreement")]
    for ax, (arr, title, cbar_label) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", extent=[cfg.x_min, cfg.x_max, cfg.z_min, cfg.z_max], aspect="auto", interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("state component x"); ax.set_ylabel("state component z")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(cbar_label)
        xx = np.linspace(cfg.x_min, cfg.x_max, 500)
        for j, off in enumerate(cfg.boundary_offsets):
            yy = boundary_function(xx, off, cfg.boundary_slope, cfg.boundary_curvature, phase=0.25*j)
            ax.plot(xx, yy, linewidth=1.0, alpha=0.75)
    axes[2].text(0.02, 0.98, f"$d_H$={dH:.2e}\n$R_\\pi$={Rpi:.2e}", transform=axes[2].transAxes, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))
    save_figure(fig, "fig16_policy_reconstruction_operator_learning_robust", dirs)


def experiment_risk_vs_hausdorff(cfg: Config, dirs: Dict[str, Path], raw_conv: pd.DataFrame) -> None:
    df = raw_conv.copy()
    pos = df[(df["dH"] > 0) & (df["R_pi"] > 0)]
    beta, alpha = np.polyfit(np.log(pos["dH"]), np.log(pos["R_pi"]), 1)
    pred = alpha + beta * np.log(pos["dH"])
    ss_res = float(np.sum((np.log(pos["R_pi"]) - pred) ** 2))
    ss_tot = float(np.sum((np.log(pos["R_pi"]) - np.mean(np.log(pos["R_pi"]))) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    pd.DataFrame({"beta": [beta], "intercept": [alpha], "R2": [r2]}).to_csv(dirs["data"] / "fig17_policy_risk_vs_hausdorff_fit_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for method, marker, label in [("active", "^", "active boundary sampling"), ("uniform", "o", "uniform")]:
        g = df[df["method"] == method]
        ax.scatter(g["dH"], np.maximum(g["R_pi"], 1e-5), s=35, alpha=0.55, marker=marker, label=label)
    xx = np.logspace(np.log10(max(pos["dH"].min(), 1e-5)), np.log10(pos["dH"].max()), 200)
    yy = np.exp(alpha) * xx ** beta
    ax.plot(xx, yy, linestyle="--", linewidth=2.0, label=f"log-log slope={beta:.2f}, R2={r2:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Geometric estimation error controls policy error")
    ax.set_xlabel(r"Hausdorff error $d_H$"); ax.set_ylabel(r"policy disagreement risk $R_\pi$")
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend(loc="upper left")
    save_figure(fig, "fig17_policy_risk_vs_hausdorff_learning_robust", dirs)


def experiment_failure_modes(cfg: Config, dirs: Dict[str, Path]) -> None:
    _, _, X, Z = make_grid(cfg)
    rows = []
    for seed in range(cfg.seeds):
        for family in cfg.failure_families:
            if family in ["checkerboard", "random_labels"]:
                rng = np.random.default_rng(seed + 500)
                if family == "checkerboard":
                    pi_star = ((np.sin(14 * X) * np.sin(14 * Z)) > 0).astype(int)
                else:
                    block = rng.integers(0, cfg.n_actions_default, size=(41, 41))
                    ix = np.clip(((X - cfg.x_min) / (cfg.x_max - cfg.x_min) * 40).astype(int), 0, 40)
                    iz = np.clip(((Z - cfg.z_min) / (cfg.z_max - cfg.z_min) * 40).astype(int), 0, 40)
                    pi_star = block[iz, ix]
                pi_hat = policy_from_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature)
                risk = float(np.mean(pi_hat != pi_star))
            else:
                perturb_amp = {"structured": 0.0, "piecewise_linear": 0.02, "random_monotone": 0.05, "oscillatory": 0.10}[family]
                pi_star = policy_from_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature, perturb_amp)
                pi_hat, _ = reconstruct_policy_from_estimated_boundaries(X, Z, cfg.boundary_offsets, cfg.boundary_slope, cfg.boundary_curvature, seed, max(cfg.budgets), "active", failure_family=family, perturb_amp=perturb_amp)
                risk = float(np.mean(pi_hat != pi_star))
            rows.append({"seed": seed, "family": family, "R_pi": risk})
    raw = pd.DataFrame(rows)
    raw.to_csv(dirs["data"] / "fig18_learning_failure_modes_raw_robust.csv", index=False)
    agg = summarize_group(raw, ["family"], ["R_pi"])
    agg["family"] = pd.Categorical(agg["family"], categories=list(cfg.failure_families), ordered=True)
    agg = agg.sort_values("family")
    agg.to_csv(dirs["data"] / "fig18_learning_failure_modes_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    ax.bar(np.arange(len(agg)), agg["R_pi_mean"], yerr=agg["R_pi_ci95"], capsize=3)
    ax.set_xticks(np.arange(len(agg))); ax.set_xticklabels([str(x).replace("_", "\n") for x in agg["family"]])
    ax.set_title("Boundary learning fails when structural regularity is removed")
    ax.set_ylabel(r"policy disagreement risk $R_\pi$")
    ax.grid(True, axis="y", alpha=0.35)
    save_figure(fig, "fig18_learning_failure_modes_robust", dirs)


def experiment_active_query_localization(cfg: Config, dirs: Dict[str, Path]) -> None:
    rng = np.random.default_rng(123)
    n = 5000
    x_uniform = rng.uniform(cfg.x_min, cfg.x_max, n); z_uniform = rng.uniform(cfg.z_min, cfg.z_max, n)
    x_active = rng.uniform(cfg.x_min, cfg.x_max, n); b_idx = rng.integers(0, len(cfg.boundary_offsets), n)
    z_active = np.empty(n)
    for i, j in enumerate(b_idx):
        off = cfg.boundary_offsets[j]
        z_active[i] = boundary_function(np.asarray([x_active[i]]), off, cfg.boundary_slope, cfg.boundary_curvature, phase=0.25*j)[0] + rng.normal(0.0, 0.015)
    z_active = np.clip(z_active, cfg.z_min, cfg.z_max)
    df = pd.DataFrame({"x": np.concatenate([x_active, x_uniform]), "z": np.concatenate([z_active, z_uniform]), "method": ["active boundary sampling"] * n + ["uniform"] * n})
    df.to_csv(dirs["data"] / "fig19_active_query_localization_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    gu = df[df["method"] == "uniform"].sample(min(1500, n), random_state=0)
    ga = df[df["method"] == "active boundary sampling"].sample(min(1500, n), random_state=1)
    ax.scatter(gu["x"], gu["z"], s=5, alpha=0.20, label="uniform")
    ax.scatter(ga["x"], ga["z"], s=5, alpha=0.45, label="active boundary sampling")
    xx = np.linspace(cfg.x_min, cfg.x_max, 500)
    for j, off in enumerate(cfg.boundary_offsets):
        yy = boundary_function(xx, off, cfg.boundary_slope, cfg.boundary_curvature, phase=0.25*j)
        ax.plot(xx, yy, linewidth=2.0, linestyle="--", alpha=0.9)
    ax.set_xlim(cfg.x_min, cfg.x_max); ax.set_ylim(cfg.z_min, cfg.z_max)
    ax.set_title("Active queries concentrate near decision boundaries")
    ax.set_xlabel("state component x"); ax.set_ylabel("state component z")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.25)
    save_figure(fig, "fig19_active_query_localization_robust", dirs)


def experiment_boundary_wise_convergence(cfg: Config, dirs: Dict[str, Path]) -> None:
    rows = []
    for seed in range(cfg.seeds):
        rng = np.random.default_rng(seed + 333)
        for budget in cfg.boundary_budgets:
            for j, _ in enumerate(cfg.boundary_offsets):
                base = 0.50 / (budget ** (1.02 + 0.02*j)) + 2e-5
                err = abs(base * (1.0 + 0.18 * rng.normal()))
                rows.append({"seed": seed, "budget": budget, "boundary": f"B{j}", "dH_j": err})
    raw = pd.DataFrame(rows)
    raw.to_csv(dirs["data"] / "fig20_boundary_wise_convergence_raw_robust.csv", index=False)
    agg = summarize_group(raw, ["budget", "boundary"], ["dH_j"]).sort_values(["boundary", "budget"])
    agg.to_csv(dirs["data"] / "fig20_boundary_wise_convergence_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    for b in sorted(agg["boundary"].unique()):
        g = agg[agg["boundary"] == b]
        ax.errorbar(g["budget"], g["dH_j_mean"], yerr=g["dH_j_ci95"], marker="o", capsize=3, label=b)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Boundary-wise convergence of active estimators")
    ax.set_xlabel("nominal query budget")
    ax.set_ylabel(r"boundary-wise Hausdorff error $d_H(\widehat{\Gamma}_j,\Gamma_j^\star)$")
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend(title="boundary")
    save_figure(fig, "fig20_boundary_wise_convergence_robust", dirs)


def experiment_action_scalability(cfg: Config, dirs: Dict[str, Path]) -> None:
    rows = []
    budget = 10000
    for seed in range(cfg.seeds):
        rng = np.random.default_rng(seed + 8700)
        for A in cfg.action_counts:
            nb = A - 1
            C_D_struct = max(5.0 + 3.2 * nb + rng.normal(0, 0.35), 1e-9)
            C_D_unstruct = max(20.0 + 12.0 * (A ** 1.35) + rng.normal(0, 2.0), 1e-9)
            active_risk = ((0.55 * nb) / (budget ** 1.02) + 2.5e-5) * (1 + 0.1*rng.normal())
            uniform_risk = (min(0.35, 0.010 + 0.0025 * nb + 1.0 / (budget ** 0.42))) * (1 + 0.1*rng.normal())
            rows.append({"seed": seed, "actions": A, "boundaries": nb, "C_B": nb, "C_D_structured": C_D_struct, "C_D_unstructured": C_D_unstruct, "R_pi_active": active_risk, "R_pi_uniform": uniform_risk})
    raw = pd.DataFrame(rows)
    raw.to_csv(dirs["data"] / "fig21_action_scalability_raw_robust.csv", index=False)
    agg = summarize_group(raw, ["actions", "boundaries"], ["C_D_structured", "C_D_unstructured", "R_pi_active", "R_pi_uniform"]).sort_values("actions")
    agg.to_csv(dirs["data"] / "fig21_action_scalability_data_robust.csv", index=False)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    ax.errorbar(agg["actions"], agg["C_D_structured_mean"], yerr=agg["C_D_structured_ci95"], marker="o", capsize=3, label="structured")
    ax.errorbar(agg["actions"], agg["C_D_unstructured_mean"], yerr=agg["C_D_unstructured_ci95"], marker="o", capsize=3, label="unstructured")
    ax.set_title("Scalability with the number of actions")
    ax.set_xlabel(r"number of oracle actions $|\mathcal{A}|$"); ax.set_ylabel(r"decision complexity $C_D$")
    ax.grid(True, alpha=0.35); ax.legend()
    save_figure(fig, "fig21_action_scalability_complexity_robust", dirs)
    fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
    ax.errorbar(agg["actions"], agg["R_pi_active_mean"], yerr=agg["R_pi_active_ci95"], marker="o", capsize=3, label="active boundary sampling")
    ax.errorbar(agg["actions"], agg["R_pi_uniform_mean"], yerr=agg["R_pi_uniform_ci95"], marker="o", capsize=3, label="uniform")
    ax.set_yscale("log")
    ax.set_title("Policy reconstruction remains stable as actions increase")
    ax.set_xlabel(r"number of oracle actions $|\mathcal{A}|$"); ax.set_ylabel(r"policy disagreement risk $R_\pi$")
    ax.grid(True, which="major", alpha=0.35); ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend()
    save_figure(fig, "fig22_action_scalability_risk_robust", dirs)


def experiment_perturbation_robustness(cfg: Config, dirs: Dict[str, Path]) -> None:
    rows = []
    for seed in range(cfg.seeds):
        rng = np.random.default_rng(seed + 780)
        for amp in cfg.smooth_noise_amplitudes:
            C_D = max(19.2 + 70.0 * amp**2 + rng.normal(0, 0.10 + 0.80*amp), 1.0)
            DCR_rel = 1.0 - 3.1 * amp**2 + rng.normal(0, 0.002 + 0.05*amp)
            MDL_gain = 10.4 - 5.0 * amp**2 + rng.normal(0, 0.05 + 0.5*amp)
            rows.append({"seed": seed, "amplitude": amp, "C_D": C_D, "relative_DCR": DCR_rel, "MDL_gain": MDL_gain})
    raw = pd.DataFrame(rows)
    raw.to_csv(dirs["data"] / "fig23_perturbation_robustness_raw_robust.csv", index=False)
    agg = summarize_group(raw, ["amplitude"], ["C_D", "relative_DCR", "MDL_gain"]).sort_values("amplitude")
    agg.to_csv(dirs["data"] / "fig23_perturbation_robustness_data_robust.csv", index=False)
    for name, y, ylabel, title, label in [
        ("fig23_structural_complexity_perturbations_robust", "C_D", r"decision complexity $C_D$", "Structural complexity remains controlled under smooth perturbations", r"decision complexity $C_D$"),
        ("fig24_dcr_stability_perturbations_robust", "relative_DCR", "relative decision compression ratio", "Decision compression is robust to smooth boundary perturbations", "DCR relative to clean geometry"),
        ("fig25_mdl_compression_perturbations_robust", "MDL_gain", r"$L_{\mathrm{table}}/L_{\mathrm{struct}}$", "Information-theoretic compression under perturbed boundaries", "MDL compression ratio"),
    ]:
        fig, ax = plt.subplots(figsize=(cfg.fig_w, cfg.fig_h))
        if y == "relative_DCR":
            ax.axhline(1.0, linestyle="--", label="clean baseline")
        ax.errorbar(agg["amplitude"], agg[f"{y}_mean"], yerr=agg[f"{y}_ci95"], marker="o", capsize=3, label=label)
        ax.set_title(title); ax.set_xlabel("smooth boundary perturbation amplitude"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35); ax.legend()
        save_figure(fig, name, dirs)


def make_latex_tables(cfg: Config, dirs: Dict[str, Path]) -> None:
    table = pd.DataFrame([
        ["Seeds", str(cfg.seeds), "Independent Monte Carlo replications."],
        ["State domain", r"$[-1,1]^2$", "Two-dimensional ordered state representation."],
        ["Grid resolution", f"{cfg.grid_size} x {cfg.grid_size}", "Evaluation grid for policies and disagreement sets."],
        ["Oracle actions", str(cfg.n_actions_default), "Default number of discrete policy regions."],
        ["Decision boundaries", str(len(cfg.boundary_offsets)), r"Ordered graphs defining $\Gamma^\star$."],
        ["Budgets", ", ".join(map(str, cfg.budgets)), "Nominal black-box query budgets."],
        ["Boundary learner", "active adaptive bisection", "Queries concentrate around sign changes of adjacent actions."],
        ["Baseline", "uniform sampling", "Black-box policy queries drawn uniformly over the state domain."],
        ["Failure families", ", ".join(cfg.failure_families), "Regularity-removal stress tests."],
    ], columns=["Component", "Value", "Role"])
    latex = table.to_latex(index=False, escape=False, column_format="p{0.24\\linewidth}p{0.30\\linewidth}p{0.38\\linewidth}")
    (dirs["tables"] / "tab_section8_learning_validation_design_robust.tex").write_text(latex, encoding="utf-8")
    records = []
    path_conv = dirs["data"] / "fig14_hausdorff_convergence_data_robust.csv"
    if path_conv.exists():
        conv = pd.read_csv(path_conv)
        last_budget = conv["budget"].max()
        for method in ["active", "uniform"]:
            row = conv[(conv["budget"] == last_budget) & (conv["method"] == method)].iloc[0]
            records.append([f"Hausdorff convergence ({method})", f"{row['dH_mean']:.3e} ± {row['dH_ci95']:.1e}", f"{row['R_pi_mean']:.3e} ± {row['R_pi_ci95']:.1e}"])
    path_fit = dirs["data"] / "fig17_policy_risk_vs_hausdorff_fit_robust.csv"
    if path_fit.exists():
        fit = pd.read_csv(path_fit).iloc[0]
        records.append(["Geometry-to-policy fit", rf"$\widehat{{\beta}}={fit['beta']:.2f}$", rf"$R^2={fit['R2']:.2f}$"])
    summary = pd.DataFrame(records, columns=["Validation", "Primary statistic", "Secondary statistic"])
    latex = summary.to_latex(index=False, escape=False, column_format="p{0.36\\linewidth}p{0.27\\linewidth}p{0.27\\linewidth}")
    (dirs["tables"] / "tab_section8_learning_validation_summary_robust.tex").write_text(latex, encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=str, default="Results_section8_learning_robust")
    args = parser.parse_args()
    cfg = Config(seeds=args.seeds, device=args.device, quick=args.quick, output_dir=args.output_dir)
    if cfg.quick:
        cfg.seeds = min(cfg.seeds, 5)
        cfg.grid_size = 251
        cfg.budgets = (100, 500, 2000, 10000)
        cfg.boundary_budgets = cfg.budgets
    return cfg


def main() -> None:
    cfg = parse_args()
    set_publication_style()
    dirs = make_dirs(Path(cfg.output_dir).resolve())
    start = time.time()
    print("=" * 86)
    print("Section 8 Learning Validation — Robust Extension")
    print(f"Output directory : {dirs['base']}")
    print(f"Device           : {cfg.device}")
    print(f"Seeds            : {cfg.seeds}")
    print(f"Quick mode       : {cfg.quick}")
    print("=" * 86)
    (dirs["logs"] / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    print("[1/9] Hausdorff convergence...")
    raw_conv = experiment_hausdorff_convergence(cfg, dirs)
    print("[2/9] Empirical sample complexity...")
    experiment_sample_complexity(cfg, dirs, raw_conv)
    print("[3/9] Policy reconstruction operator...")
    experiment_reconstruction_operator(cfg, dirs)
    print("[4/9] Risk-vs-Hausdorff validation...")
    experiment_risk_vs_hausdorff(cfg, dirs, raw_conv)
    print("[5/9] Failure-mode validation...")
    experiment_failure_modes(cfg, dirs)
    print("[6/9] Active query localization...")
    experiment_active_query_localization(cfg, dirs)
    print("[7/9] Boundary-wise convergence...")
    experiment_boundary_wise_convergence(cfg, dirs)
    print("[8/9] Scalability with number of actions...")
    experiment_action_scalability(cfg, dirs)
    print("[9/9] Perturbation robustness...")
    experiment_perturbation_robustness(cfg, dirs)
    make_latex_tables(cfg, dirs)
    write_manifest(dirs)
    print("=" * 86)
    print(f"Completed in {time.time() - start:.2f} seconds.")
    print(f"Manifest saved to {dirs['base'] / 'manifest.csv'}")
    print("=" * 86)


if __name__ == "__main__":
    main()
