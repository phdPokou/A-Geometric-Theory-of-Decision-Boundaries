#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 8 Numerical Validation — VF Final.

Final black-box numerical validation script for the policy-geometry paper.

This version fixes the remaining methodological issue in V4:
    the random seed used to sample query locations is now separated from the
    random seed defining the black-box policy oracle. This is crucial for
    random-structured families: the learner may randomize its query design, but
    the target policy itself must remain fixed across all strategies.

Core protocol:
    The learner observes only black-box policy labels

        (x,z) -> pi*(x,z),

    and never observes Q*, action gaps, thresholds, scores, or true boundaries.

Active method:
    label-only monotone bracketing + bisection along adaptively refined
    one-dimensional sections.

Expected outputs:
    - active boundary estimation dominates uniform sampling in finite samples;
    - d_H(Gamma_hat, Gamma*) decreases with the query budget;
    - policy risk decreases with geometric error;
    - sample-complexity tables are informative;
    - reconstruction fails when structural assumptions are removed.

Run:
    python run_section8_policy_geometry_vF.py --seeds 30 --device cuda

Quick test:
    python run_section8_policy_geometry_vF.py --quick --device cuda

Outputs:
    Results_section8_vF/
        data/
        figures/
        tables/
        logs/
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


STRUCTURED_FAMILIES = (
    "inventory_threshold",
    "machine_replacement",
    "admission_control",
    "resource_allocation",
    "random_structured",
)

UNSTRUCTURED_FAMILIES = (
    "oscillatory_fragmented",
)


@dataclass(frozen=True)
class RunConfig:
    outdir: Path
    seeds: int
    quick: bool
    device: str
    dtype: torch.dtype
    grid_2d: int
    actions: int
    sample_values: Tuple[int, ...]
    eps_grid: Tuple[float, ...]
    beta_z: float
    z_bins: int
    eval_size: int
    max_bisection_steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dirs(outdir: Path) -> Dict[str, Path]:
    dirs = {
        "root": outdir,
        "data": outdir / "data",
        "figures": outdir / "figures",
        "tables": outdir / "tables",
        "logs": outdir / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_table(df: pd.DataFrame, stem: str, dirs: Dict[str, Path]) -> None:
    save_df(df, dirs["tables"] / f"{stem}.csv")
    with open(dirs["tables"] / f"{stem}.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}"))


def save_figure(fig: plt.Figure, stem: str, dirs: Dict[str, Path]) -> None:
    fig.tight_layout()
    fig.savefig(dirs["figures"] / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(dirs["figures"] / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ci95(x: pd.Series) -> float:
    arr = x.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))


def ci_summary(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_sd"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(sub[metric])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Black-box policy oracle
# ---------------------------------------------------------------------

def thresholds(num_actions: int) -> np.ndarray:
    return np.linspace(-0.60, 0.60, num_actions - 1)


def score_numpy(x: np.ndarray, z: np.ndarray, family: str, oracle_seed: int, beta_z: float) -> np.ndarray:
    if family == "inventory_threshold":
        return x + beta_z * z

    if family == "machine_replacement":
        return 1.15 * x + 0.15 * z

    if family == "admission_control":
        return 0.90 * x + 0.45 * np.maximum(z, 0.0)

    if family == "resource_allocation":
        return x + 0.30 * z + 0.10 * np.tanh(2.0 * z)

    if family == "random_structured":
        rng = np.random.default_rng(oracle_seed + 7000)
        w0 = 0.65 + 0.20 * rng.random()
        w1 = 1.0 - w0
        return w0 * x + w1 * z

    if family == "oscillatory_fragmented":
        return (
            x
            + 0.25 * z
            + 0.35 * np.sin(4.0 * math.pi * x)
            + 0.30 * np.cos(5.0 * math.pi * z)
        )

    raise ValueError(f"Unknown family: {family}")


def oracle_policy(
    x: np.ndarray,
    z: np.ndarray,
    num_actions: int,
    family: str,
    oracle_seed: int,
    beta_z: float,
) -> np.ndarray:
    sc = score_numpy(x, z, family, oracle_seed, beta_z)
    return np.digitize(sc, thresholds(num_actions)).astype(int)


def true_boundary_curves(
    z_grid: np.ndarray,
    num_actions: int,
    family: str,
    oracle_seed: int,
    beta_z: float,
) -> np.ndarray:
    """
    True boundary curves, used only for evaluation.
    The learner never observes these curves.
    """
    th = thresholds(num_actions)

    if family == "inventory_threshold":
        out = th[None, :] - beta_z * z_grid[:, None]

    elif family == "machine_replacement":
        out = (th[None, :] - 0.15 * z_grid[:, None]) / 1.15

    elif family == "admission_control":
        out = (th[None, :] - 0.45 * np.maximum(z_grid[:, None], 0.0)) / 0.90

    elif family == "resource_allocation":
        out = th[None, :] - 0.30 * z_grid[:, None] - 0.10 * np.tanh(2.0 * z_grid[:, None])

    elif family == "random_structured":
        rng = np.random.default_rng(oracle_seed + 7000)
        w0 = 0.65 + 0.20 * rng.random()
        w1 = 1.0 - w0
        out = (th[None, :] - w1 * z_grid[:, None]) / w0

    else:
        return np.full((len(z_grid), num_actions - 1), np.nan)

    return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------------
# Label-only estimators
# ---------------------------------------------------------------------

def uniform_query(
    n: int,
    num_actions: int,
    family: str,
    oracle_seed: int,
    query_seed: int,
    beta_z: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(query_seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = oracle_policy(x, z, num_actions, family, oracle_seed, beta_z)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def estimate_boundaries_from_labels(
    data: pd.DataFrame,
    z_grid: np.ndarray,
    num_actions: int,
    degree: int = 2,
) -> np.ndarray:
    z = data["z"].to_numpy()
    x = data["x"].to_numpy()
    y = data["y"].to_numpy().astype(int)

    nb = len(z_grid)
    bin_edges = np.linspace(-1.0, 1.0, nb + 1)
    raw = np.full((nb, num_actions - 1), np.nan)

    for k in range(nb):
        idx = (z >= bin_edges[k]) & (z < bin_edges[k + 1])
        if idx.sum() < 4:
            continue

        xk = x[idx]
        yk = y[idx]

        for a in range(num_actions - 1):
            lower = xk[yk <= a]
            upper = xk[yk >= a + 1]
            if len(lower) > 0 and len(upper) > 0:
                raw[k, a] = 0.5 * (np.max(lower) + np.min(upper))

    est = np.full_like(raw, np.nan)
    for a in range(num_actions - 1):
        ok = np.isfinite(raw[:, a])
        if ok.sum() >= degree + 1:
            deg = min(degree, ok.sum() - 1)
            coef = np.polyfit(z_grid[ok], raw[ok, a], deg=deg)
            est[:, a] = np.polyval(coef, z_grid)
        elif ok.sum() >= 2:
            coef = np.polyfit(z_grid[ok], raw[ok, a], deg=1)
            est[:, a] = np.polyval(coef, z_grid)
        else:
            est[:, a] = np.linspace(-0.5, 0.5, num_actions - 1)[a]

    return np.clip(est, -1.0, 1.0)


def locate_boundary_by_bisection(
    z_value: float,
    boundary_index: int,
    num_actions: int,
    family: str,
    oracle_seed: int,
    beta_z: float,
    steps: int,
) -> Tuple[float, int]:
    queries = 0
    a = boundary_index

    left = -1.0
    right = 1.0

    y_left = int(oracle_policy(np.array([left]), np.array([z_value]), num_actions, family, oracle_seed, beta_z)[0])
    y_right = int(oracle_policy(np.array([right]), np.array([z_value]), num_actions, family, oracle_seed, beta_z)[0])
    queries += 2

    if y_left >= a + 1:
        return -1.0, queries
    if y_right <= a:
        return 1.0, queries

    for _ in range(steps):
        mid = 0.5 * (left + right)
        y_mid = int(oracle_policy(np.array([mid]), np.array([z_value]), num_actions, family, oracle_seed, beta_z)[0])
        queries += 1

        if y_mid <= a:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right), queries


def active_adaptive_bisection_estimator(
    budget: int,
    num_actions: int,
    family: str,
    oracle_seed: int,
    query_seed: int,
    beta_z: float,
    z_grid: np.ndarray,
    max_steps: int,
) -> Tuple[np.ndarray, int, pd.DataFrame]:
    """
    Pure black-box active estimator using action-label bracketing.

    Improvements over V4:
        - oracle_seed and query_seed are separated;
        - the number of z-sections grows with budget;
        - interpolation is piecewise-linear on the localized sections;
        - optional curvature refinement adds sections where the preliminary
          boundary estimate bends most strongly.

    The learner only receives labels pi*(x,z).
    """
    rng = np.random.default_rng(query_seed)
    num_boundaries = num_actions - 1

    # Bisection depth grows slowly with budget.
    steps = int(max(5, min(max_steps, math.floor(math.log2(max(budget, 32))) - 2)))
    cost_pair = steps + 2

    # Reserve budget for all boundaries at each z-section.
    max_sections = max(4, budget // max(num_boundaries * cost_pair, 1))
    base_sections = min(len(z_grid), max_sections)

    # Preliminary quasi-uniform sections.
    z_probe = np.linspace(-1.0, 1.0, base_sections)
    if base_sections > 4:
        jitter = rng.normal(0.0, 0.15 / base_sections, size=base_sections)
        z_probe = np.clip(z_probe + jitter, -1.0, 1.0)
        z_probe[0], z_probe[-1] = -1.0, 1.0
        z_probe = np.unique(np.sort(z_probe))

    query_count = 0
    trace_rows = []

    def evaluate_sections(sections: np.ndarray) -> Tuple[np.ndarray, int, List[dict]]:
        raw_local = np.zeros((len(sections), num_boundaries))
        q_total = 0
        rows = []
        for i, z0 in enumerate(sections):
            for a in range(num_boundaries):
                bx, qn = locate_boundary_by_bisection(
                    z_value=float(z0),
                    boundary_index=a,
                    num_actions=num_actions,
                    family=family,
                    oracle_seed=oracle_seed,
                    beta_z=beta_z,
                    steps=steps,
                )
                raw_local[i, a] = bx
                q_total += qn
                rows.append({
                    "z": float(z0),
                    "boundary": a,
                    "estimated_x": float(bx),
                    "bisection_steps": steps,
                    "queries": qn,
                })
        return raw_local, q_total, rows

    raw, q_used, rows = evaluate_sections(z_probe)
    query_count += q_used
    trace_rows.extend(rows)

    # Curvature refinement if remaining budget exists.
    remaining = budget - query_count
    if remaining >= num_boundaries * cost_pair * 4 and len(z_probe) >= 5:
        # Preliminary interpolation.
        est_pre = np.zeros((len(z_grid), num_boundaries))
        for a in range(num_boundaries):
            est_pre[:, a] = np.interp(z_grid, z_probe, raw[:, a])

        curvature = np.zeros(len(z_grid))
        for a in range(num_boundaries):
            second = np.abs(np.gradient(np.gradient(est_pre[:, a], z_grid), z_grid))
            curvature += second

        # Pick additional z sections in high-curvature regions.
        add_sections = min(len(z_grid), remaining // max(num_boundaries * cost_pair, 1))
        if add_sections > 0:
            prob = curvature + 1e-6
            prob = prob / prob.sum()
            candidates = rng.choice(z_grid, size=int(add_sections), replace=False, p=prob)
            z_probe2 = np.unique(np.sort(np.concatenate([z_probe, candidates])))

            raw2, q_used2, rows2 = evaluate_sections(z_probe2)
            if q_used2 <= budget:
                z_probe = z_probe2
                raw = raw2
                query_count = q_used2
                trace_rows = rows2

    # Final piecewise-linear reconstruction on full z-grid.
    est = np.zeros((len(z_grid), num_boundaries))
    for a in range(num_boundaries):
        est[:, a] = np.interp(z_grid, z_probe, raw[:, a])

    est = np.clip(est, -1.0, 1.0)
    trace = pd.DataFrame(trace_rows)
    trace["nominal_budget"] = budget
    trace["actual_queries_total"] = query_count
    return est, query_count, trace


def hausdorff_curve_error(est: np.ndarray, true: np.ndarray) -> float:
    return float(np.nanmax(np.abs(est - true)))


def policy_risk_from_estimate(
    est: np.ndarray,
    num_actions: int,
    family: str,
    oracle_seed: int,
    beta_z: float,
    n_eval: int,
) -> float:
    rng = np.random.default_rng(oracle_seed + 98765)
    x = rng.uniform(-1.0, 1.0, size=n_eval)
    z = rng.uniform(-1.0, 1.0, size=n_eval)

    y_true = oracle_policy(x, z, num_actions, family, oracle_seed, beta_z)

    z_grid = np.linspace(-1.0, 1.0, est.shape[0])
    b = np.column_stack([np.interp(z, z_grid, est[:, a]) for a in range(num_actions - 1)])
    y_hat = np.sum(x[:, None] >= b, axis=1).astype(int)

    return float(np.mean(y_hat != y_true))


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_blackbox_tessellation(cfg: RunConfig, dirs: Dict[str, Path]) -> None:
    m = cfg.grid_2d
    xs = np.linspace(-1.0, 1.0, m)
    zs = np.linspace(-1.0, 1.0, m)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")

    yy = oracle_policy(
        xx.reshape(-1),
        zz.reshape(-1),
        cfg.actions,
        "inventory_threshold",
        0,
        cfg.beta_z,
    ).reshape(m, m)

    save_df(
        pd.DataFrame({"x": xx.reshape(-1), "z": zz.reshape(-1), "policy": yy.reshape(-1)}),
        dirs["data"] / "fig01_blackbox_tessellation_data_vF.csv",
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(yy.T, origin="lower", extent=[-1, 1, -1, 1], aspect="auto")
    ax.set_xlabel("ordered state component $x$")
    ax.set_ylabel("auxiliary state component $z$")
    ax.set_title(r"Black-box optimal policy tessellation")
    fig.colorbar(im, ax=ax, label="oracle action")
    save_figure(fig, "fig01_blackbox_policy_tessellation_vF", dirs)

    boundary = np.zeros_like(yy, dtype=float)
    boundary[:-1, :] = np.maximum(boundary[:-1, :], yy[:-1, :] != yy[1:, :])
    boundary[:, :-1] = np.maximum(boundary[:, :-1], yy[:, :-1] != yy[:, 1:])

    save_df(
        pd.DataFrame({"x": xx.reshape(-1), "z": zz.reshape(-1), "boundary": boundary.reshape(-1)}),
        dirs["data"] / "fig02_blackbox_boundary_data_vF.csv",
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(boundary.T, origin="lower", extent=[-1, 1, -1, 1], aspect="auto")
    ax.set_xlabel("ordered state component $x$")
    ax.set_ylabel("auxiliary state component $z$")
    ax.set_title(r"Target boundary geometry $\Gamma^*$")
    fig.colorbar(im, ax=ax, label="boundary indicator")
    save_figure(fig, "fig02_blackbox_boundary_geometry_vF", dirs)


def plot_reconstruction_example(cfg: RunConfig, dirs: Dict[str, Path]) -> None:
    family = "resource_allocation"
    oracle_seed = 0
    query_seed = 123456
    budget = max(cfg.sample_values)
    z_grid = np.linspace(-1.0, 1.0, cfg.z_bins)

    true = true_boundary_curves(z_grid, cfg.actions, family, oracle_seed, cfg.beta_z)
    est, query_count, trace = active_adaptive_bisection_estimator(
        budget=budget,
        num_actions=cfg.actions,
        family=family,
        oracle_seed=oracle_seed,
        query_seed=query_seed,
        beta_z=cfg.beta_z,
        z_grid=z_grid,
        max_steps=cfg.max_bisection_steps,
    )

    save_df(trace, dirs["data"] / "fig08_active_adaptive_trace_vF.csv")

    rows = []
    for a in range(cfg.actions - 1):
        for i, z in enumerate(z_grid):
            rows.append({
                "boundary": a,
                "z": z,
                "true_x": true[i, a],
                "estimated_x": est[i, a],
                "queries": query_count,
            })
    save_df(pd.DataFrame(rows), dirs["data"] / "fig08_reconstructed_boundaries_data_vF.csv")

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for a in range(cfg.actions - 1):
        ax.plot(true[:, a], z_grid, linestyle="-", label=f"true {a}/{a+1}")
        ax.plot(est[:, a], z_grid, linestyle="--", label=f"estimated {a}/{a+1}")
    ax.set_xlabel("state component $x$")
    ax.set_ylabel("state component $z$")
    ax.set_title(r"Black-box boundary reconstruction")
    ax.legend(fontsize=8)
    save_figure(fig, "fig08_blackbox_boundary_reconstruction_vF", dirs)


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

def experiment_blackbox_learning(cfg: RunConfig, dirs: Dict[str, Path]) -> pd.DataFrame:
    records = []
    active_traces = []
    z_grid = np.linspace(-1.0, 1.0, cfg.z_bins)

    for oracle_seed in range(cfg.seeds):
        set_seed(oracle_seed)

        for family in STRUCTURED_FAMILIES:
            true = true_boundary_curves(z_grid, cfg.actions, family, oracle_seed, cfg.beta_z)

            for budget in cfg.sample_values:
                # Uniform baseline: randomized query design, fixed target oracle.
                u_data = uniform_query(
                    n=budget,
                    num_actions=cfg.actions,
                    family=family,
                    oracle_seed=oracle_seed,
                    query_seed=oracle_seed + 10000,
                    beta_z=cfg.beta_z,
                )
                est_u = estimate_boundaries_from_labels(u_data, z_grid, cfg.actions, degree=2)
                d_u = hausdorff_curve_error(est_u, true)
                r_u = policy_risk_from_estimate(
                    est_u,
                    cfg.actions,
                    family,
                    oracle_seed,
                    cfg.beta_z,
                    n_eval=cfg.eval_size,
                )
                records.append({
                    "seed": oracle_seed,
                    "family": family,
                    "strategy": "uniform",
                    "samples": budget,
                    "queries": budget,
                    "d_H": d_u,
                    "policy_risk": r_u,
                })

                # Active adaptive bisection: randomized design, fixed target oracle.
                est_a, q_count, trace = active_adaptive_bisection_estimator(
                    budget=budget,
                    num_actions=cfg.actions,
                    family=family,
                    oracle_seed=oracle_seed,
                    query_seed=oracle_seed + 20000,
                    beta_z=cfg.beta_z,
                    z_grid=z_grid,
                    max_steps=cfg.max_bisection_steps,
                )
                d_a = hausdorff_curve_error(est_a, true)
                r_a = policy_risk_from_estimate(
                    est_a,
                    cfg.actions,
                    family,
                    oracle_seed,
                    cfg.beta_z,
                    n_eval=cfg.eval_size,
                )
                records.append({
                    "seed": oracle_seed,
                    "family": family,
                    "strategy": "active_adaptive_bisection",
                    "samples": budget,
                    "queries": q_count,
                    "d_H": d_a,
                    "policy_risk": r_a,
                })

                if oracle_seed < 2 and budget in (min(cfg.sample_values), max(cfg.sample_values)):
                    trace = trace.copy()
                    trace["seed"] = oracle_seed
                    trace["family"] = family
                    trace["samples"] = budget
                    active_traces.append(trace)

    df = pd.DataFrame(records)
    save_df(df, dirs["data"] / "blackbox_learning_all_results_vF.csv")

    if active_traces:
        save_df(pd.concat(active_traces, ignore_index=True), dirs["data"] / "active_adaptive_trace_samples_vF.csv")

    summary = ci_summary(df, ["strategy", "samples"], ["d_H", "policy_risk", "queries"])
    save_df(summary, dirs["data"] / "fig03_04_blackbox_learning_summary_vF.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for strategy, sub in summary.groupby("strategy"):
        ax.errorbar(sub["samples"], sub["d_H_mean"], yerr=sub["d_H_ci95"], marker="o", label=strategy)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal query budget")
    ax.set_ylabel(r"Hausdorff error $d_H(\widehat{\Gamma},\Gamma^*)$")
    ax.set_title(r"Black-box boundary estimation")
    ax.legend()
    save_figure(fig, "fig03_blackbox_hausdorff_vs_queries_vF", dirs)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for strategy, sub in summary.groupby("strategy"):
        ax.errorbar(sub["samples"], sub["policy_risk_mean"], yerr=sub["policy_risk_ci95"], marker="o", label=strategy)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal query budget")
    ax.set_ylabel(r"policy disagreement risk $\mathcal{R}_{\pi}$")
    ax.set_title(r"Policy reconstruction from black-box queries")
    ax.legend()
    save_figure(fig, "fig04_blackbox_policy_risk_vs_queries_vF", dirs)

    pivot = summary.pivot(index="samples", columns="strategy", values="policy_risk_mean").reset_index()
    if {"uniform", "active_adaptive_bisection"}.issubset(set(pivot.columns)):
        pivot["risk_reduction_factor"] = pivot["uniform"] / pivot["active_adaptive_bisection"].clip(lower=1e-12)
        save_df(pivot, dirs["data"] / "fig05_active_efficiency_gain_data_vF.csv")

        fig, ax = plt.subplots(figsize=(6.3, 4.6))
        ax.plot(pivot["samples"], pivot["risk_reduction_factor"], marker="o")
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("nominal query budget")
        ax.set_ylabel("uniform risk / active risk")
        ax.set_title(r"Active boundary sampling efficiency gain")
        save_figure(fig, "fig05_active_sampling_efficiency_gain_vF", dirs)

    save_df(df, dirs["data"] / "fig06_policy_risk_vs_dh_blackbox_data_vF.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for strategy, sub in df.groupby("strategy"):
        ax.scatter(sub["d_H"], sub["policy_risk"], alpha=0.35, label=strategy)
    ax.set_xlabel(r"Hausdorff error $d_H$")
    ax.set_ylabel(r"policy disagreement risk $\mathcal{R}_{\pi}$")
    ax.set_title(r"Geometric estimation error controls policy error")
    ax.legend()
    save_figure(fig, "fig06_blackbox_policy_risk_vs_geometric_error_vF", dirs)

    save_table(ci_summary(df, ["family", "strategy", "samples"], ["d_H", "policy_risk", "queries"]),
               "table02_blackbox_learning_by_family_vF", dirs)

    rows = []
    for strategy in sorted(df["strategy"].unique()):
        sub = df[df["strategy"] == strategy]
        for eps in cfg.eps_grid:
            for success_level in (0.80, 0.90):
                candidates = []
                for n, block in sub.groupby("samples"):
                    success = float(np.mean(block["d_H"] <= eps))
                    if success >= success_level:
                        candidates.append(n)
                rows.append({
                    "strategy": strategy,
                    "epsilon": eps,
                    "success_level": success_level,
                    "empirical_N_Gamma": min(candidates) if candidates else np.nan,
                    "seeds": cfg.seeds,
                    "families": len(STRUCTURED_FAMILIES),
                })
    save_table(pd.DataFrame(rows), "table03_empirical_boundary_sample_complexity_vF", dirs)
    save_table(summary, "table01_blackbox_learning_summary_vF", dirs)

    return df


def experiment_failure_modes(cfg: RunConfig, dirs: Dict[str, Path]) -> pd.DataFrame:
    records = []
    z_grid = np.linspace(-1.0, 1.0, cfg.z_bins)
    budget = max(cfg.sample_values)

    for oracle_seed in range(cfg.seeds):
        for family in list(STRUCTURED_FAMILIES) + list(UNSTRUCTURED_FAMILIES):
            est, q_count, _ = active_adaptive_bisection_estimator(
                budget=budget,
                num_actions=cfg.actions,
                family=family,
                oracle_seed=oracle_seed,
                query_seed=oracle_seed + 30000,
                beta_z=cfg.beta_z,
                z_grid=z_grid,
                max_steps=cfg.max_bisection_steps,
            )
            risk = policy_risk_from_estimate(est, cfg.actions, family, oracle_seed, cfg.beta_z, n_eval=cfg.eval_size)
            records.append({
                "seed": oracle_seed,
                "family": family,
                "structure": "structured" if family in STRUCTURED_FAMILIES else "unstructured",
                "samples": budget,
                "queries": q_count,
                "policy_risk": risk,
            })

    df = pd.DataFrame(records)
    save_df(df, dirs["data"] / "blackbox_failure_modes_data_vF.csv")

    summary = ci_summary(df, ["structure"], ["policy_risk", "queries"])
    save_table(summary, "table04_blackbox_failure_modes_vF", dirs)

    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.bar(summary["structure"], summary["policy_risk_mean"], yerr=summary["policy_risk_ci95"])
    ax.set_ylabel(r"policy disagreement risk $\mathcal{R}_{\pi}$")
    ax.set_title(r"Black-box reconstruction fails when structure is removed")
    save_figure(fig, "fig07_blackbox_failure_modes_vF", dirs)

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 8 VF final black-box boundary learning experiments.")
    parser.add_argument("--outdir", type=str, default="Results_section8_vF")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--grid-2d", type=int, default=240)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    if args.quick:
        seeds = min(args.seeds, 3)
        sample_values = (100, 250, 500, 1000, 2500)
        eps_grid = (0.20, 0.10, 0.05, 0.025)
        z_bins = 100
        eval_size = 15000
        max_steps = 14
        grid_2d = min(args.grid_2d, 120)
    else:
        seeds = args.seeds
        sample_values = (100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000)
        eps_grid = (0.20, 0.10, 0.05, 0.025, 0.01, 0.005)
        z_bins = 240
        eval_size = 100000
        max_steps = 22
        grid_2d = args.grid_2d

    return RunConfig(
        outdir=Path(args.outdir),
        seeds=seeds,
        quick=args.quick,
        device=device,
        dtype=torch.float32,
        grid_2d=grid_2d,
        actions=4,
        sample_values=sample_values,
        eps_grid=eps_grid,
        beta_z=0.35,
        z_bins=z_bins,
        eval_size=eval_size,
        max_bisection_steps=max_steps,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    dirs = make_dirs(cfg.outdir)

    print("=" * 84)
    print("Section 8 Numerical Validation — VF Final Black-Box Boundary Learning")
    print(f"Output directory : {cfg.outdir.resolve()}")
    print(f"Device           : {cfg.device}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"Seeds            : {cfg.seeds}")
    print(f"Quick mode       : {cfg.quick}")
    print("=" * 84)

    start = time.time()

    metadata = {
        "script": "run_section8_policy_geometry_vF.py",
        "seeds": cfg.seeds,
        "quick": cfg.quick,
        "device": cfg.device,
        "cuda_available": torch.cuda.is_available(),
        "grid_2d": cfg.grid_2d,
        "actions": cfg.actions,
        "sample_values": cfg.sample_values,
        "eps_grid": cfg.eps_grid,
        "z_bins": cfg.z_bins,
        "eval_size": cfg.eval_size,
        "structured_families": STRUCTURED_FAMILIES,
        "unstructured_families": UNSTRUCTURED_FAMILIES,
        "black_box_protocol": "learner observes only (x,z,pi*(x,z)); no Q*, no gaps, no thresholds",
        "critical_fix": "separates oracle_seed from query_seed so the target policy is fixed across strategies",
        "active_protocol": "adaptive monotone bracketing and bisection using action labels only",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(dirs["logs"] / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[1/4] Plotting black-box target geometry...")
    plot_blackbox_tessellation(cfg, dirs)

    print("[2/4] Running VF black-box boundary learning...")
    experiment_blackbox_learning(cfg, dirs)

    print("[3/4] Plotting representative boundary reconstruction...")
    plot_reconstruction_example(cfg, dirs)

    print("[4/4] Running failure-mode validation...")
    experiment_failure_modes(cfg, dirs)

    manifest_rows = []
    for subdir in ("data", "figures", "tables", "logs"):
        for p in sorted((cfg.outdir / subdir).glob("*")):
            manifest_rows.append({"type": subdir, "path": str(p)})

    save_df(pd.DataFrame(manifest_rows), dirs["root"] / "manifest.csv")

    print("=" * 84)
    print(f"Completed in {time.time() - start:.2f} seconds.")
    print(f"Manifest saved to {dirs['root'] / 'manifest.csv'}")
    print("=" * 84)


if __name__ == "__main__":
    main()
