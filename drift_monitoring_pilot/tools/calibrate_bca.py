#!/usr/bin/env python3
"""Calibrate conservative drift thresholds with BCa bootstrap intervals."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from drift_utils import (
    SCHEMA_CALIBRATION,
    bca_interval,
    bootstrap_stat_values,
    drift_numeric_from_snip,
    drift_stat_from_sample,
    feature_centroid_drift,
    feature_snip_from_matrix,
    filter_latest_window,
    find_feature_files,
    find_metric_files,
    infer_repo_name,
    load_feature_matrix,
    make_numeric_snip,
    read_numeric_rows,
    rolling_windows,
    source_hashes,
    stable_seed,
    utc_now_iso,
    values_for_metric,
    write_gzip_json,
)

DRIFT_KINDS = ("mean_delta", "p95_delta", "psi")
HARD_MAX_BYTES = 50 * 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate BCa drift thresholds for one or more repos.")
    p.add_argument("--repos", default="", help="Comma-separated repo directories. Defaults to current directory.")
    p.add_argument("--repo", action="append", default=[], help="Repo directory. Can be repeated.")
    p.add_argument(
        "--history-input",
        action="append",
        default=[],
        help=(
            "Historical metric/feature file or directory relative to each repo. "
            "Can be repeated. Defaults to the repo root for backward compatibility."
        ),
    )
    p.add_argument("--out", required=True, help="Output calibration .json.gz path.")
    p.add_argument("--window-days", type=int, default=30, help="Rolling baseline window size.")
    p.add_argument("--min-samples", type=int, default=30, help="Minimum rows per window/metric.")
    p.add_argument("--min-windows", type=int, default=10, help="Minimum rolling windows before using window-drift BCa.")
    p.add_argument("--B", type=int, default=2000, help="Bootstrap replicates for numeric thresholds.")
    p.add_argument("--seed", type=int, default=12345, help="Base seed stored in the artifact.")
    p.add_argument("--alpha", type=float, default=0.05, help="Confidence interval alpha; 0.05 means 95%% CI.")
    p.add_argument("--alarm-quantile", type=float, default=0.95, help="Historic drift quantile used as threshold target.")
    p.add_argument("--hist-bins", type=int, default=32, help="Histogram bins in baseline summaries.")
    p.add_argument("--feature-k", type=int, default=8, help="Feature prototype centroids.")
    p.add_argument("--feature-B", type=int, default=200, help="Bootstrap replicates for feature threshold calibration.")
    p.add_argument("--max-bytes", type=int, default=HARD_MAX_BYTES, help="Reject calibration artifact above this size.")
    p.add_argument("--demo", action="store_true", help="Create a synthetic demo repo under the output directory parent.")
    return p.parse_args()


def _repo_items(args: argparse.Namespace) -> List[Path]:
    items: List[str] = []
    if args.repos.strip():
        items.extend([x.strip() for x in args.repos.split(",") if x.strip()])
    items.extend(args.repo)
    if not items:
        items = [os.getcwd()]
    return [Path(x) for x in items]


def _history_inputs_for_repo(repo_path: Path, history_inputs: Sequence[str]) -> List[Path]:
    if not history_inputs:
        return [repo_path]
    paths: List[Path] = []
    for raw in history_inputs:
        path = Path(raw)
        paths.append(path if path.is_absolute() else repo_path / path)
    return paths


def _make_demo_repo(root: Path, seed: int) -> Path:
    """Create a synthetic repo-like folder for smoke testing."""
    from datetime import datetime, timedelta, timezone
    import json

    rng = np.random.default_rng(seed)
    repo = root / "demo_repo"
    hist = repo / "monitoring" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    with (hist / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for day in range(90):
            # small natural seasonality without true regression
            base_error = 0.025 + 0.003 * np.sin(day / 7.0)
            base_latency = 150 + 8 * np.sin(day / 10.0)
            for j in range(10):
                ts = now - timedelta(days=90 - day, hours=j)
                row = {
                    "timestamp": ts.isoformat().replace("+00:00", "Z"),
                    "error_rate": float(np.clip(rng.normal(base_error, 0.006), 0.0, 1.0)),
                    "latency_ms": float(np.clip(rng.normal(base_latency, 20), 1.0, None)),
                    "quality_score": float(np.clip(rng.normal(0.86 - base_error, 0.04), 0.0, 1.0)),
                }
                f.write(json.dumps(row) + "\n")
    centers = rng.normal(size=(4, 32))
    assignments = rng.integers(0, 4, size=1200)
    X = centers[assignments] + rng.normal(scale=0.25, size=(1200, 32))
    np.save(hist / "features.npy", X)
    return repo


def _calibrate_numeric_metric(
    metric: str,
    baseline_summary: Mapping[str, Any],
    baseline_values: np.ndarray,
    window_summaries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    repo_name: str,
) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {}
    for kind in DRIFT_KINDS:
        drift_values = [drift_numeric_from_snip(s, baseline_summary, kind) for s in window_summaries]
        drift_arr = np.asarray([x for x in drift_values if np.isfinite(x)], dtype=float)
        seed = stable_seed(args.seed, repo_name, metric, kind)
        if drift_arr.size >= args.min_windows and float(np.max(drift_arr)) > 0.0:
            ci = bca_interval(
                drift_arr,
                stat_fn=lambda x, q=args.alarm_quantile: float(np.quantile(x, q)),
                B=args.B,
                alpha=args.alpha,
                seed=seed,
            )
            method = "bca_rolling_window_drift_quantile"
            supporting_n = int(drift_arr.size)
        else:
            # Sparse history fallback: create the sampling-noise distribution by
            # bootstrapping the baseline rows, then use BCa on the q95 of those
            # bootstrap-error values.  This is conservative for small pilots and
            # is explicitly marked in the artifact.
            stat = lambda sample, base=baseline_summary, k=kind: drift_stat_from_sample(sample, base, k)
            errors = bootstrap_stat_values(baseline_values, stat, B=args.B, seed=seed)
            ci = bca_interval(
                errors,
                stat_fn=lambda x, q=args.alarm_quantile: float(np.quantile(x, q)),
                B=min(args.B, 1000),
                alpha=args.alpha,
                seed=stable_seed(seed, "fallback-ci"),
            )
            method = "bca_over_bootstrap_sampling_error_fallback"
            supporting_n = int(errors.size)
        thresholds[kind] = {
            "threshold": float(ci["ci_high"]),
            "ci_low": float(ci["ci_low"]),
            "ci_high": float(ci["ci_high"]),
            "theta_hat": float(ci["theta_hat"]),
            "method": method,
            "supporting_n": supporting_n,
            "B": int(args.B),
            "seed": int(seed),
            "alpha": float(args.alpha),
            "alarm_quantile": float(args.alarm_quantile),
        }
    return thresholds


def _bootstrap_feature_threshold(X: np.ndarray, baseline_features: Mapping[str, Any], args: argparse.Namespace, repo_name: str) -> Dict[str, Any]:
    rng = np.random.default_rng(stable_seed(args.seed, repo_name, "features"))
    n = X.shape[0]
    B = max(50, int(args.feature_B))
    values = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_features = feature_snip_from_matrix(
            X[idx],
            k=args.feature_k,
            seed=args.seed,
        )
        values[b] = feature_centroid_drift(boot_features, baseline_features)
    ci = bca_interval(
        values,
        stat_fn=lambda x, q=args.alarm_quantile: float(np.quantile(x, q)),
        B=min(1000, max(100, B)),
        alpha=args.alpha,
        seed=stable_seed(args.seed, repo_name, "features-ci"),
    )
    return {
        "centroid_drift": {
            "threshold": float(ci["ci_high"]),
            "ci_low": float(ci["ci_low"]),
            "ci_high": float(ci["ci_high"]),
            "theta_hat": float(ci["theta_hat"]),
            "method": "bca_over_feature_bootstrap_drift_quantile",
            "supporting_n": int(values.size),
            "B": int(B),
            "seed": int(stable_seed(args.seed, repo_name, "features")),
            "alpha": float(args.alpha),
            "alarm_quantile": float(args.alarm_quantile),
        }
    }


def calibrate_repo(repo_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    repo_name = infer_repo_name(str(repo_path))
    history_inputs = _history_inputs_for_repo(repo_path, args.history_input)
    metric_files = find_metric_files(history_inputs)
    rows = read_numeric_rows(metric_files)
    feature_files = find_feature_files(history_inputs)
    X = load_feature_matrix(feature_files)
    warnings: List[str] = []

    baseline_rows = filter_latest_window(rows, args.window_days) if rows else []
    if len(baseline_rows) < args.min_samples and len(rows) >= args.min_samples:
        warnings.append(
            f"Latest {args.window_days}d window has {len(baseline_rows)} rows; using all {len(rows)} rows for baseline."
        )
        baseline_rows = rows
    if len(baseline_rows) < args.min_samples and X is None:
        raise ValueError(
            f"{repo_name}: not enough rows for calibration: {len(baseline_rows)} < min_samples={args.min_samples}"
        )

    baseline_numeric = make_numeric_snip(baseline_rows, hist_bins=args.hist_bins, min_samples=args.min_samples) if baseline_rows else {}
    windows = rolling_windows(rows, args.window_days, min_samples=args.min_samples) if rows else []
    numeric_thresholds: Dict[str, Any] = {}
    for metric, base_summary in baseline_numeric.items():
        window_summaries: List[Mapping[str, Any]] = []
        for win in windows:
            win_snip = make_numeric_snip(win, hist_bins=args.hist_bins, min_samples=args.min_samples)
            if metric in win_snip:
                window_summaries.append(win_snip[metric])
        baseline_values = values_for_metric(baseline_rows, metric)
        if baseline_values.size < args.min_samples:
            warnings.append(f"Metric {metric} skipped: {baseline_values.size} samples < {args.min_samples}.")
            continue
        numeric_thresholds[metric] = _calibrate_numeric_metric(
            metric=metric,
            baseline_summary=base_summary,
            baseline_values=baseline_values,
            window_summaries=window_summaries,
            args=args,
            repo_name=repo_name,
        )

    baseline_features = None
    feature_thresholds: Dict[str, Any] = {}
    if X is not None and X.shape[0] >= max(2, args.min_samples):
        baseline_features = feature_snip_from_matrix(X, k=args.feature_k, seed=args.seed)
        feature_thresholds = _bootstrap_feature_threshold(X, baseline_features, args, repo_name)
    elif X is not None:
        warnings.append(f"Features skipped: {X.shape[0]} rows < min_samples={args.min_samples}.")

    if not baseline_numeric and baseline_features is None:
        raise ValueError(f"{repo_name}: no calibratable metrics/features found")

    return {
        "repo": repo_name,
        "baseline": {
            "numeric": baseline_numeric,
            "features": baseline_features,
        },
        "thresholds": {
            "numeric": numeric_thresholds,
            "features": feature_thresholds,
        },
        "source": {
            "history_inputs": [str(p) for p in history_inputs],
            "metric_file_count": len(metric_files),
            "feature_file_count": len(feature_files),
            "hashes": source_hashes(metric_files + feature_files, base=Path.cwd()),
            "raw_metric_rows": len(rows),
            "baseline_rows": len(baseline_rows),
            "rolling_windows": len(windows),
        },
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    if args.max_bytes > HARD_MAX_BYTES:
        raise SystemExit(f"--max-bytes may not exceed {HARD_MAX_BYTES} bytes (50 KiB hard cap)")
    if args.window_days <= 0:
        raise SystemExit("--window-days must be positive")
    if args.min_samples < 2:
        raise SystemExit("--min-samples must be at least 2")
    if args.B < 100:
        raise SystemExit("--B should be at least 100")
    if not 0.5 <= args.alarm_quantile < 1.0:
        raise SystemExit("--alarm-quantile should be in [0.5, 1.0)")

    repo_paths = _repo_items(args)
    if args.demo:
        demo_root = Path.cwd() / "_demo_data"
        repo_paths = [_make_demo_repo(demo_root, seed=args.seed)]

    repos: Dict[str, Any] = {}
    failures: List[str] = []
    for repo_path in repo_paths:
        try:
            cfg = calibrate_repo(repo_path, args)
            repos[cfg["repo"]] = cfg
            print(f"Calibrated {cfg['repo']}: {len(cfg['baseline']['numeric'])} numeric metrics")
        except Exception as exc:  # keep other repos alive in a 30-repo pilot
            msg = f"{repo_path}: {exc}"
            failures.append(msg)
            print(f"WARNING: {msg}", file=sys.stderr)

    if not repos:
        raise SystemExit("No repos were calibrated successfully. " + "; ".join(failures))

    payload = {
        "schema_version": SCHEMA_CALIBRATION,
        "created_at": utc_now_iso(),
        "parameters": {
            "window_days": args.window_days,
            "min_samples": args.min_samples,
            "min_windows": args.min_windows,
            "B": args.B,
            "seed": args.seed,
            "alpha": args.alpha,
            "alarm_quantile": args.alarm_quantile,
            "hist_bins": args.hist_bins,
            "feature_k": args.feature_k,
            "feature_B": args.feature_B,
            "history_input": args.history_input,
        },
        "privacy": {
            "raw_rows_included": False,
            "raw_features_included": False,
            "hard_max_bytes": HARD_MAX_BYTES,
            "requested_max_bytes": args.max_bytes,
        },
        "repos": repos,
        "failures": failures,
    }
    out = Path(args.out)
    meta = write_gzip_json(payload, out)
    if meta["bytes"] > args.max_bytes:
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(
            f"Calibration artifact too large: {meta['bytes']} bytes > --max-bytes {args.max_bytes}. "
            "Reduce --hist-bins or calibrate fewer metrics/repos per artifact."
        )
    print(f"Wrote {out} ({meta['bytes']} bytes)")
    print(f"sha256={meta['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
