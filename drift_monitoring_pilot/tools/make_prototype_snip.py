#!/usr/bin/env python3
"""Create a tiny drift-monitoring snip artifact.

The artifact contains summary statistics, histograms and optional quantized
feature prototypes.  It does not contain raw rows or raw feature vectors.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np

from drift_utils import (
    SCHEMA_SNIP,
    MetricRow,
    feature_snip_from_matrix,
    filter_latest_window,
    find_feature_files,
    find_metric_files,
    load_feature_matrix,
    make_numeric_snip,
    read_numeric_rows,
    source_hashes,
    utc_now_iso,
    write_gzip_json,
)

HARD_MAX_BYTES = 50 * 1024


def _demo_rows(seed: int = 12345, n: int = 240) -> List[MetricRow]:
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rows: List[MetricRow] = []
    for i in range(n):
        ts = now - timedelta(hours=(n - i) * 3)
        base_error = 0.025 + 0.003 * np.sin(i / 80.0)
        rows.append(
            MetricRow(
                timestamp=ts,
                values={
                    "error_rate": float(np.clip(rng.normal(base_error, 0.006), 0.0, 1.0)),
                    "latency_ms": float(np.clip(rng.normal(150.0, 20.0), 1.0, None)),
                    "quality_score": float(np.clip(rng.normal(0.86 - base_error, 0.04), 0.0, 1.0)),
                },
            )
        )
    return rows


def _demo_features(seed: int = 12345, n: int = 500, d: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, d))
    assignments = rng.integers(0, centers.shape[0], size=n)
    return centers[assignments] + rng.normal(scale=0.25, size=(n, d))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a compact monitoring snip from metrics/features.")
    p.add_argument("--input", action="append", default=[], help="Metric file or directory. Can be repeated.")
    p.add_argument("--features", action="append", default=[], help="Feature/embedding file or directory. Can be repeated.")
    p.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY") or Path.cwd().name, help="Repo id stored in the snip.")
    p.add_argument("--out", required=True, help="Output .json.gz/.pq.gz path.")
    p.add_argument("--window-days", type=int, default=30, help="Keep only the latest N days when timestamps exist.")
    p.add_argument("--hist-bins", type=int, default=32, help="Histogram bins per numeric metric.")
    p.add_argument("--min-samples", type=int, default=2, help="Minimum rows per numeric metric in the snip.")
    p.add_argument("--feature-k", type=int, default=8, help="Number of feature centroids.")
    p.add_argument("--feature-max-rows", type=int, default=5000, help="Max feature rows used to build prototypes.")
    p.add_argument("--feature-max-dim", type=int, default=64, help="Project feature vectors to at most this dimension.")
    p.add_argument("--seed", type=int, default=12345, help="Deterministic seed for projection/k-means.")
    p.add_argument("--max-bytes", type=int, default=32 * 1024, help="Reject artifact if gzip size exceeds this.")
    p.add_argument("--demo", action="store_true", help="Generate synthetic demo data instead of reading files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_bytes > HARD_MAX_BYTES:
        raise SystemExit(f"--max-bytes may not exceed {HARD_MAX_BYTES} bytes (50 KiB hard cap)")
    if args.window_days <= 0:
        raise SystemExit("--window-days must be positive")
    if args.hist_bins < 4:
        raise SystemExit("--hist-bins should be at least 4")

    input_paths = [Path(p) for p in args.input] or [Path.cwd()]
    feature_paths_arg = [Path(p) for p in args.features]

    if args.demo:
        rows = _demo_rows(seed=args.seed)
        metric_files: List[Path] = []
        X = _demo_features(seed=args.seed)
        feature_files: List[Path] = []
    else:
        metric_files = find_metric_files(input_paths)
        rows = read_numeric_rows(metric_files)
        feature_files = find_feature_files(feature_paths_arg or input_paths)
        X = load_feature_matrix(feature_files)

    window_rows = filter_latest_window(rows, args.window_days) if rows else []
    numeric = make_numeric_snip(window_rows, hist_bins=args.hist_bins, min_samples=args.min_samples) if window_rows else {}

    features = None
    if X is not None and X.shape[0] >= 2:
        features = feature_snip_from_matrix(
            X,
            k=args.feature_k,
            seed=args.seed,
            max_rows=args.feature_max_rows,
            max_dim=args.feature_max_dim,
        )

    if not numeric and features is None:
        searched = ", ".join(str(p) for p in input_paths)
        raise SystemExit(
            "No monitorable numeric metrics or feature vectors found. "
            f"Searched: {searched}. Pass --input/--features explicitly or use --demo."
        )

    timestamps = [r.timestamp for r in window_rows if r.timestamp is not None]
    window = {
        "days": args.window_days,
        "start": min(timestamps).isoformat().replace("+00:00", "Z") if timestamps else None,
        "end": max(timestamps).isoformat().replace("+00:00", "Z") if timestamps else None,
        "row_count": len(window_rows),
    }
    source_files = metric_files + ([] if args.demo else feature_files)
    payload = {
        "schema_version": SCHEMA_SNIP,
        "created_at": utc_now_iso(),
        "repo": args.repo,
        "window": window,
        "privacy": {
            "raw_rows_included": False,
            "raw_features_included": False,
            "hard_max_bytes": HARD_MAX_BYTES,
            "requested_max_bytes": args.max_bytes,
        },
        "source": {
            "metric_file_count": len(metric_files),
            "feature_file_count": 0 if args.demo else len(feature_files),
            "hashes": source_hashes(source_files, base=Path.cwd()) if source_files else [],
            "demo": bool(args.demo),
        },
        "numeric": numeric,
        "features": features,
        "parameters": {
            "hist_bins": args.hist_bins,
            "min_samples": args.min_samples,
            "feature_k": args.feature_k,
            "feature_max_rows": args.feature_max_rows,
            "feature_max_dim": args.feature_max_dim,
            "seed": args.seed,
        },
    }

    out = Path(args.out)
    meta = write_gzip_json(payload, out)
    if meta["bytes"] > args.max_bytes:
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        raise SystemExit(
            f"Snip too large: {meta['bytes']} bytes > --max-bytes {args.max_bytes}. "
            "Reduce --hist-bins, --feature-k or --feature-max-dim."
        )

    print(f"Wrote {out} ({meta['bytes']} bytes)")
    print(f"sha256={meta['sha256']}")
    print(f"numeric_metrics={len(numeric)} features={'yes' if features is not None else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
