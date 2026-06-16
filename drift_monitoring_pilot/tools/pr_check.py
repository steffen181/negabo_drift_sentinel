#!/usr/bin/env python3
"""Compare a PR snip against a calibration artifact and fail on drift alarms."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from drift_utils import (
    SCHEMA_CALIBRATION,
    SCHEMA_SNIP,
    bca_interval,
    drift_numeric_from_snip,
    drift_stat_from_sample,
    feature_centroid_drift,
    histogram_to_sample,
    read_gzip_json,
    stable_seed,
    utc_now_iso,
)

DRIFT_KINDS = ("mean_delta", "p95_delta", "psi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fail a PR when conservative drift CI exceeds calibrated thresholds.")
    p.add_argument("--calib", required=True, help="Calibration .json.gz path.")
    p.add_argument("--snip", required=True, help="PR snip .json.gz/.pq.gz path.")
    p.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY") or "", help="Repo id. Defaults to GITHUB_REPOSITORY or snip repo.")
    p.add_argument("--B", type=int, default=0, help="PR bootstrap replicates. 0 means use calibration B.")
    p.add_argument("--seed", type=int, default=0, help="PR bootstrap seed. 0 means derive from calibration seed.")
    p.add_argument("--alpha", type=float, default=0.05, help="PR CI alpha; 0.05 means 95%% CI.")
    p.add_argument("--max-points", type=int, default=2000, help="Max pseudo-sample points reconstructed from a histogram.")
    p.add_argument("--report", default="artifacts/pr-drift-report.json", help="JSON report path.")
    p.add_argument("--markdown", default="", help="Optional markdown report path.")
    p.add_argument("--warn-only", action="store_true", help="Do not exit non-zero on alarms.")
    return p.parse_args()


def _select_repo_config(calib: Mapping[str, Any], snip: Mapping[str, Any], requested_repo: str) -> Mapping[str, Any]:
    repos = calib.get("repos", {})
    if not isinstance(repos, Mapping) or not repos:
        raise ValueError("Calibration artifact has no repos")
    candidates = [requested_repo, snip.get("repo", "")]
    for repo in candidates:
        if repo and repo in repos:
            return repos[repo]
    # Convenience fallback for single-repo artifacts.
    if len(repos) == 1:
        return next(iter(repos.values()))
    raise ValueError(
        f"No calibration entry for repo={requested_repo or snip.get('repo')!r}. "
        f"Available: {', '.join(repos.keys())}"
    )


def _rule_threshold(rule: Mapping[str, Any]) -> float:
    if "threshold" in rule:
        return float(rule["threshold"])
    if "ci_high" in rule:
        return float(rule["ci_high"])
    raise ValueError("Threshold rule has neither threshold nor ci_high")


def _check_numeric_metric(
    metric: str,
    current_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    rules: Mapping[str, Any],
    B: int,
    alpha: float,
    seed: int,
    max_points: int,
) -> Dict[str, Any]:
    sample = histogram_to_sample(current_summary, max_points=max_points)
    checks: Dict[str, Any] = {}
    for kind in DRIFT_KINDS:
        if kind not in rules:
            continue
        threshold = _rule_threshold(rules[kind])
        point = drift_numeric_from_snip(current_summary, baseline_summary, kind)
        stat = lambda x, base=baseline_summary, k=kind: drift_stat_from_sample(x, base, k)
        ci = bca_interval(sample, stat, B=B, alpha=alpha, seed=stable_seed(seed, metric, kind), max_jackknife=1000)
        ci_high = float(ci["ci_high"])
        checks[kind] = {
            "point": float(point),
            "theta_hat_from_histogram_sample": float(ci["theta_hat"]),
            "ci_low": float(ci["ci_low"]),
            "ci_high": ci_high,
            "threshold": threshold,
            "alarm": bool(ci_high > threshold),
            "B": int(B),
            "seed": int(stable_seed(seed, metric, kind)),
            "method": ci["method"],
        }
    return checks


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = []
    status = "FAIL" if report.get("alarm") else "PASS"
    lines.append(f"# PR drift check: {status}")
    lines.append("")
    lines.append(f"Repo: `{report.get('repo')}`  ")
    lines.append(f"Created: `{report.get('created_at')}`")
    lines.append("")
    lines.append("## Numeric checks")
    numeric = report.get("numeric", {})
    if not numeric:
        lines.append("No numeric checks were run.")
    else:
        lines.append("| metric | check | point | ci_high | threshold | alarm |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for metric, checks in numeric.items():
            for kind, result in checks.items():
                lines.append(
                    f"| `{metric}` | `{kind}` | {result['point']:.6g} | "
                    f"{result['ci_high']:.6g} | {result['threshold']:.6g} | {result['alarm']} |"
                )
    lines.append("")
    lines.append("## Feature checks")
    features = report.get("features")
    if not features:
        lines.append("No feature check was run.")
    else:
        lines.append("Feature checks use prototype point estimates; PR snips do not contain raw feature rows, so no PR CI is computed.")
        lines.append("")
        lines.append("| check | point estimate | threshold | alarm | method |")
        lines.append("|---|---:|---:|---:|---|")
        for kind, result in features.items():
            lines.append(
                f"| `{kind}` | {result['point']:.6g} | {result['threshold']:.6g} | "
                f"{result['alarm']} | `{result.get('method', 'point_estimate')}` |"
            )
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    calib = read_gzip_json(Path(args.calib))
    snip = read_gzip_json(Path(args.snip))
    if calib.get("schema_version") != SCHEMA_CALIBRATION:
        raise SystemExit(f"Unexpected calibration schema: {calib.get('schema_version')}")
    if snip.get("schema_version") != SCHEMA_SNIP:
        raise SystemExit(f"Unexpected snip schema: {snip.get('schema_version')}")

    repo_cfg = _select_repo_config(calib, snip, args.repo or str(snip.get("repo", "")))
    params = calib.get("parameters", {})
    B = int(args.B or params.get("B", 1000))
    seed = int(args.seed or params.get("seed", 12345))
    if B < 100:
        raise SystemExit("PR bootstrap B must be >= 100")

    baseline_numeric = repo_cfg.get("baseline", {}).get("numeric", {}) or {}
    threshold_numeric = repo_cfg.get("thresholds", {}).get("numeric", {}) or {}
    current_numeric = snip.get("numeric", {}) or {}
    numeric_report: Dict[str, Any] = {}
    warnings = []

    for metric, current_summary in current_numeric.items():
        if metric not in baseline_numeric:
            warnings.append(f"Metric {metric} exists in PR snip but not in calibration; skipped.")
            continue
        if metric not in threshold_numeric:
            warnings.append(f"Metric {metric} has no thresholds in calibration; skipped.")
            continue
        numeric_report[metric] = _check_numeric_metric(
            metric=metric,
            current_summary=current_summary,
            baseline_summary=baseline_numeric[metric],
            rules=threshold_numeric[metric],
            B=B,
            alpha=args.alpha,
            seed=seed,
            max_points=args.max_points,
        )

    for metric in baseline_numeric.keys():
        if metric not in current_numeric:
            warnings.append(f"Metric {metric} exists in calibration but not in PR snip.")

    feature_report: Optional[Dict[str, Any]] = None
    baseline_features = repo_cfg.get("baseline", {}).get("features")
    current_features = snip.get("features")
    feature_rules = repo_cfg.get("thresholds", {}).get("features", {}) or {}
    if baseline_features and current_features and "centroid_drift" in feature_rules:
        point = feature_centroid_drift(current_features, baseline_features)
        threshold = _rule_threshold(feature_rules["centroid_drift"])
        # The snip has only quantized prototypes, not raw features; we therefore
        # use the point estimate as conservative bound for this prototype check.
        feature_report = {
            "centroid_drift": {
                "point": float(point),
                "ci_low": float(point),
                "ci_high": float(point),
                "threshold": threshold,
                "alarm": bool(point > threshold),
                "method": "prototype_point_estimate_no_raw_features_in_snip",
            }
        }
    elif baseline_features and not current_features:
        warnings.append("Calibration contains feature baseline, but PR snip has no feature prototypes.")

    alarm = any(result["alarm"] for checks in numeric_report.values() for result in checks.values())
    if feature_report:
        alarm = alarm or any(result["alarm"] for result in feature_report.values())

    report: Dict[str, Any] = {
        "schema_version": "drift-pr-report/v1",
        "created_at": utc_now_iso(),
        "repo": repo_cfg.get("repo") or snip.get("repo"),
        "alarm": bool(alarm),
        "parameters": {"B": B, "seed": seed, "alpha": args.alpha, "max_points": args.max_points},
        "numeric": numeric_report,
        "features": feature_report,
        "warnings": warnings,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = _markdown_report(report)
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON report: {report_path}")

    if alarm and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
