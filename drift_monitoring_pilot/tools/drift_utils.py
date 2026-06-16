#!/usr/bin/env python3
"""Shared utilities for tiny drift-monitoring artifacts.

The module is intentionally dependency-light.  It uses numpy plus the Python
standard library, so it can run in GitHub Actions without a service backend.

Data model used by the scripts:
- raw rows are read from JSONL/CSV into MetricRow(timestamp, values)
- public artifacts contain only summaries, histograms, hashes and optional
  quantized feature prototypes; they never contain raw rows or raw feature
  vectors
- BCa confidence intervals are computed locally from raw rows during
  calibration and approximately from histogram sketches during PR checks
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_SNIP = "drift-snip/v1"
SCHEMA_CALIBRATION = "drift-calibration/v1"
DEFAULT_QUANTILES: Tuple[Tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)
TIMESTAMP_KEYS = {"timestamp", "time", "created_at", "date", "datetime", "ts"}
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "artifacts",
}


@dataclass(frozen=True)
class MetricRow:
    timestamp: Optional[datetime]
    values: Dict[str, float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _round_floats(obj: Any, digits: int = 10) -> Any:
    """Round floats recursively to keep gzip artifacts small and stable."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        # round() keeps integers readable while bounding binary noise.
        return round(obj, digits)
    if isinstance(obj, np.floating):
        x = float(obj)
        return round(x, digits) if math.isfinite(x) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Mapping):
        return {str(k): _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, digits) for v in obj]
    return obj


def canonical_json_bytes(obj: Mapping[str, Any]) -> bytes:
    cleaned = _round_floats(obj)
    return json.dumps(
        cleaned,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gzip_json(obj: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """Write canonical gzip JSON and return size/hash metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(obj)
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    return {
        "path": str(path),
        "bytes": len(compressed),
        "sha256": sha256_bytes(compressed),
        "json_sha256": sha256_bytes(payload),
    }


def read_gzip_json(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return obj


def relative_path(path: Path, base: Optional[Path] = None) -> str:
    base = base or Path.cwd()
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def source_hashes(paths: Sequence[Path], base: Optional[Path] = None, max_files: int = 200) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for p in list(paths)[:max_files]:
        if p.is_file():
            out.append({"path": relative_path(p, base), "sha256": sha256_file(p)})
    return out


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds > 1_000_000_000_000:  # milliseconds
            seconds = seconds / 1000.0
        try:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Numeric timestamps serialized as strings.
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
            try:
                return parse_timestamp(float(text))
            except ValueError:
                return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # Last-resort support for YYYY-MM-DD.
            try:
                dt = datetime.strptime(text, "%Y-%m-%d")
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def flatten_numeric(obj: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Extract numeric scalar fields from a JSON object.

    Nested dictionaries are flattened as "parent.child". Lists are ignored here;
    embedding vectors are handled by load_feature_matrix().
    """
    out: Dict[str, float] = {}
    for key, value in obj.items():
        k = str(key)
        full_key = f"{prefix}.{k}" if prefix else k
        if k.lower() in TIMESTAMP_KEYS:
            continue
        if isinstance(value, Mapping):
            out.update(flatten_numeric(value, full_key))
            continue
        if isinstance(value, (list, tuple)):
            continue
        x = _maybe_float(value)
        if x is not None:
            out[full_key] = x
    return out


def _timestamp_from_mapping(obj: Mapping[str, Any]) -> Optional[datetime]:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in TIMESTAMP_KEYS:
        if key in lowered:
            ts = parse_timestamp(lowered[key])
            if ts is not None:
                return ts
    return None


def _iter_jsonl(path: Path) -> Iterator[MetricRow]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, Mapping):
                continue
            values = flatten_numeric(obj)
            if values:
                yield MetricRow(timestamp=_timestamp_from_mapping(obj), values=values)


def _iter_csv(path: Path) -> Iterator[MetricRow]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        for row in reader:
            ts = _timestamp_from_mapping(row)
            values: Dict[str, float] = {}
            for key, value in row.items():
                if key is None or key.lower() in TIMESTAMP_KEYS:
                    continue
                x = _maybe_float(value)
                if x is not None:
                    values[key] = x
            if values:
                yield MetricRow(timestamp=ts, values=values)


def read_numeric_rows(paths: Sequence[Path]) -> List[MetricRow]:
    rows: List[MetricRow] = []
    for path in paths:
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith(".jsonl") or suffixes.endswith(".ndjson"):
            rows.extend(_iter_jsonl(path))
        elif suffixes.endswith(".csv"):
            rows.extend(_iter_csv(path))
    return rows


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def find_metric_files(inputs: Sequence[Path]) -> List[Path]:
    """Find likely metric files under the supplied files/directories."""
    files: List[Path] = []
    name_re = re.compile(r"(metric|metrics|score|scores|eval|quality|monitor)", re.IGNORECASE)
    for raw in inputs:
        path = raw.expanduser()
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() in {".jsonl", ".ndjson", ".csv"}:
                files.append(path)
            continue
        for p in path.rglob("*"):
            if not p.is_file() or _should_skip(p):
                continue
            if p.suffix.lower() not in {".jsonl", ".ndjson", ".csv"}:
                continue
            # Explicit monitoring-ish names by default.  This avoids accidentally
            # reading large unrelated CSVs in a repository root.
            if name_re.search(p.name) or "monitoring" in {part.lower() for part in p.parts}:
                files.append(p)
    return sorted(set(files))


def find_feature_files(inputs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    name_re = re.compile(r"(feature|features|embedding|embeddings|vector|vectors)", re.IGNORECASE)
    for raw in inputs:
        path = raw.expanduser()
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() in {".npy", ".csv", ".jsonl", ".ndjson"}:
                files.append(path)
            continue
        for p in path.rglob("*"):
            if not p.is_file() or _should_skip(p):
                continue
            if p.suffix.lower() not in {".npy", ".csv", ".jsonl", ".ndjson"}:
                continue
            if name_re.search(p.name):
                files.append(p)
    return sorted(set(files))


def filter_latest_window(rows: Sequence[MetricRow], window_days: int) -> List[MetricRow]:
    stamped = [r for r in rows if r.timestamp is not None]
    if not stamped:
        return list(rows)
    end = max(r.timestamp for r in stamped if r.timestamp is not None)
    start = end - timedelta(days=window_days)
    return [r for r in rows if r.timestamp is not None and start <= r.timestamp <= end]


def rolling_windows(rows: Sequence[MetricRow], window_days: int, min_samples: int, step_days: int = 1) -> List[List[MetricRow]]:
    stamped = [r for r in rows if r.timestamp is not None]
    if not stamped:
        return [list(rows)] if len(rows) >= min_samples else []
    min_ts = min(r.timestamp for r in stamped if r.timestamp is not None)
    max_ts = max(r.timestamp for r in stamped if r.timestamp is not None)
    windows: List[List[MetricRow]] = []
    end = min_ts + timedelta(days=window_days)
    while end <= max_ts + timedelta(seconds=1):
        start = end - timedelta(days=window_days)
        win = [r for r in stamped if r.timestamp is not None and start <= r.timestamp <= end]
        if len(win) >= min_samples:
            windows.append(win)
        end += timedelta(days=step_days)
    # Always include the latest rolling window if it was missed by date stepping.
    latest = filter_latest_window(rows, window_days)
    if len(latest) >= min_samples:
        latest_keys = {(r.timestamp, tuple(sorted(r.values.items()))) for r in latest}
        existing = [
            {(r.timestamp, tuple(sorted(r.values.items()))) for r in win} == latest_keys
            for win in windows
        ]
        if not any(existing):
            windows.append(latest)
    return windows


def numeric_columns(rows: Sequence[MetricRow], min_samples: int = 1) -> List[str]:
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.values.items():
            if math.isfinite(value):
                counts[key] = counts.get(key, 0) + 1
    return sorted(k for k, n in counts.items() if n >= min_samples)


def values_for_metric(rows: Sequence[MetricRow], metric: str) -> np.ndarray:
    values = [r.values[metric] for r in rows if metric in r.values and math.isfinite(r.values[metric])]
    return np.asarray(values, dtype=float)


def numeric_summary(values: np.ndarray, hist_bins: int = 32) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty numeric array")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        pad = max(abs(vmin) * 1e-6, 1e-9)
        hist_min, hist_max = vmin - pad, vmax + pad
    else:
        hist_min, hist_max = vmin, vmax
    counts, edges = np.histogram(values, bins=int(hist_bins), range=(hist_min, hist_max))
    summary: Dict[str, Any] = {
        "n": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "min": vmin,
        "max": vmax,
        "hist": {
            "edges": edges.astype(float).tolist(),
            "counts": counts.astype(int).tolist(),
        },
    }
    for name, q in DEFAULT_QUANTILES:
        summary[name] = float(np.quantile(values, q))
    return summary


def make_numeric_snip(rows: Sequence[MetricRow], hist_bins: int = 32, min_samples: int = 1) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for metric in numeric_columns(rows, min_samples=min_samples):
        values = values_for_metric(rows, metric)
        if values.size >= min_samples:
            out[metric] = numeric_summary(values, hist_bins=hist_bins)
    return out


def histogram_to_sample(summary: Mapping[str, Any], max_points: int = 2000) -> np.ndarray:
    hist = summary.get("hist", {})
    counts = np.asarray(hist.get("counts", []), dtype=int)
    edges = np.asarray(hist.get("edges", []), dtype=float)
    if counts.size == 0 or edges.size != counts.size + 1:
        mean = float(summary.get("mean", 0.0))
        n = int(summary.get("n", 1))
        return np.full(min(max(n, 2), max_points), mean, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = int(np.sum(counts))
    if total <= 0:
        return np.asarray([float(summary.get("mean", 0.0))], dtype=float)
    target = min(max(total, 2), max_points)
    if total <= target:
        sample = np.repeat(centers, counts)
    else:
        expected = counts / total * target
        alloc = np.floor(expected).astype(int)
        remainder = target - int(np.sum(alloc))
        if remainder > 0:
            order = np.argsort(-(expected - alloc))
            alloc[order[:remainder]] += 1
        # Preserve rare non-empty bins where possible.
        if target >= np.count_nonzero(counts):
            zero_nonempty = np.where((counts > 0) & (alloc == 0))[0]
            for idx in zero_nonempty:
                donor_candidates = np.where(alloc > 1)[0]
                if donor_candidates.size == 0:
                    break
                donor = donor_candidates[np.argmax(alloc[donor_candidates])]
                alloc[donor] -= 1
                alloc[idx] = 1
        sample = np.repeat(centers, alloc)
    if sample.size < 2:
        sample = np.asarray([float(summary.get("mean", sample[0] if sample.size else 0.0))] * 2, dtype=float)
    return sample.astype(float)


def hist_counts_for_values(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    edges_arr = np.asarray(edges, dtype=float)
    if edges_arr.size < 2:
        return np.asarray([], dtype=int)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(edges_arr.size - 1, dtype=int)
    # np.histogram drops values outside the supplied range. For drift PSI those
    # tails are signal, so clamp them into the outer baseline buckets.
    bin_idx = np.searchsorted(edges_arr, x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, edges_arr.size - 2)
    return np.bincount(bin_idx, minlength=edges_arr.size - 1).astype(int)


def psi_counts(expected_counts: Sequence[int], current_counts: Sequence[int], eps: float = 1e-6) -> float:
    """Population Stability Index between two count histograms.

    expected_counts is the baseline distribution; current_counts is the observed
    distribution.  Small epsilon smoothing makes the value finite for empty bins.
    """
    expected = np.asarray(expected_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)
    if expected.size == 0 or current.size == 0 or expected.size != current.size:
        return 0.0
    expected = expected + eps
    current = current + eps
    expected = expected / np.sum(expected)
    current = current / np.sum(current)
    return float(np.sum((current - expected) * np.log(current / expected)))


def drift_stat_from_sample(sample: np.ndarray, baseline_summary: Mapping[str, Any], kind: str) -> float:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0
    if kind == "mean_delta":
        return float(abs(np.mean(sample) - float(baseline_summary["mean"])))
    if kind == "p95_delta":
        return float(abs(np.quantile(sample, 0.95) - float(baseline_summary["q95"])))
    if kind == "psi":
        hist = baseline_summary.get("hist", {})
        edges = hist.get("edges", [])
        baseline_counts = hist.get("counts", [])
        current_counts = hist_counts_for_values(sample, edges)
        return psi_counts(baseline_counts, current_counts)
    raise ValueError(f"Unknown drift kind: {kind}")


def drift_numeric_from_snip(current_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any], kind: str) -> float:
    if kind == "mean_delta":
        return float(abs(float(current_summary["mean"]) - float(baseline_summary["mean"])))
    if kind == "p95_delta":
        return float(abs(float(current_summary["q95"]) - float(baseline_summary["q95"])))
    if kind == "psi":
        sample = histogram_to_sample(current_summary, max_points=5000)
        return drift_stat_from_sample(sample, baseline_summary, kind="psi")
    raise ValueError(f"Unknown drift kind: {kind}")


def bootstrap_stat_values(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    B: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("Need at least two finite values for bootstrap")
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(B), dtype=float)
    n = x.size
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        out[b] = float(stat_fn(x[idx]))
    return out[np.isfinite(out)]


def bca_interval(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
    max_jackknife: int = 1000,
) -> Dict[str, Any]:
    """Bias-corrected and accelerated bootstrap confidence interval.

    Returns a dictionary with theta_hat, ci_low and ci_high.  When the BCa
    correction becomes numerically unstable for degenerate data, the function
    falls back to the ordinary percentile interval and marks the method.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        raise ValueError("BCa needs at least two finite observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if B < 100:
        raise ValueError("B should be at least 100 for a stable interval")

    theta_hat = float(stat_fn(x))
    boot = bootstrap_stat_values(x, stat_fn, B=int(B), seed=int(seed))
    if boot.size == 0:
        raise ValueError("Bootstrap produced no finite statistics")

    nd = NormalDist()
    prop_less = float(np.mean(boot < theta_hat))
    prop_less = min(max(prop_less, 1.0 / (boot.size + 1.0)), boot.size / (boot.size + 1.0))
    z0 = nd.inv_cdf(prop_less)

    if n <= max_jackknife:
        jk_indices = np.arange(n)
    else:
        jk_indices = np.unique(np.linspace(0, n - 1, max_jackknife, dtype=int))
    jack = np.empty(jk_indices.size, dtype=float)
    for j, idx in enumerate(jk_indices):
        # For n > max_jackknife this is an approximate jackknife over a
        # deterministic subset; it keeps PR runs bounded.
        jack[j] = float(stat_fn(np.delete(x, idx)))
    jack = jack[np.isfinite(jack)]
    if jack.size < 2:
        acceleration = 0.0
    else:
        jack_mean = float(np.mean(jack))
        diffs = jack_mean - jack
        denom = 6.0 * float(np.sum(diffs ** 2) ** 1.5)
        acceleration = float(np.sum(diffs ** 3) / denom) if denom > 0 else 0.0
        if not math.isfinite(acceleration):
            acceleration = 0.0

    z_low = nd.inv_cdf(alpha / 2.0)
    z_high = nd.inv_cdf(1.0 - alpha / 2.0)

    def adjusted_alpha(z: float) -> float:
        denom = 1.0 - acceleration * (z0 + z)
        if abs(denom) < 1e-12:
            return float("nan")
        return nd.cdf(z0 + (z0 + z) / denom)

    a_low = adjusted_alpha(z_low)
    a_high = adjusted_alpha(z_high)
    method = "bca"
    if not (math.isfinite(a_low) and math.isfinite(a_high) and 0 <= a_low <= 1 and 0 <= a_high <= 1):
        a_low, a_high = alpha / 2.0, 1.0 - alpha / 2.0
        method = "percentile_fallback"
    if a_low > a_high:
        a_low, a_high = a_high, a_low

    ci_low = float(np.quantile(boot, a_low))
    ci_high = float(np.quantile(boot, a_high))
    return {
        "theta_hat": theta_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": float(alpha),
        "B": int(B),
        "seed": int(seed),
        "method": method,
        "z0": float(z0),
        "acceleration": float(acceleration),
        "bootstrap_std": float(np.std(boot, ddof=1)) if boot.size > 1 else 0.0,
        "n": n,
    }


def _load_feature_vectors_from_jsonl(path: Path) -> List[List[float]]:
    vectors: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, Mapping):
                continue
            for key in ("embedding", "embeddings", "feature", "features", "vector", "vectors"):
                value = obj.get(key)
                if isinstance(value, list) and value and all(_maybe_float(v) is not None for v in value):
                    vectors.append([float(v) for v in value])
                    break
    return vectors


def load_feature_matrix(paths: Sequence[Path]) -> Optional[np.ndarray]:
    arrays: List[np.ndarray] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim == 2 and arr.size:
                arrays.append(arr)
        elif suffix == ".csv":
            rows: List[List[float]] = []
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                numeric_fields = [name for name in reader.fieldnames if name and name.lower() not in TIMESTAMP_KEYS]
                for row in reader:
                    values = [_maybe_float(row.get(name)) for name in numeric_fields]
                    if values and all(v is not None for v in values):
                        rows.append([float(v) for v in values if v is not None])
            if rows:
                arrays.append(np.asarray(rows, dtype=float))
        elif suffix in {".jsonl", ".ndjson"}:
            vectors = _load_feature_vectors_from_jsonl(path)
            if vectors:
                width = len(vectors[0])
                vectors = [v for v in vectors if len(v) == width]
                arrays.append(np.asarray(vectors, dtype=float))
    if not arrays:
        return None
    # Keep only arrays with the dominant feature dimension.
    dims: Dict[int, int] = {}
    for arr in arrays:
        dims[arr.shape[1]] = dims.get(arr.shape[1], 0) + arr.shape[0]
    dominant_dim = max(dims.items(), key=lambda kv: kv[1])[0]
    compatible = [arr for arr in arrays if arr.shape[1] == dominant_dim]
    X = np.vstack(compatible)
    X = X[np.all(np.isfinite(X), axis=1)]
    return X if X.size else None


def _deterministic_sample_rows(X: np.ndarray, max_rows: int) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    idx = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
    return X[idx]


def _project_features(X: np.ndarray, max_dim: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    n, d = X.shape
    if d <= max_dim:
        return X, {"kind": "identity", "original_dim": int(d), "projected_dim": int(d)}
    rng = np.random.default_rng(int(seed))
    R = rng.normal(loc=0.0, scale=1.0 / math.sqrt(max_dim), size=(d, max_dim))
    return X @ R, {
        "kind": "gaussian_random_projection",
        "original_dim": int(d),
        "projected_dim": int(max_dim),
        "seed": int(seed),
    }


def _kmeans(X: np.ndarray, k: int, iterations: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(int(seed))
    # Deterministic-with-seed initialization.  Sorting keeps identical seeds
    # stable across platforms for the same n.
    init_idx = np.sort(rng.choice(n, size=k, replace=False))
    centroids = X[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(int(iterations)):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return centroids, labels


def feature_snip_from_matrix(
    X: np.ndarray,
    k: int = 8,
    seed: int = 12345,
    max_rows: int = 5000,
    max_dim: int = 64,
    iterations: int = 20,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 2:
        raise ValueError("Need at least two feature rows")
    original_n, original_dim = X.shape
    X_small = _deterministic_sample_rows(X, max_rows=max_rows)
    X_proj, projection = _project_features(X_small, max_dim=max_dim, seed=seed)
    centroids, labels = _kmeans(X_proj, k=k, iterations=iterations, seed=seed)
    counts = np.bincount(labels, minlength=centroids.shape[0]).astype(int)
    cmin = float(np.min(centroids))
    cmax = float(np.max(centroids))
    if cmin == cmax:
        scale = 1.0
        q = np.zeros_like(centroids, dtype=np.uint8)
    else:
        scale = (cmax - cmin) / 255.0
        q = np.clip(np.round((centroids - cmin) / scale), 0, 255).astype(np.uint8)
    return {
        "n": int(original_n),
        "sampled_n": int(X_small.shape[0]),
        "original_dim": int(original_dim),
        "projected_dim": int(X_proj.shape[1]),
        "k": int(centroids.shape[0]),
        "projection": projection,
        "quantization": {"dtype": "uint8", "min": cmin, "scale": float(scale)},
        "centroids_q8": q.astype(int).tolist(),
        "assignment_hist": counts.tolist(),
    }


def dequantize_centroids(feature_summary: Mapping[str, Any]) -> np.ndarray:
    q = np.asarray(feature_summary.get("centroids_q8", []), dtype=float)
    quant = feature_summary.get("quantization", {})
    cmin = float(quant.get("min", 0.0))
    scale = float(quant.get("scale", 1.0))
    if q.ndim != 2 or q.size == 0:
        return np.empty((0, 0), dtype=float)
    return cmin + q * scale


def _weighted_avg_min_distance(A: np.ndarray, B: np.ndarray, weights: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return 0.0
    distances = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    mins = np.min(distances, axis=1)
    weights = weights.astype(float)
    if weights.size != mins.size or np.sum(weights) <= 0:
        weights = np.ones_like(mins)
    return float(np.sum(mins * weights) / np.sum(weights))


def feature_centroid_drift(current_features: Mapping[str, Any], baseline_features: Mapping[str, Any]) -> float:
    A = dequantize_centroids(current_features)
    B = dequantize_centroids(baseline_features)
    if A.size == 0 or B.size == 0:
        return 0.0
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Feature dimensions differ: current={A.shape[1]}, baseline={B.shape[1]}")
    wA = np.asarray(current_features.get("assignment_hist", [1] * A.shape[0]), dtype=float)
    wB = np.asarray(baseline_features.get("assignment_hist", [1] * B.shape[0]), dtype=float)
    dim_norm = math.sqrt(max(A.shape[1], 1))
    symmetric = 0.5 * (_weighted_avg_min_distance(A, B, wA) + _weighted_avg_min_distance(B, A, wB))
    return float(symmetric / dim_norm)


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(base_seed), *(str(p) for p in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def infer_repo_name(path_or_name: str) -> str:
    text = str(path_or_name).strip()
    if not text:
        return Path.cwd().name
    if "/" in text and not Path(text).exists():
        return text
    return Path(text).resolve().name if Path(text).exists() else text
