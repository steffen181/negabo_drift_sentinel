# Drift monitoring pilot

This is a tiny GitHub-Actions-friendly drift monitor. It creates small hashed
snips from repo-local metrics/features, calibrates conservative numeric BCa
thresholds, and fails PRs when the upper confidence bound for a numeric drift
check exceeds the calibrated threshold.

The artifacts are data-minimized and aggregated: they contain summaries,
histograms, hashes and optional quantized feature prototypes, not raw rows or
raw feature vectors. They are not a formal anonymization guarantee, especially
for small datasets or high prototype counts.

## Installation

This directory is a drop-in package. To make the sample GitHub Actions run in a
target repository, copy these paths to that repository root:

- `.github/workflows/*`
- `tools/`
- `requirements.txt`

Workflows nested below another directory are templates only; GitHub Actions runs
workflow files from `.github/workflows` at the repository root.

## Expected data

Keep historical calibration data separate from current PR data:

- `monitoring/history/` for representative historical metrics/features
- `monitoring/current/` for CI-generated current PR metrics/features
- `monitoring/pilot-calibration.json.gz` for the committed or downloaded
  calibration artifact used by PR checks

The scripts read monitoring-ish files such as:

- `monitoring/history/metrics.jsonl`
- `monitoring/current/metrics.csv`
- `monitoring/history/features.npy`
- JSONL rows with numeric scalar fields and optional `timestamp`
- CSV files with numeric columns and optional `timestamp`

A JSONL metric row can look like this:

```json
{"timestamp":"2026-06-14T12:00:00Z","error_rate":0.021,"latency_ms":142,"quality_score":0.88}
```

Feature files can be `.npy` arrays of shape `(n, d)` or JSONL rows with an
`embedding`, `features` or `vector` list.

## Smoke test

```bash
pip install -r requirements.txt
python tools/calibrate_bca.py --demo --out artifacts/pilot-calibration.json.gz --B 500 --feature-B 50
python tools/make_prototype_snip.py --input _demo_data/demo_repo/monitoring/history --features _demo_data/demo_repo/monitoring/history --repo demo_repo --out artifacts/pr-snip.json.gz
python tools/pr_check.py --calib artifacts/pilot-calibration.json.gz --snip artifacts/pr-snip.json.gz --B 500
```

## Production flow

1. Write representative historical data under `monitoring/history/`.
2. Run `calibrate-thresholds` manually on that historical data. The sample
   workflow calls:

   ```bash
   python -u tools/calibrate_bca.py \
     --repos "." \
     --history-input monitoring/history \
     --window-days 30 \
     --min-samples 30 \
     --B 2000 \
     --out artifacts/pilot-calibration.json.gz
   ```

3. Put `pilot-calibration.json.gz` somewhere the PR workflow can read it.  The
   sample workflow expects it at `monitoring/pilot-calibration.json.gz`.
4. Before the PR drift step, generate current PR metrics/features under
   `monitoring/current/`.
5. The PR workflow creates `artifacts/pr-snip.json.gz` from `monitoring/current`,
   compares it, and exits 1 if any numeric conservative bound exceeds the
   threshold.

Numeric checks use BCa intervals and alarm on `ci_high > threshold`. Feature
checks use quantized prototype point estimates because PR snips do not contain
raw feature rows; their report is marked as point-estimate-only.

For quick local experiments, `calibrate_bca.py` still defaults to scanning the
repo root when `--history-input` is omitted. Production workflows should pass an
explicit history path.
