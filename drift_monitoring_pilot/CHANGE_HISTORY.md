# Change History

## 2026-06-16 - Hardened drift monitoring pilot

### Summary

Hardened the drop-in drift monitoring pilot by separating historical calibration
data from current PR data, fixing PSI tail handling, clarifying privacy language,
and adding focused validation coverage.

### Interface and workflow changes

- Added `tools/calibrate_bca.py --history-input`, repeatable per calibration
  run. Relative paths are resolved under each `--repo`; omitting the option keeps
  the previous repo-root auto-discovery behavior for local experiments.
- Updated `.github/workflows/calibrate-thresholds.yml` to calibrate from
  `monitoring/history`.
- Updated `.github/workflows/pr-drift-check.yml` to require
  `monitoring/current`, then build PR snips with:

  ```bash
  python -u tools/make_prototype_snip.py \
    --input "$CURRENT" \
    --features "$CURRENT" \
    --out artifacts/pr-snip.json.gz \
    --max-bytes 32768
  ```

- Documented the intended drop-in install shape: copy `.github/workflows/*`,
  `tools/`, and `requirements.txt` to the target repository root.

### Behavior changes

- `tools/drift_utils.py::hist_counts_for_values(...)` now clamps values outside
  the baseline histogram edge range into the first or last bucket. PSI therefore
  keeps tail drift signal instead of silently dropping out-of-range current
  values.
- `tools/pr_check.py` still reports feature checks as prototype point estimates,
  but the Markdown report now states that no PR confidence interval is computed
  because PR snips do not contain raw feature rows.
- `README.md` now describes artifacts as data-minimized and aggregated rather
  than formally anonymized, and calls out the privacy caveat for small datasets
  or high prototype counts.

### Tests added

- `tests/test_drift_utils.py`
  - Verifies that PSI histogram counts clamp out-of-range values into edge
    buckets.
  - Verifies that numeric snip creation and histogram pseudo-sample
    reconstruction remain stable for normal in-range data.
- `tests/test_smoke_flow.py`
  - Builds a temporary repo with `monitoring/history` and `monitoring/current`.
  - Runs `tools/calibrate_bca.py --history-input monitoring/history`.
  - Runs `tools/make_prototype_snip.py` against `monitoring/current`.
  - Runs `tools/pr_check.py` against the generated calibration and snip.

### Validation run

The implementation was validated locally from `drift_monitoring_pilot/` with:

```bash
python3 -m py_compile tools/*.py tests/*.py
python3 -m unittest discover -s tests
```

Both commands completed successfully.

### Caveats

- The pilot directory is still a drop-in package. GitHub Actions will only run
  the workflow files after they are copied to `.github/workflows` at the target
  repository root.
- The feature drift PR check remains point-estimate-only. Numeric checks use BCa
  intervals and alarm on `ci_high > threshold`.
- No long-running production calibration was executed for this change history;
  validation used local unit and smoke tests only.
