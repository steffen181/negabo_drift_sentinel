import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SmokeFlowTests(unittest.TestCase):
    def test_explicit_history_and_current_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "demo_repo"
            history = repo / "monitoring" / "history"
            current = repo / "monitoring" / "current"
            history.mkdir(parents=True)
            current.mkdir(parents=True)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            rng = np.random.default_rng(12345)
            rows = []

            with (history / "metrics.jsonl").open("w", encoding="utf-8") as f:
                for day in range(90):
                    base_error = 0.025 + 0.003 * np.sin(day / 7.0)
                    base_latency = 150 + 8 * np.sin(day / 10.0)
                    for j in range(10):
                        ts = now - timedelta(days=90 - day, hours=j)
                        row = {
                            "timestamp": ts.isoformat().replace("+00:00", "Z"),
                            "error_rate": float(np.clip(rng.normal(base_error, 0.006), 0.0, 1.0)),
                            "latency_ms": float(np.clip(rng.normal(base_latency, 20), 1.0, None)),
                        }
                        rows.append((ts, row))
                        f.write(json.dumps(row) + "\n")

            with (current / "metrics.jsonl").open("w", encoding="utf-8") as f:
                latest_rows = [row for ts, row in rows if ts >= now - timedelta(days=30)]
                for j, row in enumerate(latest_rows):
                    current_row = dict(row)
                    ts = now - timedelta(hours=len(latest_rows) - j)
                    current_row["timestamp"] = ts.isoformat().replace("+00:00", "Z")
                    f.write(json.dumps(current_row) + "\n")

            calib = tmp_path / "pilot-calibration.json.gz"
            snip = tmp_path / "pr-snip.json.gz"
            report = tmp_path / "pr-drift-report.json"
            common = {"cwd": ROOT, "check": True, "text": True, "capture_output": True}

            subprocess.run(
                [
                    sys.executable,
                    "tools/calibrate_bca.py",
                    "--repo",
                    str(repo),
                    "--history-input",
                    "monitoring/history",
                    "--out",
                    str(calib),
                    "--min-samples",
                    "10",
                    "--B",
                    "200",
                    "--feature-B",
                    "50",
                ],
                **common,
            )
            subprocess.run(
                [
                    sys.executable,
                    "tools/make_prototype_snip.py",
                    "--input",
                    str(current),
                    "--features",
                    str(current),
                    "--repo",
                    "demo_repo",
                    "--out",
                    str(snip),
                ],
                **common,
            )
            subprocess.run(
                [
                    sys.executable,
                    "tools/pr_check.py",
                    "--calib",
                    str(calib),
                    "--snip",
                    str(snip),
                    "--B",
                    "200",
                    "--report",
                    str(report),
                ],
                **common,
            )

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["alarm"])
            self.assertIn("error_rate", payload["numeric"])


if __name__ == "__main__":
    unittest.main()
