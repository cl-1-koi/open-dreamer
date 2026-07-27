"""Tests for the build wrapper's telemetry parsing and summarization.

Pure-function tests only: nothing here launches Docker or builds an image.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


wrapper = load_module(
    "build_coinrun_runner", REPO_ROOT / "scripts" / "build_coinrun_runner.py"
)


class StageDurationParsingTests(unittest.TestCase):
    LOG = """
#5 [builder 2/10] RUN apt-get update
#5 DONE 149.2s

#13 [builder  6/10] COPY pyproject.toml uv.lock README.md ./
#13 CACHED

#14 [builder  7/10] RUN uv sync --frozen
#14 25.06 Prepared 126 packages in 24.85s
#14 DONE 34.9s

#24 exporting to image
#24 DONE 12.0s
"""

    def test_named_steps_report_their_seconds(self):
        durations = wrapper.parse_stage_durations(self.LOG)
        self.assertAlmostEqual(durations["builder 2/10"], 149.2)
        self.assertAlmostEqual(durations["builder  7/10"], 34.9)

    def test_cached_steps_are_recorded_as_zero_not_dropped(self):
        durations = wrapper.parse_stage_durations(self.LOG)
        self.assertEqual(durations["builder  6/10"], 0.0)

    def test_unnamed_steps_fall_back_to_a_step_label(self):
        self.assertAlmostEqual(wrapper.parse_stage_durations(self.LOG)["step-24"], 12.0)

    def test_progress_noise_lines_are_ignored(self):
        durations = wrapper.parse_stage_durations(self.LOG)
        self.assertNotIn("step-14", durations)  # the "25.06 Prepared" line is not a DONE

    def test_empty_log_yields_no_durations(self):
        self.assertEqual(wrapper.parse_stage_durations(""), {})


class ResourceSampleTests(unittest.TestCase):
    def test_cpu_utilization_is_derived_from_idle_share(self):
        # 1000 jiffies elapsed, 250 idle -> 75% busy.
        row = wrapper.derive_sample(
            {"cpu_total": 1000.0, "cpu_idle": 500.0},
            {"cpu_total": 2000.0, "cpu_idle": 750.0},
            5.0,
        )
        self.assertAlmostEqual(row["cpu_utilization_percent"], 75.0)

    def test_counter_deltas_become_per_second_rates(self):
        row = wrapper.derive_sample(
            {"net_rx_bytes": 0.0, "disk_write_bytes": 0.0},
            {"net_rx_bytes": 500.0, "disk_write_bytes": 1000.0},
            5.0,
        )
        self.assertAlmostEqual(row["net_rx_bytes_per_s"], 100.0)
        self.assertAlmostEqual(row["disk_write_bytes_per_s"], 200.0)

    def test_absent_counters_stay_absent_rather_than_zero(self):
        row = wrapper.derive_sample({}, {"load_average_1m": 0.25}, 5.0)
        self.assertEqual(row["load_average_1m"], 0.25)
        for missing in ("cpu_utilization_percent", "net_rx_bytes_per_s", "disk_read_bytes_per_s"):
            self.assertNotIn(missing, row)

    def test_zero_interval_does_not_divide_by_zero(self):
        row = wrapper.derive_sample({"net_rx_bytes": 0.0}, {"net_rx_bytes": 5.0}, 0.0)
        self.assertNotIn("net_rx_bytes_per_s", row)

    def test_every_row_is_timestamped(self):
        row = wrapper.derive_sample({}, {}, 5.0)
        self.assertTrue(row["utc"].endswith("Z"))

    def test_live_counter_read_is_portable_and_non_fatal(self):
        # Whatever this host provides, reading must not raise.
        counters = wrapper.read_counters()
        self.assertIsInstance(counters, dict)


class ResourceSummaryTests(unittest.TestCase):
    ROWS = [
        {"cpu_utilization_percent": 10.0, "net_rx_bytes_per_s": 100.0,
         "mem_available_bytes": 900.0, "load_average_1m": 0.5},
        {"cpu_utilization_percent": 30.0, "net_rx_bytes_per_s": 300.0,
         "mem_available_bytes": 700.0, "load_average_1m": 1.5},
    ]

    def test_mean_and_peak_are_reported_for_utilization(self):
        summary = wrapper.summarize_resource_samples(self.ROWS)
        self.assertEqual(summary["cpu_utilization_percent_mean"], 20.0)
        self.assertEqual(summary["cpu_utilization_percent_peak"], 30.0)

    def test_network_mean_and_peak_distinguish_a_download_bound_build(self):
        summary = wrapper.summarize_resource_samples(self.ROWS)
        self.assertEqual(summary["net_rx_bytes_per_s_mean"], 200.0)
        self.assertEqual(summary["net_rx_bytes_per_s_peak"], 300.0)

    def test_memory_pressure_is_reported_as_a_minimum(self):
        summary = wrapper.summarize_resource_samples(self.ROWS)
        self.assertEqual(summary["mem_available_bytes_min"], 700.0)

    def test_sample_count_is_always_present(self):
        self.assertEqual(wrapper.summarize_resource_samples([])["sample_count"], 0)

    def test_missing_metrics_are_omitted_not_zeroed(self):
        summary = wrapper.summarize_resource_samples([{"load_average_1m": 1.0}])
        self.assertNotIn("cpu_utilization_percent_mean", summary)
        self.assertEqual(summary["load_average_1m_peak"], 1.0)


class ObservationsCsvTests(unittest.TestCase):
    PATH = REPO_ROOT / "docs" / "coinrun_runner_build_observations.csv"

    def rows(self):
        lines = [
            line for line in self.PATH.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        return list(csv.DictReader(lines))

    def test_historical_rows_are_labelled_reconstructed_not_measured(self):
        rows = self.rows()
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["source"], "reconstructed")
            self.assertTrue(row["provenance"].strip(), "each row must say where its times came from")

    def test_unknown_timings_are_blank_rather_than_invented(self):
        # The canceled/terminated rows have no reliable timestamps; they must be
        # empty rather than estimated.
        outcomes = {row["outcome"] for row in self.rows()}
        self.assertIn("canceled", outcomes)
        self.assertIn("terminated", outcomes)
        for row in self.rows():
            if row["outcome"] in ("canceled", "terminated") and not row["provenance"].startswith("end"):
                self.assertEqual(row["duration_seconds"], "")

    def test_the_size_regression_and_its_fix_are_both_recorded(self):
        sizes = {row["image_id"]: row["image_size"] for row in self.rows() if row["image_id"]}
        self.assertEqual(sizes.get("e40264b07eb9"), "45.2GB")
        self.assertIn("0610c1953e26", sizes)


if __name__ == "__main__":
    unittest.main()
