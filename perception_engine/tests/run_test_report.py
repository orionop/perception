#!/usr/bin/env python3
"""
Test report runner with scored rubric.

Runs all tests via pytest, parses results, and prints a scored report
aligned with the test plan's rubric.

Usage:
    python perception_engine/tests/run_test_report.py
"""

import subprocess
import sys
import re
from collections import defaultdict
from pathlib import Path


# Module → (max_unit, max_edge, max_integration) scoring caps.
MODULE_RUBRIC = {
    "data_types":            {"unit": 3, "edge": 3, "integration": 3},
    "config_loader":         {"unit": 3, "edge": 3, "integration": 3},
    "cost_mapping":          {"unit": 3, "edge": 3, "integration": 3},
    "mask_remapping":        {"unit": 3, "edge": 3, "integration": 3},
    "planner":               {"unit": 3, "edge": 3, "integration": 3},
    "segmentation_metrics":  {"unit": 3, "edge": 3, "integration": 3},
    "robustness":            {"unit": 3, "edge": 3, "integration": 3},
    "safety":                {"unit": 3, "edge": 3, "integration": 3},
    "end_to_end":            {"unit": 0, "edge": 0, "integration": 3},
}

# Map test file names → rubric module.
FILE_TO_MODULE = {
    "test_data_types":            "data_types",
    "test_config_loader":         "config_loader",
    "test_cost_mapping":          "cost_mapping",
    "test_mask_remapping":        "mask_remapping",
    "test_planner":               "planner",
    "test_segmentation_metrics":  "segmentation_metrics",
    "test_robustness":            "robustness",
    "test_safety":                "safety",
    "test_integration":           "integration",  # spreads across modules
    "test_end_to_end":            "end_to_end",
}

# Integration tests cover multiple modules — credit them.
INTEGRATION_CREDITS = {
    "test_mask_to_costmap_to_planner": ["cost_mapping", "planner"],
    "test_costmap_to_safety":          ["cost_mapping", "safety"],
    "test_metrics_with_remapped_gt":   ["mask_remapping", "segmentation_metrics"],
    "test_planner_on_mixed_cost_grid": ["cost_mapping", "planner"],
    "test_robustness_perturbation_integration": ["robustness"],
    "test_robustness_unknown_perturbation_skipped": ["robustness"],
    "test_full_pipeline_synthetic":    ["data_types", "config_loader", "cost_mapping",
                                        "planner", "segmentation_metrics", "safety"],
    "test_benchmark_two_models":       ["data_types", "config_loader", "cost_mapping",
                                        "planner", "segmentation_metrics"],
    "test_visualization_outputs_saved": ["cost_mapping", "planner"],
    "test_e2e_with_robustness_enabled": ["robustness", "data_types", "config_loader"],
    "test_safety_with_diagonal_planner": ["safety"],
    "test_remap_then_costmap": ["mask_remapping", "cost_mapping"],
    "test_remap_preserves_spatial_structure": ["mask_remapping"],
}


def run_tests():
    """Run pytest and capture line-level output."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "perception_engine/tests/", "-v",
         "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    return result.stdout + result.stderr, result.returncode


def parse_results(output: str):
    """Parse pytest verbose output into (file, test, status) tuples."""
    pattern = re.compile(
        r"perception_engine/tests/(\w+)\.py::(\w+)\s+(PASSED|FAILED|ERROR)"
    )
    results = []
    for m in pattern.finditer(output):
        results.append((m.group(1), m.group(2), m.group(3)))
    return results


def compute_scores(results):
    """Compute per-module scores based on the rubric."""
    module_pass = defaultdict(int)
    module_fail = defaultdict(int)
    module_total = defaultdict(int)
    integration_pass = defaultdict(set)

    for file_name, test_name, status in results:
        module = FILE_TO_MODULE.get(file_name)
        if module and module not in ("integration", "end_to_end"):
            module_total[module] += 1
            if status == "PASSED":
                module_pass[module] += 1
            else:
                module_fail[module] += 1

        # Credit integration / E2E passes.
        if test_name in INTEGRATION_CREDITS and status == "PASSED":
            for mod in INTEGRATION_CREDITS[test_name]:
                integration_pass[mod].add(test_name)

    scores = {}
    for mod, rubric in MODULE_RUBRIC.items():
        p = module_pass.get(mod, 0)
        t = module_total.get(mod, 0)

        # Unit score: up to 3 based on pass rate.
        if rubric["unit"] > 0 and t > 0:
            unit = min(3, int(3 * p / t + 0.5))
        else:
            unit = 0

        # Edge score: 3 if ≥ 3 tests pass, 2 if ≥ 2, etc.
        edge = min(3, max(0, p - 1)) if rubric["edge"] > 0 else 0

        # Integration score: based on integration tests passing.
        integ_count = len(integration_pass.get(mod, set()))
        integ = min(3, integ_count) if rubric["integration"] > 0 else 0

        # E2E is pure integration.
        if mod == "end_to_end":
            e2e_pass = sum(
                1 for f, n, s in results
                if f == "test_end_to_end" and s == "PASSED"
            )
            unit, edge, integ = 0, 0, min(3, e2e_pass)

        scores[mod] = {
            "unit": unit,
            "edge": edge,
            "integration": integ,
            "total": unit + edge + integ,
            "max": rubric["unit"] + rubric["edge"] + rubric["integration"],
        }

    return scores


def print_report(results, scores, raw_output, returncode):
    """Print the scored test report."""
    total_pass = sum(1 for _, _, s in results if s == "PASSED")
    total_fail = sum(1 for _, _, s in results if s != "PASSED")
    total = len(results)

    print("=" * 76)
    print("  PERCEPTION EVALUATION ENGINE — TEST REPORT")
    print("=" * 76)
    print()

    # Summary.
    status = "✅ ALL PASSED" if total_fail == 0 else f"❌ {total_fail} FAILED"
    print(f"  Tests Run:    {total}")
    print(f"  Passed:       {total_pass}")
    print(f"  Failed:       {total_fail}")
    print(f"  Status:       {status}")
    print()

    # Per-file breakdown.
    print("-" * 76)
    print(f"  {'Test File':<35} {'Passed':>8} {'Failed':>8} {'Total':>8}")
    print("-" * 76)

    file_stats = defaultdict(lambda: {"pass": 0, "fail": 0})
    for f, _, s in results:
        if s == "PASSED":
            file_stats[f]["pass"] += 1
        else:
            file_stats[f]["fail"] += 1

    for f in sorted(file_stats.keys()):
        p = file_stats[f]["pass"]
        fl = file_stats[f]["fail"]
        t = p + fl
        icon = "✅" if fl == 0 else "❌"
        print(f"  {icon} {f:<33} {p:>8} {fl:>8} {t:>8}")

    print("-" * 76)
    print()

    # Rubric scorecard.
    print("=" * 76)
    print("  SCORED RUBRIC")
    print("=" * 76)
    print()
    print(f"  {'Module':<25} {'Unit':>6} {'Edge':>6} {'Integ':>6} "
          f"{'Score':>7} {'Max':>5} {'Grade':>7}")
    print("  " + "-" * 70)

    grand_total = 0
    grand_max = 0

    for mod in MODULE_RUBRIC:
        s = scores[mod]
        pct = (s["total"] / s["max"] * 100) if s["max"] > 0 else 0
        grade = _grade(pct)
        print(
            f"  {mod:<25} {s['unit']:>6}/3 {s['edge']:>6}/3 "
            f"{s['integration']:>6}/3 {s['total']:>5}/{s['max']:<3} {grade:>7}"
        )
        grand_total += s["total"]
        grand_max += s["max"]

    print("  " + "-" * 70)
    grand_pct = (grand_total / grand_max * 100) if grand_max > 0 else 0
    print(
        f"  {'TOTAL':<25} {'':>6} {'':>6} {'':>6} "
        f"{grand_total:>5}/{grand_max:<3} {_grade(grand_pct):>7}"
    )
    print()
    print(f"  Overall Score: {grand_total}/{grand_max} ({grand_pct:.1f}%)")
    print(f"  Overall Grade: {_grade(grand_pct)}")
    print()

    # Failed tests detail.
    failed = [(f, n) for f, n, s in results if s != "PASSED"]
    if failed:
        print("=" * 76)
        print("  FAILED TESTS")
        print("=" * 76)
        for f, n in failed:
            print(f"  ✗ {f}::{n}")
        print()

    print("=" * 76)


def _grade(pct: float) -> str:
    if pct >= 90:
        return "A"
    if pct >= 80:
        return "B"
    if pct >= 70:
        return "C"
    if pct >= 60:
        return "D"
    return "F"


if __name__ == "__main__":
    raw_output, returncode = run_tests()
    results = parse_results(raw_output)
    scores = compute_scores(results)
    print_report(results, scores, raw_output, returncode)
    sys.exit(0 if returncode == 0 else 1)
