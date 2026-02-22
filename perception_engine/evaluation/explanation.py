"""
Human-readable explanation generator.

Translates raw ``BenchmarkReport`` metrics into plain-English narratives
that a non-technical user can understand. The output is both printed to
stdout and saved as a text/JSON file for UI consumption.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from perception_engine.core.data_types import BenchmarkReport

logger = logging.getLogger(__name__)


def generate_explanation(
    reports: List[BenchmarkReport],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a structured, human-readable explanation of results.

    Args:
        reports: List of benchmark reports (one per model).
        config: Experiment configuration for context.

    Returns:
        A dict containing explanation sections, suitable for JSON
        serialization or UI rendering.
    """
    class_names = None
    if config and "class_names" in config:
        names = config["class_names"]
        if isinstance(names, list):
            class_names = {i: n for i, n in enumerate(names)}
        else:
            class_names = names

    explanation: Dict[str, Any] = {
        "title": "Perception Engine — Results Explained",
        "overview": _overview(reports),
        "models": [],
    }

    for report in reports:
        model_expl = _explain_model(report, class_names)
        explanation["models"].append(model_expl)

    if len(reports) > 1:
        explanation["comparison"] = _explain_comparison(reports)

    return explanation


def save_explanation(
    explanation: Dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save the explanation as both JSON and formatted text.

    Args:
        explanation: Explanation dict from ``generate_explanation``.
        output_dir: Directory to save files into.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON (for UI consumption).
    json_path = out / "explanation.json"
    with open(json_path, "w") as f:
        json.dump(explanation, f, indent=2, default=_json_default)

    # Human-readable text.
    txt_path = out / "explanation.txt"
    with open(txt_path, "w") as f:
        f.write(_render_text(explanation))

    logger.info("Explanation saved to %s and %s", json_path, txt_path)
    return json_path


def print_explanation(explanation: Dict[str, Any]) -> None:
    """Print the explanation to stdout in a readable format."""
    print(_render_text(explanation))


# ------------------------------------------------------------------
# Per-model explanation
# ------------------------------------------------------------------

def _explain_model(
    report: BenchmarkReport,
    class_names: Optional[Dict[int, str]],
) -> Dict[str, Any]:
    """Generate explanation sections for a single model."""
    seg = report.segmentation_output
    nav = report.navigation_result
    safety = report.safety_report
    metrics = report.metrics

    expl: Dict[str, Any] = {"model_name": report.model_name, "sections": []}

    # 1. Segmentation quality.
    if metrics:
        miou = metrics.get("mean_iou", 0)
        pix_acc = metrics.get("pixel_accuracy", 0)

        quality = _quality_label(miou)
        expl["sections"].append({
            "title": "How well does the model understand the scene?",
            "metric": f"{miou * 100:.1f}% mean IoU",
            "explanation": (
                f"The model correctly identified {pix_acc * 100:.1f}% of all pixels. "
                f"Its overall scene understanding score (mean IoU) is "
                f"{miou * 100:.1f}%, which is considered **{quality}**. "
                f"A perfect model scores 100%."
            ),
        })

        # Per-class breakdown.
        pci = metrics.get("per_class_iou", {})
        if pci:
            best_cls = max(pci, key=pci.get)
            worst_cls = min(pci, key=pci.get)
            expl["sections"].append({
                "title": "Which terrain types does it recognise best?",
                "best_class": f"{best_cls} ({pci[best_cls] * 100:.1f}%)",
                "worst_class": f"{worst_cls} ({pci[worst_cls] * 100:.1f}%)",
                "explanation": (
                    f"The model is most accurate at detecting **{best_cls}** "
                    f"({pci[best_cls] * 100:.1f}% IoU) and struggles most with "
                    f"**{worst_cls}** ({pci[worst_cls] * 100:.1f}% IoU). "
                    f"Classes with low IoU are often small, rare, or visually "
                    f"similar to other terrain."
                ),
                "all_classes": {
                    name: f"{iou * 100:.1f}%" for name, iou in pci.items()
                },
            })

    # 2. Speed.
    expl["sections"].append({
        "title": "How fast is the model?",
        "metric": f"{seg.inference_time_ms:.1f} ms per image",
        "explanation": (
            f"The model takes {seg.inference_time_ms:.1f} milliseconds "
            f"to process one image. "
            + _speed_context(seg.inference_time_ms)
        ),
    })

    # 3. Confidence.
    avg_conf = float(seg.confidence_map.mean())
    expl["sections"].append({
        "title": "How confident is the model in its predictions?",
        "metric": f"{avg_conf * 100:.1f}% average confidence",
        "explanation": (
            f"On average, the model is {avg_conf * 100:.1f}% sure about "
            f"its pixel-level predictions. "
            + (
                "This is a good level of certainty."
                if avg_conf >= 0.7
                else "This is relatively low — the model is uncertain about "
                     "many areas, which could affect navigation reliability."
            )
        ),
    })

    # 4. Navigation.
    if nav.path_found:
        expl["sections"].append({
            "title": "Can the robot find a safe path?",
            "metric": "✅ Path found",
            "explanation": (
                f"The A* planner successfully found a route through the scene. "
                f"The path is {len(nav.path)} steps long with a total "
                f"traversal cost of {nav.path_cost:.1f}."
            ),
        })
    else:
        expl["sections"].append({
            "title": "Can the robot find a safe path?",
            "metric": "❌ No path found",
            "explanation": (
                "The planner could not find any path from start to goal. "
                "This likely means too many obstacles are blocking the way, "
                "or the model is misclassifying traversable terrain as obstacles."
            ),
        })

    # 5. Safety.
    if safety:
        expl["sections"].append({
            "title": "How safe is the planned route?",
            "metric": f"{safety.safety_score * 100:.1f}% safety score",
            "explanation": (
                f"The safety score is {safety.safety_score * 100:.1f}% "
                f"(100% = perfectly safe). "
                f"{safety.obstacle_overlap_pct:.1f}% of the path overlaps "
                f"with obstacles. "
                + _safety_verdict(safety.safety_score)
            ),
        })

    # 6. Robustness.
    if report.robustness:
        worst_pert = max(
            report.robustness,
            key=lambda p: report.robustness[p].get("miou_drop", 0),
        )
        worst_drop = report.robustness[worst_pert].get("miou_drop", 0)

        pert_details = {}
        for p, d in report.robustness.items():
            drop = d.get("miou_drop", 0)
            pert_details[p] = {
                "drop": f"{drop * 100:.1f}%",
                "verdict": (
                    "Minimal impact" if drop < 0.05
                    else "Moderate degradation" if drop < 0.15
                    else "Significant drop — model is sensitive to this"
                ),
            }

        expl["sections"].append({
            "title": "How does the model handle tough conditions?",
            "metric": f"Worst impact: {worst_pert} (−{worst_drop * 100:.1f}%)",
            "explanation": (
                f"We tested the model under difficult conditions like "
                f"changes in brightness, blur, noise, and contrast. "
                f"The biggest accuracy drop came from **{worst_pert}** "
                f"(−{worst_drop * 100:.1f}% IoU). "
                + (
                    "Overall, the model is quite robust to these disturbances."
                    if worst_drop < 0.10
                    else "The model shows some sensitivity — this could be "
                         "a concern in real-world conditions."
                )
            ),
            "perturbations": pert_details,
        })

    return expl


# ------------------------------------------------------------------
# Multi-model comparison
# ------------------------------------------------------------------

def _explain_comparison(reports: List[BenchmarkReport]) -> Dict[str, Any]:
    """Compare models in plain language."""
    scored = []
    for r in reports:
        miou = r.metrics.get("mean_iou", 0) if r.metrics else 0
        scored.append((r.model_name, miou, r.segmentation_output.inference_time_ms))

    best = max(scored, key=lambda x: x[1])
    fastest = min(scored, key=lambda x: x[2])

    return {
        "title": "Model Comparison",
        "best_accuracy": f"{best[0]} ({best[1] * 100:.1f}% mIoU)",
        "fastest": f"{fastest[0]} ({fastest[2]:.1f} ms)",
        "explanation": (
            f"Among the {len(reports)} models tested, **{best[0]}** achieved "
            f"the highest accuracy ({best[1] * 100:.1f}% mIoU)"
            + (
                f" and **{fastest[0]}** was the fastest ({fastest[2]:.1f} ms). "
                if fastest[0] != best[0]
                else f" and was also the fastest ({fastest[2]:.1f} ms). "
            )
            + "Consider the trade-off between accuracy and speed for your "
              "deployment needs."
        ),
    }


# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------

def _overview(reports: List[BenchmarkReport]) -> str:
    n = len(reports)
    has_gt = any(r.metrics for r in reports)
    has_nav = any(r.navigation_result.path_found for r in reports)

    parts = [
        f"We evaluated {n} model{'s' if n > 1 else ''} on the provided image."
    ]
    if has_gt:
        parts.append(
            "Since a ground-truth mask was provided, we measured how "
            "accurately each model identifies different terrain types."
        )
    if has_nav:
        parts.append(
            "We also tested whether the robot could find a safe path "
            "through the scene and rated the route's safety."
        )
    return " ".join(parts)


# ------------------------------------------------------------------
# Text renderer
# ------------------------------------------------------------------

def _render_text(explanation: Dict[str, Any]) -> str:
    """Render the explanation dict as formatted plain text."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append(f"  {explanation['title']}")
    lines.append(sep)
    lines.append("")
    lines.append(f"  {explanation['overview']}")
    lines.append("")

    for model in explanation.get("models", []):
        lines.append(f"  {'─' * 66}")
        lines.append(f"  Model: {model['model_name']}")
        lines.append(f"  {'─' * 66}")
        lines.append("")

        for section in model.get("sections", []):
            lines.append(f"  ▸ {section['title']}")
            if "metric" in section:
                lines.append(f"    {section['metric']}")
            lines.append("")
            # Word-wrap explanation at ~65 chars.
            text = section.get("explanation", "")
            for para in text.split(". "):
                para = para.strip()
                if para and not para.endswith("."):
                    para += "."
                if para:
                    lines.append(f"    {para}")
            lines.append("")

    comparison = explanation.get("comparison")
    if comparison:
        lines.append(f"  {'─' * 66}")
        lines.append(f"  {comparison['title']}")
        lines.append(f"  {'─' * 66}")
        lines.append("")
        lines.append(f"    Best accuracy: {comparison['best_accuracy']}")
        lines.append(f"    Fastest:       {comparison['fastest']}")
        lines.append("")
        for para in comparison["explanation"].split(". "):
            para = para.strip()
            if para and not para.endswith("."):
                para += "."
            if para:
                lines.append(f"    {para}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _quality_label(miou: float) -> str:
    if miou >= 0.80:
        return "excellent"
    if miou >= 0.65:
        return "good"
    if miou >= 0.50:
        return "moderate"
    if miou >= 0.35:
        return "fair"
    return "poor"


def _speed_context(ms: float) -> str:
    if ms < 30:
        return "That's fast enough for real-time use (30+ frames per second)."
    if ms < 100:
        return "That's good for near-real-time applications (10+ FPS)."
    if ms < 500:
        return "This is suitable for offline analysis but too slow for real-time."
    return "This is quite slow — consider a lighter model for faster results."


def _safety_verdict(score: float) -> str:
    if score >= 0.90:
        return "This route is considered very safe for the robot."
    if score >= 0.70:
        return "This route is reasonably safe but has some risk areas."
    if score >= 0.50:
        return "This route has notable risks — the robot should proceed with caution."
    return "This route is unsafe — the robot should not follow this path."


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
