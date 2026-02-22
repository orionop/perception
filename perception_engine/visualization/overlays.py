"""
Visualization overlays.

Provides matplotlib-based rendering of segmentation masks, cost maps,
and planned paths overlaid on the original image.  All functions save
to disk and return the figure — no interactive UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend.

logger = logging.getLogger(__name__)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    alpha: float = 0.45,
    save_path: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """Overlay a coloured segmentation mask on the original image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        mask: ``(H, W)`` integer class ID array.
        num_classes: Total number of semantic classes (for colour map).
        alpha: Mask overlay opacity.
        save_path: If provided, save the figure to this path.
        class_names: Optional class ID → name mapping for the legend.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Resize image to match mask if shapes differ.
    display_img = _match_shape(image, mask.shape)

    ax.imshow(display_img)
    cmap = plt.cm.get_cmap("tab20", num_classes)
    ax.imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=num_classes - 1)
    ax.set_title("Segmentation Mask Overlay")
    ax.axis("off")

    # Add legend if class names provided.
    if class_names:
        handles = []
        for cid, cname in sorted(class_names.items()):
            colour = cmap(cid / max(num_classes - 1, 1))
            handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=colour, label=cname)
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.8,
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Mask overlay saved to %s", save_path)

    return fig


def overlay_path(
    image: np.ndarray,
    cost_map: np.ndarray,
    path: Optional[List[Tuple[int, int]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay the cost map and planned path on the original image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        cost_map: ``(H, W)`` traversability cost map.
        path: Ordered list of ``(row, col)`` coordinates, or ``None``
            if no path was found.
        save_path: Optional file save path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    display_img = _match_shape(image, cost_map.shape)

    # Left: cost map.
    ax_cost = axes[0]
    # Replace inf with a large finite value for display.
    display_cost = np.where(
        np.isfinite(cost_map), cost_map, np.nanmax(cost_map[np.isfinite(cost_map)]) * 1.5
    )
    ax_cost.imshow(display_cost, cmap="hot", interpolation="nearest")
    ax_cost.set_title("Traversability Cost Map")
    ax_cost.axis("off")

    # Right: path overlay on image.
    ax_path = axes[1]
    ax_path.imshow(display_img)
    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax_path.plot(cols, rows, color="lime", linewidth=2, label="Path")
        ax_path.plot(cols[0], rows[0], "go", markersize=8, label="Start")
        ax_path.plot(cols[-1], rows[-1], "r*", markersize=12, label="Goal")
        ax_path.legend(loc="upper right", fontsize=8)
        ax_path.set_title("Planned Path")
    else:
        ax_path.set_title("No Path Found")
    ax_path.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Path overlay saved to %s", save_path)

    return fig


def overlay_confidence(
    confidence_map: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render the confidence map as a heatmap.

    Args:
        confidence_map: ``(H, W)`` float array of per-pixel confidence.
        save_path: Optional file save path.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(confidence_map, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Confidence")
    ax.set_title("Model Confidence Map")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confidence map saved to %s", save_path)

    return fig


def _match_shape(
    image: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Resize an image to match a 2-D target shape if needed."""
    if image.shape[:2] == target_shape:
        return image

    from PIL import Image

    pil = Image.fromarray(image)
    pil = pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(pil)
