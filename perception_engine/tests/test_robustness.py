"""Tests for robustness perturbation functions."""

import numpy as np

from perception_engine.evaluation.robustness import (
    _brightness_shift,
    _contrast_shift,
    _gaussian_blur,
    _gaussian_noise,
)


def _make_dummy_image() -> np.ndarray:
    """Create a small synthetic RGB image."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


def test_brightness_preserves_shape():
    img = _make_dummy_image()
    result = _brightness_shift(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_blur_preserves_shape():
    img = _make_dummy_image()
    result = _gaussian_blur(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_noise_preserves_shape():
    img = _make_dummy_image()
    result = _gaussian_noise(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_contrast_preserves_shape():
    img = _make_dummy_image()
    result = _contrast_shift(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_noise_adds_variation():
    """Noise should produce a different image."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = _gaussian_noise(img, std=50.0)
    # With std=50, it's statistically near-impossible for all pixels
    # to remain exactly 128.
    assert not np.array_equal(img, result)
