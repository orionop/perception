"""Tests for mask remapping utilities."""

import numpy as np

from perception_engine.engine.mask_remapping import (
    build_mapping_from_config,
    build_remap_lut,
    remap_mask,
)


def test_remap_basic():
    """Verify non-contiguous values are remapped to contiguous indices."""
    mask = np.array([[100, 200], [10000, 300]], dtype=np.int32)
    mapping = {100: 0, 200: 1, 300: 2, 10000: 9}

    result = remap_mask(mask, mapping)

    assert result[0, 0] == 0
    assert result[0, 1] == 1
    assert result[1, 0] == 9
    assert result[1, 1] == 2


def test_remap_unmapped_gets_ignore_index():
    """Unmapped raw values should receive the ignore index."""
    mask = np.array([[100, 999]], dtype=np.int32)
    mapping = {100: 0}

    result = remap_mask(mask, mapping, ignore_index=255)

    assert result[0, 0] == 0
    assert result[0, 1] == 255


def test_build_mapping_from_config_dict():
    """Config with dict-style mapping."""
    config = {"mask_value_mapping": {100: 0, 200: 1}}
    result = build_mapping_from_config(config)
    assert result == {100: 0, 200: 1}


def test_build_mapping_from_config_list():
    """Config with list-style mapping."""
    config = {
        "mask_value_mapping": [
            {"raw": 100, "index": 0},
            {"raw": 200, "index": 1},
        ]
    }
    result = build_mapping_from_config(config)
    assert result == {100: 0, 200: 1}


def test_build_mapping_from_config_none():
    """No mapping configured → returns None."""
    config = {}
    result = build_mapping_from_config(config)
    assert result is None


def test_lut_size():
    """LUT should be large enough to cover all mapped values."""
    mapping = {100: 0, 10000: 9}
    lut = build_remap_lut(mapping)
    assert len(lut) == 10001
    assert lut[100] == 0
    assert lut[10000] == 9
    assert lut[50] == 255  # Unmapped → ignore
