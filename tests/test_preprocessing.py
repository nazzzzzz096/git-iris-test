import numpy as np
from .preprocessing import normalize


def test_normalize_range():
    arr = np.array([10, 20, 30])
    result = normalize(arr)
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_negative_values():
    arr = np.array([-5, 0, 5])
    result = normalize(arr)
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_empty_array():
    import pytest
    with pytest.raises(ValueError):
        normalize(np.array([]))
