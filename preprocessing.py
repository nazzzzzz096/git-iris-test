"""
Preprocessing utilities.
"""

import numpy as np


def normalize(values):
    """
    Normalize a numpy array between 0 and 1.
    """
    values = np.array(values)

    if values.size == 0:
        raise ValueError("Input array is empty")

    return (values - values.min()) / (values.max() - values.min())
