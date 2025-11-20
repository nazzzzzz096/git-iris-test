"""
Load the trained model and run inference.
"""

import pickle
import numpy as np


def load_model():
    """Load the trained model from file."""
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def predict(sample):
    """
    Predict class label for a single sample.
    sample must be 1x4 array.
    """
    model = load_model()
    sample = np.array(sample).reshape(1, -1)
    return int(model.predict(sample)[0])
