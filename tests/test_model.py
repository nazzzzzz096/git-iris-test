from .models import predict, load_model
import os


def test_model_file_exists():
    assert os.path.exists("model.pkl")


def test_model_prediction_output():
    model = load_model()
    sample = [5.1, 3.5, 1.4, 0.2]  # Iris Setosa-like sample
    pred = predict(sample)
    assert pred in [0, 1, 2]
