"""
Train a Logistic Regression model on Iris dataset.
Saves the model as model.pkl.
"""

import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model():
    """Train and save a logistic regression model."""
    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    train_model()
    print("Model trained and saved as model.pkl")
