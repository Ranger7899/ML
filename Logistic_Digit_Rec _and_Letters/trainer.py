import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_param=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None

    def initialize_params(self, n_features, n_classes):
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

    def compute_cost(self, X, y, num_classes):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = softmax(z)
        y_one_hot = np.eye(num_classes)[y]

        cost = -(1 / m) * np.sum(y_one_hot * np.log(predictions + 1e-15))

        if self.regularization_param:
            cost += (self.regularization_param / (2 * m)) * np.sum(self.weights ** 2)

        return cost

    def compute_gradients(self, X, y, num_classes):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = softmax(z)
        y_one_hot = np.eye(num_classes)[y]

        dw = (1 / m) * np.dot(X.T, (predictions - y_one_hot))
        db = (1 / m) * np.sum(predictions - y_one_hot, axis=0, keepdims=True)

        if self.regularization_param:
            dw += (self.regularization_param / m) * self.weights

        return dw, db

    def fit(self, X, y, num_classes, epochs=10):
        n_features = X.shape[1]
        self.initialize_params(n_features, num_classes)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in tqdm(range(self.num_iterations), desc="Training Progress"):
                dw, db = self.compute_gradients(X, y, num_classes)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                if i % 100 == 0:
                    cost = self.compute_cost(X, y, num_classes)
                    print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = softmax(z)
        return np.argmax(probabilities, axis=1)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
        print(f"Model loaded from {filepath}")

def load_dataset(csv_file: str, sample_size: int = None):
    """
    Load dataset from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        sample_size (int, optional): Number of samples to load. If None, load all data.

    Returns:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
    """
    data = pd.read_csv(csv_file)
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)
    y = data.iloc[:, 0].values  # First column is the label
    X = data.iloc[:, 1:].values  # Remaining columns are pixel values
    return X, y

def main():
    # Load dataset from a CSV file with a limited sample size
    train_csv_file = "emnist-bymerge-train.csv"  # Replace with the path to your training CSV file
    X_train, y_train = load_dataset(train_csv_file, sample_size=20000)  # Load a subset of 20,000 samples

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the model
    num_classes = 47
    model = LogisticRegression(
        learning_rate=0.003,
        num_iterations=1000,
        regularization_param=0.1
    )
    model.fit(X_train, y_train, num_classes, epochs=10)

    # Validate the model
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions) * 100
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save the model
    model.save_model("logistic_regression_model.pkl")

    # Load test dataset from a different CSV file with a limited sample size
    test_csv_file = "emnist-bymerge-test.csv"  # Replace with the path to your test CSV file
    X_test, y_test = load_dataset(test_csv_file, sample_size=5000)  # Load a subset of 5,000 samples

    # Test the model on the test dataset
    test_predictions = model.predict(X_test)
    f1 = f1_score(y_test, test_predictions, average='weighted')
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
