from PIL import Image
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegressionSoftmax:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, regularization_param: float = 0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None

    def initialize_params(self, n_features: int, n_classes: int):
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

    def compute_cost(self, X: np.ndarray, y: np.ndarray, num_classes: int) -> float:
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = softmax(z)
        y_one_hot = np.eye(num_classes)[y]  # One-hot encode labels
        cost = -(1 / m) * np.sum(y_one_hot * np.log(predictions + 1e-15))  # Add epsilon for stability

        if self.regularization_param:
            reg_cost = (self.regularization_param / (2 * m)) * np.sum(self.weights ** 2)
            cost += reg_cost
        return cost

    def compute_gradients(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = softmax(z)
        y_one_hot = np.eye(num_classes)[y]  # One-hot encode labels

        dw = (1 / m) * np.dot(X.T, (predictions - y_one_hot))
        db = (1 / m) * np.sum(predictions - y_one_hot, axis=0, keepdims=True)

        if self.regularization_param:
            dw += (self.regularization_param / m) * self.weights
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        n_features = X.shape[1]
        self.initialize_params(n_features, num_classes)
        for i in range(self.num_iterations):
            dw, db = self.compute_gradients(X, y, num_classes)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 100 == 0:
                cost = self.compute_cost(X, y, num_classes)
                print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        probabilities = softmax(z)
        return np.argmax(probabilities, axis=1)

    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
        print(f"Model loaded from {filepath}")


def single_image_test(model: LogisticRegressionSoftmax, image_path: str, save_input_path: str = "input_data.csv"):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match training data dimensions
    img_array = np.array(img)
    img_flatten = img_array.flatten()
    img_normalized = img_flatten.reshape(1, -1) / 255.0  # Normalize to [0, 1] and reshape for prediction

    # Save the normalized input data to a CSV file
    pd.DataFrame(img_normalized).to_csv(save_input_path, index=False, header=False)
    print(f"Input data saved to {save_input_path}")

    z = np.dot(img_normalized, model.weights) + model.bias
    probabilities = softmax(z)
    prediction = np.argmax(probabilities)  # Predicted class
    print(f"Predicted Class for the given image: {prediction}")
    print(f"Class Probabilities: {probabilities.flatten()}")


def main():
    # Load dataset
    data = pd.read_csv("C:/Users/ymasa/Desktop/Simple-Neural-Network-for-Handwritten-Digit-Recognition/train.csv")
    y = data['label'].values
    X = data.drop('label', axis=1).values

    # Normalize features
    X = X / 255.0

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Train logistic regression model
    # num_classes = len(np.unique(y_train))  # Number of unique classes (e.g., 10 for MNIST)
    # model = LogisticRegressionSoftmax(learning_rate=0.06, num_iterations=1500, regularization_param=0.1)
    # model.fit(X_train, y_train, num_classes)

    # # Validate model
    # val_predictions = model.predict(X_val)
    # val_accuracy = accuracy_score(y_val, val_predictions) * 100
    # print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # # Save the model
    # model.save_model("softmax_model.pkl")

    model = LogisticRegressionSoftmax()
    model.load_model("softmax_model.pkl")

    # Test the model on a specific image
    single_image_test(model, "C:/Users/ymasa/Desktop/Simple-Neural-Network-for-Handwritten-Digit-Recognition/Handwritten_Images/handwritten_5.png")


if __name__ == "__main__":
    main()
