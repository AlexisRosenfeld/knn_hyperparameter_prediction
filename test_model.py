import pandas as pd
import joblib
from sklearn.metrics import r2_score, accuracy_score

def load_data(data_path):
    """Loads the dataset."""
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure the dataset exists.")

def load_model(model_path):
    """Loads the trained model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is trained and saved.")

def evaluate_model(model, data, task):
    """
    Evaluates the model based on the specified task (regression or classification).
    Dynamically uses the first column as the dependent variable and others as features.
    """
    # Split data into features (X) and target (y)
    y = data.iloc[:, 0]  # The first column as the dependent variable
    X = data.iloc[:, 1:]  # All other columns as features

    # Predict using the model
    y_pred = model.predict(X)

    if task == "r":  # Regression
        # Calculate R² Score
        r2 = r2_score(y, y_pred)
        print(f"R² Score: {r2:.2f}")
    elif task == "c":  # Classification
        # Round predictions for classification (if applicable)
        y_pred = y_pred.round().astype(int)
        # Calculate Accuracy
        acc = accuracy_score(y, y_pred)
        print(f"Accuracy Score: {acc:.2f}")
    else:
        raise ValueError("Invalid task specified. Use 'r' for regression or 'c' for classification.")
if __name__ == "__main__":
    import argparse

    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Evaluate a machine learning model.")
    parser.add_argument("model_type", type=str, help="Type of model (e.g., 'regression_model.pkl').")
    parser.add_argument("data_file", type=str, help="Name of the dataset file (e.g., 'processed_dataset.csv').")
    parser.add_argument("task", type=str, choices=["r", "c"], help="Task type: 'r' for regression, 'c' for classification.")
    args = parser.parse_args()

    # Paths
    MODEL_PATH = f"./prediction/{args.model_type}"
    DATA_PATH = f"./extraction/{args.data_file}"

    print("Loading dataset...")
    dataset = load_data(DATA_PATH)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Evaluating the model...")
    evaluate_model(model, dataset, args.task)