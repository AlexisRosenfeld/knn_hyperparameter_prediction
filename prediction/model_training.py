import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os



def load_training_data(data_path):
    """Loads the training dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}. Please ensure the dataset exists.")
    return pd.read_csv(data_path)

def train_regression_model(data):
    """Trains a regression model using the training dataset."""
    # Select features and target variable
    X = data.drop(columns=["best_k"])
    y = data["best_k"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model


def train_global(data):
    X = data.drop(columns=["best_k"])
    y = data["best_k"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, X, y, X_train, X_test, y_train, y_test

def save_model(model, output_path):
    """Saves the trained model to a file."""
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

def train_model(TRAINING_DATA_PATH = "processed_dataset.csv"):
    # Paths
    
    MODEL_OUTPUT_PATH = "regression_model.pkl"
    # Load training data
    print("Loading training data...")
    training_data = load_training_data(TRAINING_DATA_PATH)

    # Train regression model
    print("Training regression model...")
    regression_model = train_regression_model(training_data)

    # Save the model
    print("Saving the model...")
    save_model(regression_model, MODEL_OUTPUT_PATH)
