# **Regression Model for Hyperparameter Prediction**

This folder contains scripts and utilities for building a regression model to predict the best hyperparameter (`best_k`) for k-Nearest Neighbors (k-NN) using features extracted from the datasets.

## **Features**
- **Training Data Loading**: Loads the processed dataset from the `extraction` folder.
- **Model Training**: Trains a regression model using scikit-learn's `LinearRegression`.
- **Model Evaluation**: Evaluates the model using Mean Squared Error (MSE) and R² Score.
- **Model Saving**: Saves the trained regression model for future use.

## **Files**
- `train_regression_model.py`: Script for training the regression model.
- `regression_model.pkl`: Saved regression model after training (generated).
- `README.md`: Documentation for this folder.

## **Usage**

### **1. Train the Model**
Run the `train_regression_model.py` script to train the regression model:

```bash
python train_regression_model.py