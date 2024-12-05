# **Automatic Detection of Hyperparameters for k-NN Models**

This project explores the relationship between dataset characteristics and hyperparameter selection, particularly the best `k` for k-Nearest Neighbors (k-NN). The pipeline includes data generation, feature extraction, and prediction to automate and optimize hyperparameter detection for machine learning models.

## **Features**
1. **Data Generation**:
   - Generates synthetic datasets with varying characteristics (size, distribution types, noise levels).
   - Saves datasets as CSV files in the `raw_generated_data` folder.

2. **Feature Extraction**:
   - Processes raw datasets to extract relevant features.
   - Calculates key metrics such as noise levels, class counts, and dataset dimensions.
   - Stores processed data as `processed_dataset.csv`.

3. **Prediction**:
   - Builds a regression model to predict the best hyperparameter (`best_k`) based on dataset features.
   - Trains and evaluates the model using scikit-learn's regression utilities.
   - Saves the trained model for future use.

