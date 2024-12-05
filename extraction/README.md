# **Feature Extraction for Machine Learning Datasets**

This folder contains scripts and utilities for extracting features from raw datasets and creating a processed dataset of interest. The script processes datasets generated in the `raw_generated_data` folder of the `generation` module.

## **Features**
- **Raw Data Loading**: Automatically scans and loads all `.csv` files from the `raw_generated_data` folder.
- **Feature Extraction**: Extracts key features, including:
  - Number of rows (`n_rows`)
  - Number of classes (`n_classes`)
  - Number of features (`n_features`)
  - Noise level (`noise_level`)
- **Hyperparameter Tuning**: Determines the best `k` for a k-Nearest Neighbors (k-NN) model.
- **Processed Dataset**: Combines extracted features and `best_k` into a single dataset saved as `processed_dataset.csv`.

## **Files**
- `extract_features.py`: Core script for loading, processing, and extracting features from raw datasets.
- `processed_dataset.csv`: Output dataset containing extracted features and best hyperparameter values.

## **Usage**

### **1. Run Feature Extraction**
Run the `extract_features.py` script to process the raw datasets and create the dataset of interest: (not for the moment, only in notebook)

