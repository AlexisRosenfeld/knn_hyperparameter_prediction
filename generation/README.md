# **Data Generation for Machine Learning**

This folder contains scripts and utilities for generating synthetic datasets for machine learning experiments. The datasets are stored in the `raw_generated_data` folder as CSV files.

## **Features**
- **Synthetic Data Generation:** Randomly generates datasets with varying characteristics:
  - Number of rows
  - Number of classes
  - Types of distributions (random, linear, quadratic, partial)
  - Noise levels
- **Visualization:** Allows visualization of generated datasets as scatter plots.
- **Automated Storage:** Each dataset is saved in the `raw_generated_data` folder with filenames like `g1.csv`, `g2.csv`, etc.

## **Files**
- `data_generation.py`: Core script for generating and saving datasets.
- `raw_generated_data/`: Directory where generated datasets are saved as CSV files.

## **Usage**

### **1. Generating Datasets**
To generate datasets, use the `genf_dataset` function in `data_generation.py`. Here's an example:

```python
from data_generation import genf_dataset

# Generate and save 10 datasets
for i in range(1, 11):
    genf_dataset(index=i)