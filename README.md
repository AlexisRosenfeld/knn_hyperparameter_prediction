# **Automatic Detection of Hyperparameters for k-NN Models**

This project explores the relationship between dataset characteristics and hyperparameter selection, particularly the best `k` for k-Nearest Neighbors (k-NN). The pipeline includes data generation, feature extraction, and prediction to automate and optimize hyperparameter detection for machine learning models.

<figure>
  <img style="float: left;" src="fig/fig1.mmd.svg"/>
   <figcaption>Pipeline of the processs</figcaption>

</figure>

<br>
<br>

## **Step of the pipeline**

1. **Data Generation**:
```
from generation.data_generation import genf_multiple_datasets
genf_multiple_datasets(100)
```
   - Generates synthetic datasets with varying characteristics (size, distribution types, noise levels).
   - Saves datasets as CSV files in the `raw_generated_data` folder.




2. **Feature Extraction**:
```
from extraction.features_extraction import create_dataset
create_dataset("raw_generated_data", "processed_dataset.csv")
```
   - Processes raw datasets to extract relevant features.
   - Calculates key metrics such as noise levels, class counts, and dataset dimensions.
   - Stores processed data as `processed_dataset.csv`.

3. **Prediction**:
   - Builds a regression model to predict the best hyperparameter (`best_k`) based on dataset features.
   - Trains and evaluates the model using scikit-learn's regression utilities.
   - Saves the trained model for future use.
```
from prediction.model_training import train_global


model, X, y, X_train, X_test, y_train, y_test = train_global(data)
```



## **Usage**
To summary, run the following command to get a model 
```
from generation.data_generation import genf_multiple_datasets
genf_multiple_datasets(100)



```
## **Structure of the depository**

raw_generated_data : where you put the raw data