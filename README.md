# SVM VS Catboost

**Report** is under: Comparative_Analysis_of_SVM_and_CatBoost_on_SPECTF_ShengY.pdf

## Table of Contents
- Installation
- SVM with only Numpy and Pandas
- Catboost

# Installation for Prerequistions

To use the SVM classifier, simply download the Python script or clone this repository.  
However, since the program needs to graph, balance(not usually needed), having accuracy scores. We need the following packages:  
- sklearn.metrics  
- seaborn
- matplotlib.pyplot
- cvxopt

To use Catboosting classifier, we need the following as additional installations:
- sklearn.model_selection
- catboost

For installation, kindly type and execute the following code in shell or console!  

```sh 
pip install <package_name>
```

# Support Vector Machine (SVM) Classifier

This Python script provides a comprehensive implementation of Support Vector Machine (SVM) for classification tasks. The program offers multiple kernel options(linear, quadratic, rbf), performance metrics, cross-validation, and hyperparameter tuning. I used SPECTF data for experimenting (can use other data but needs some minor changes on codes).

## Table of Contents
- Usage
  - Running the Program
  - Actions
  - Optional Parameters
- Warnings


## Usage

### Running the Program

To run the program, open your terminal or command prompt, navigate to the directory where my script(final_svm.py) is located, and execute the following command:
```sh
python final_svm.py <data_location> <target> <action> <kernel> [optional parameters]
```

Where:

- `<data_location>`: The path to the CSV file containing the dataset.
- `<target>`: The name of the column containing the target variable in the dataset.
- `<action>`: The action to perform, which can be one of the following: train, search, or predict.
- `<kernel>`: The kernel to use for the SVM. Choices are linear, quadratic, or rbf.

### Actions

- train: Train the SVM model with the provided dataset and display performance metrics. Optionally, plot confusion matrix and ROC curve by setting `graph_op=True`.
- search: Perform a grid search for the optimal hyperparameters (C and gamma) using cross-validation. Use the `strong` parameter to control the granularity of the search.
- predict: Train the SVM model with the provided dataset and make predictions on a test dataset. Optionally, save the predictions to a CSV file by setting `save=True`.

### Optional Parameters

This script offers several optional parameters to customize the SVM model and its evaluation:

- fold: The number of folds to use for cross-validation (default: 10).
- hold: The proportion of the dataset to be used as the testing set for each fold (default: 0.2).
- C: The regularization parameter for the SVM (default: 1.0).
- normalize: Whether to normalize the dataset (default: True).
- balance_data: Whether to balance the classes in the dataset (default: False).
- graph_op: Whether to plot the confusion matrix and ROC curve during training (default: False).
- tol: The tolerance for stopping criterion in the SVM optimization (default: 1e-5).
- iter: The maximum number of iterations for the SVM optimization (default: 1e10).
- gamma: The gamma parameter for the RBF kernel (default: 0.1).
- strong: Whether to perform a more granular grid search (default: False).
- save: Whether to save the predictions to a CSV file during the predict action (default: False).

## Warnings

- Ensure that the dataset is in CSV format with no header row. The script assumes that the target variable column is the first column in the dataset.
- The program may take a huge amount of time to execute depending on the dataset size and the chosen action(even for regular grid search , it will take about 23 minutes to for rbf kernel). A more granular grid search (`strong=True`) will significantly increase the computation time.




# CatBoost Model Training and Prediction

This script provides functionality to train, search for optimal hyperparameters using grid search, and predict using the CatBoost model on a SPECTF data (can use other data but needs some minor changes on codes).

## Usage

To run the program (final_cat.py), use the following command:

```sh
python final_cat.py <data_location> <target> <action> [<key=value>...]
```

#### Arguments
- `<data_location>`: The path to the CSV file containing the dataset.
- `<target>`: The name of the column containing the target variable in the dataset.
- `<action>`: The action to perform, which can be one of the following: train, search, or predict. 
- [<key=value>...]: Optional key-value pairs for hyperparameters.  


### Actions
- train: Train the CatBoost model with the given hyperparameters and display the performance metrics.  
- search: Perform a grid search to find the optimal hyperparameters for the CatBoost model and display the results.  
- predict: Train the CatBoost model with the given hyperparameters and make predictions on a test dataset.  

#### Examples  

- **Train a CatBoost model:**  
```sh
python final_cat.py data.csv target_column train
```

- **Perform a grid search for optimal hyperparameters:**  
```sh
python final_cat.py data.csv target_column search
```

- **Make predictions with a CatBoost model:**  
```sh
python final_cat.py data.csv target_column predict
```


### Warnings  
- Make sure the input dataset file is in CSV format and has appropriate column names.  
- The dataset should not contain any missing values, as the CatBoost model does not handle missing data.    
- Ensure that the target column is present in the dataset and has the correct name.  
- The grid search functionality may take a long time to run, depending on the number of hyperparameter combinations and the size of the dataset.  

### Hyperparameters
You can pass hyperparameters as key-value pairs in the command line. For example:
```sh
python final_cat.py data.csv target_column train depth=6 iterations=1000 learning_rate=0.01
```

**For a complete list of hyperparameters and their descriptions, please refer to their CatBoost documentation because I am simply passing those arguments into their catboost model.**


