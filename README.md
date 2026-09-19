# KNN Pipeline and GridSearchCV Mini Project

A hands-on Machine Learning mini-project focused on K-Nearest Neighbors classification using feature scaling, pipelines, cross-validation, and hyperparameter tuning.

## Project Objective

The goal of this project is to understand how KNN performance changes with different values of K and different neighbor weighting strategies.

The experiment compares:

- K = 3, 5, 7, 11, 21
- `weights="uniform"`
- `weights="distance"`
- 5-fold cross-validation

## Dataset

This project uses the Scikit-learn Wine Classification Dataset.

The dataset contains:

- 178 samples
- 13 numerical features
- 3 target classes

## Machine Learning Workflow

1. Load the Wine dataset
2. Perform basic exploratory data analysis
3. Split the dataset into training and test sets
4. Build a Pipeline using:
   - `StandardScaler`
   - `KNeighborsClassifier`
5. Use `GridSearchCV`
6. Test different K values and weighting methods
7. Compare cross-validation accuracy
8. Select the best model
9. Evaluate the final model on the test set

## Why StandardScaler?

KNN is a distance-based algorithm.

Features with larger numerical values can dominate the distance calculation, so `StandardScaler` is used to place features on a comparable scale.

## Pipeline

```python
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])
