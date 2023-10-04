# Diabetes Severity Prediction

## Overview

This project is a combination of a full-stack website and a machine learning model for predicting the severity level of diabetes in patients. The machine learning model is designed to classify patients into three categories: Type 1, Type 2, or Type 3 diabetes based on their health data.

## Prerequisites

Before running the project, make sure you have the following libraries and tools installed:

- Python (3.x)
- Jupyter Notebook (optional but recommended)
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

## Data

The project uses two datasets, 'diabetes.csv' and 'diabetes1.csv', which should be placed in the same directory as your code. These datasets are merged into 'merged_diabetes_data.csv' for analysis and modeling..

## Data Preprocessing

- The datasets are merged into one.
- Missing values in columns 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', and 'DiabetesPedigreeFunction' are replaced with their respective column means.
- The preprocessed data is saved as 'merged_diabetes_data.csv'.

## Exploratory Data Analysis (EDA)

- EDA is performed to understand the data distribution and relationships.
- Visualizations include count plots, histograms, box plots, correlation matrices, and more.

## Machine Learning Models

Three machine learning models are implemented:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC)

- Data is split into training and testing sets.
- Standardization is applied to the feature data.
- Each model is trained and tested, and evaluation metrics (accuracy, precision, recall, F1-score) are calculated.
- Confusion matrices and classification reports are generated for each model.

## Diabetes Severity Prediction

- You can use the trained Logistic Regression model to predict diabetes severity for a new patient by providing their health data as input.



```python
# Example usage to predict diabetes severity for a patient
patient = np.array([[1., 150., 70., 45., 0., 40., 1.5, 25]])
patient = scaler.transform(patient)
logistic_pred = model1.predict(patient)
result = logistic_pred[0]
if result == 1:
    print('DIABETES')
else:
    print('NO DIABETES')
