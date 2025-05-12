# Machine Learning Models 

This repository contains seven separate Jupyter Notebooks, each demonstrating a different machine learning model using the same synthetic dataset.

## Overview

The dataset (`ml_customer_data.csv`) is a synthetic collection of 500 samples that contains:
- **age**: Customer's age (integer)
- **salary**: Customer's salary (integer)
- **purchased**: Target variable (0 = not purchased, 1 = purchased)

The purpose of this project is to demonstrate various machine learning techniques on the same dataset. Each notebook walks through the process of:
1. Loading the dataset
2. Splitting the data into training and testing sets
3. Training a model
4. Evaluating the model’s performance
5. (Optional) Visualizing the output

## Notebooks

- **[Logistic_Regression](./1_Logistic_Regression.ipynb):**  
- **Type**: Supervised (Classification)

- **Description**: Logistic Regression is used to predict binary outcomes. It applies a logistic (sigmoid) function to estimate the probability that a data point belongs to a certain class (e.g., purchased or not). This notebook trains a logistic regression model on the dataset and reports performance metrics such as accuracy, confusion matrix, classification report, and model coefficients.

- **[2_Linear_Regression.ipynb](./2_Linear_Regression.ipynb):**  
  Applies Linear Regression to predict the target variable as a continuous value. (Note: Here 'purchased' is treated numerically for demonstration purposes.) Evaluation is based on mean squared error (MSE).

- **[3_Decision_Tree.ipynb](./3_Decision_Tree.ipynb):**  
  Demonstrates how to use a Decision Tree Classifier to predict purchase decisions. It provides accuracy and other performance metrics.

- **[4_Random_Forest.ipynb](./4_Random_Forest.ipynb):**  
  Uses the Random Forest Classifier (an ensemble of Decision Trees) to improve prediction performance and reduce overfitting.

- **[5_SVM.ipynb](./5_SVM.ipynb):**  
  Implements a Support Vector Machine (SVM) for classification. The notebook shows how SVM finds the best separating hyperplane and outputs the accuracy score.

- **[6_KNN.ipynb](./6_KNN.ipynb):**  
  Uses the K-Nearest Neighbors algorithm to predict whether a customer will purchase, based on the majority vote among the nearest neighbors. It includes accuracy evaluation.

- **[7_KMeans.ipynb](./7_KMeans.ipynb):**  
  Applies K-Means clustering (an unsupervised learning method) to group customers based on their age and salary. A scatter plot is generated to visualize the clusters.
