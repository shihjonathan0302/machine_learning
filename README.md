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

- **[Logistic_Regression](./1_Logistic_Regression.ipynb)**  
  - **Type:** Supervised (Classification)  
  - **Description:** Logistic Regression is used to predict binary outcomes. It applies a logistic (sigmoid) function to estimate the probability that a data point belongs to a certain class (e.g., purchased or not). This notebook trains a logistic regression model on the dataset and reports performance metrics such as accuracy, confusion matrix, classification report, and model coefficients.

- **[Linear_Regression](./2_Linear_Regression.ipynb)**  
  - **Type:** Supervised (Regression)  
  - **Description:** Linear Regression models the relationship between one or more input features and a continuous target variable. Although the target in this dataset is binary, this notebook demonstrates how linear regression would approximate numeric predictions and evaluates performance using mean squared error (MSE).

- **[Decision_Tree](./3_Decision_Tree.ipynb)**  
  - **Type:** Supervised (Classification)  
  - **Description:** Decision Trees classify data by asking a sequence of binary questions. The algorithm recursively splits the dataset based on feature values to build a tree structure. This model is easy to interpret and visualize. The notebook demonstrates purchase prediction and reports accuracy.

- **[Random_Forest](./4_Random_Forest.ipynb)**  
  - **Type:** Supervised (Classification)  
  - **Description:** Random Forest is an ensemble model that builds multiple decision trees on random subsets of the data and averages their predictions. It improves accuracy and reduces overfitting. This notebook applies a Random Forest Classifier to the dataset and reports performance.

- **[SVM](./5_SVM.ipynb)**  
  - **Type:** Supervised (Classification)  
  - **Description:** Support Vector Machine (SVM) finds the optimal hyperplane that separates different classes with the maximum margin. This notebook demonstrates how SVM can be used for binary classification of purchase behavior and evaluates the results.

- **[NN](./6_KNN.ipynb)**  
  - **Type:** Supervised (Classification)  
  - **Description:** K-Nearest Neighbors (KNN) is a non-parametric algorithm that classifies a data point based on the majority label of its K nearest neighbors in the feature space. This notebook uses KNN to predict purchase decisions and reports accuracy.

- **[KMeans](./7_KMeans.ipynb)**  
  - **Type:** Unsupervised (Clustering)  
  - **Description:** K-Means clustering partitions data into K groups by minimizing intra-cluster variance. This notebook applies K-Means to group customers based on their age and salary, and visualizes the results using a scatter plot with cluster assignments.
