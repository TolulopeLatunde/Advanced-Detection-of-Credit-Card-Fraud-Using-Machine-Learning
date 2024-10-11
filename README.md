# Advanced-Detection-of-Credit-Card-Fraud-Using-Machine-Learning-Techniques

Project Overview

This project aims to build and evaluate various machine learning models to detect fraudulent credit card transactions effectively. Fraud detection in credit card transactions is a critical challenge faced by financial institutions, as fraudulent activities can cause significant financial losses. The goal of this project is to develop a robust model that can identify fraudulent transactions from a dataset with a high degree of accuracy, while also addressing challenges such as class imbalance.

Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/code
Key Features

Data Preprocessing: Includes handling missing values, feature scaling (standardizing the Amount feature), and splitting data into training and testing sets.

Class Imbalance Handling: Utilizes the SMOTE technique to oversample the minority class (fraudulent transactions), ensuring models can learn effectively despite the imbalance.

Machine Learning Models: Implements and compares the performance of several models, including:
  Random Forest Classifier
  XGBoost Classifier
  One-Class SVM (for anomaly detection)
  Logistic Regression
  Autoencoder (for unsupervised anomaly detection)
  
Model Evaluation: Models are evaluated based on classification metrics such as precision, recall, F1-score, accuracy, and the area under the ROC curve (AUC).

ROC Curve Analysis: Visualizes the ROC curves for all models to compare their performance in detecting fraud.

Project Objectives
1. Accurately Identify Fraudulent Transactions: Train models that can identify fraudulent transactions with high precision and recall.
2. Handle Class Imbalance: Implement strategies to address the imbalance between fraudulent and non-fraudulent transactions, ensuring that minority fraud cases are adequately detected.
3. Compare Model Performance: Evaluate and compare the performance of various machine learning algorithms to identify the most effective approach for fraud detection.

Dataset
The dataset used in this project contains credit card transactions with features such as transaction amount and anonymized data. It includes a binary classification target (Class) where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.

Tools Used
Python
Pandas and NumPy for data manipulation
Scikit-learn for building machine learning models
Imbalanced-learn (SMOTE) for handling class imbalance
Matplotlib for ROC curve visualization
TensorFlow/Keras for implementing Autoencoder

Results
The project compares the performance of different models through metrics like precision, recall, and AUC. The results are visualized using ROC curves, allowing for a comprehensive comparison of model effectiveness in detecting fraudulent transactions.

Conclusion
This project demonstrates the potential of machine learning techniques in detecting credit card fraud and emphasizes the importance of addressing class imbalance to improve the accuracy and reliability of fraud detection systems.

