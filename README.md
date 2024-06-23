# Credit-Card-fraud-Detection
## Introduction
This project aims to build a machine learning model to identify fraudulent credit card transactions. The project involves preprocessing and normalizing the transaction data, handling class imbalance issues, and training a classification algorithm to classify transactions as fraudulent or genuine. The model's performance is evaluated using metrics like precision, recall, and F1-score. Techniques like oversampling or undersampling are considered to improve results.
## Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
### Source
The dataset is publicly available on [Kaggle](https://www.kaggle.com/) and can be accessed [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).
### Structure
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1 to V28**: Result of a PCA transformation to protect user identities and sensitive features.
- **Amount**: Transaction amount.
- **Class**: Response variable (1 for fraud, 0 for normal transactions).
## Objective
The primary objective of this project is to build a classification model that can accurately identify fraudulent transactions.
## Analysis Process
### Data Preprocessing
1. **Data Loading**: Load the dataset and perform initial exploration.
2. **Missing Values**: Check for and handle any missing values.
3. **Normalization**: Normalize the 'Amount' and 'Time' features using standard scaling.
4. **Class Imbalance Handling**: Address class imbalance using techniques such as Over-sampling or Under_Sampling Technique.
### Exploratory Data Analysis (EDA)
1. **Data Distribution**: Analyze the distribution of features.
2. **Correlation Analysis**: Investigate correlations between features.
### Model Building
1. **Data Splitting**: Split the dataset into training and testing sets.
2. **Model Selection**: Train classification models such as Logistic Regression and Random Forest.
### Model Evaluation
1. **Performance Metrics**: Evaluate models using precision, recall, F1-score, and ROC-AUC.
2. **Confusion Matrix**: Analyze the confusion matrix to understand model performance.
3. **Undersampling**: Implement techniques undersampling to improve model performance on the minority class.
### Results
1. **Best Model**: Identify the best-performing model based on evaluation metrics.
2. **Feature Importance**: Analyze feature importance to understand the impact of different features.
### Conclusion
The project successfully developed a machine learning model for fraud detection with high precision and recall, demonstrating its potential for practical application in real-world credit card fraud detection systems.

## Tools and Technologies
- **Python**: Programming language used for analysis and modeling.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning library.
- **Imbalanced-learn**: Handling class imbalance.
