# Stroke Prediction Analysis

This project focuses on analyzing a comprehensive dataset related to stroke prediction. The objective is to evaluate the dataset's quality, identify significant features influencing stroke prediction, and evaluate the effectiveness of various modeling techniques for predicting the likelihood of a stroke. The analysis leverages both traditional statistical methods and machine learning techniques to derive insights and build predictive models.

## Table of Contents
- [Problem Definition](#problem-definition)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Data Splitting](#data-splitting)
- [Model Selection](#model-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Validation](#model-validation)
  - [Results Interpretation](#results-interpretation)
- [Deployment](#deployment)
  - [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

## Problem Definition
The central objective of this analysis is to discern the indicators associated with the likelihood of stroke using the Stroke Prediction Dataset. This dataset encompasses a variety of health-related features, and the goal is to assess data integrity, pinpoint significant features, and evaluate the effectiveness of multiple modeling methods for stroke prediction.

## Data Collection
The dataset is a collection of health-related information aimed at assessing the risk of stroke in individuals. It includes various attributes such as age, hypertension, heart disease, glucose levels, BMI, and more, with the target variable being the occurrence of a stroke (binary classification: Stroke vs. No Stroke).

## Data Preprocessing
The preprocessing steps were crucial to ensuring data quality and model effectiveness. These include:

### Data Cleaning
Outliers were identified and handled using boxplots and histograms to ensure data integrity.

### Exploratory Data Analysis (EDA)
Correlation matrices were constructed to examine relationships among variables and identify potential predictors for stroke prediction.

### Feature Engineering
New features were created, and the importance of existing features was assessed to select the most relevant predictors for model development.

### Data Splitting
The dataset was divided into training and testing sets to evaluate model performance properly.

## Model Selection
A variety of machine learning models were trained and evaluated to predict the likelihood of stroke, including:

### Model Training
The models employed for training include:
- Logistic Regression
- Decision Trees
- Random Forest Classifier
- Automated Machine Learning (AutoML) Models such as Gradient Boosting Machines (GBM) and XGBoost

### Model Evaluation
Metrics such as precision, recall, accuracy, and F1 score were used to evaluate model performance. Cross-validation was performed to assess generalization.

### Hyperparameter Tuning
Hyperparameter tuning was performed using grid search and randomized search to optimize model performance.

### Model Validation
Validation techniques were used to assess model effectiveness, especially for classifying instances in the "No Stroke" category.

### Results Interpretation
SHAP (SHapley Additive exPlanations) was used for model interpretability, providing insights into the feature importance and their contributions to the predictions.

## Deployment
The model was deployed as a part of an analytical pipeline, with the possibility for integration into a healthcare application.

### Monitoring and Maintenance
Ongoing monitoring and maintenance strategies are suggested to ensure the model's continued effectiveness and to track performance over time.

## Conclusion
The analysis revealed key predictors of stroke, such as average glucose level, LDL, BMI, age, and stress levels, across multiple models. The models provided useful insights, but the overall accuracy remains modest, indicating the need for further refinement and exploration. 

- Logistic regression served as a baseline model, offering interpretable results.
- Decision trees provided intuitive insights into stroke risk.
- Random Forest classifiers improved accuracy by leveraging ensemble learning.
- AutoML models (using H2O) automated the selection and tuning of various algorithms, improving prediction accuracy.
- SHAP analysis helped interpret the contribution of features such as glucose levels, BMI, and LDL to the model predictions.

## Technologies Used
- **Python**
  - Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, SHAP
- **AutoML**: H2O AutoML
- **Machine Learning Models**: Logistic Regression, Decision Trees, Random Forest, XGBoost, GBM
- **Data Visualization**: Matplotlib, Seaborn

## References
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Letter Value Plot: The Easy-to-Understand Boxplot for Large Datasets](https://towardsdatascience.com/letter-value-plot-the-easy-to-understand-boxplot-for-large-datasets-12d6c1279c97)
- [Interpretable Machine Learning: Logistic Regression](https://christophm.github.io/interpretable-ml-book/logistic.html)
- [Understanding Logistic Regression in Python](https://www.datacamp.com/tutorial/understanding-logistic-regression-python)
- [AutoML Capabilities of H2O Library](https://www.kaggle.com/code/paradiselost/tutorial-automl-capabilities-of-h2o-library)
- [AutoML Capabilities of H2O Library (Second Link)](https://www.kaggle.com/code/paradiselost/tutorial-automl-capabilities-of-h2o-library)
- [What is Multicollinearity?](https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/)
- [Scikit-learn RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
