# ML-Pipeline-for-Diabetes-Classification-using-Pima-Indians-Dataset
A machine learning project to predict the onset of diabetes based on diagnostic measurements. This repository covers EDA, data preprocessing (handling class imbalance and missing values), and classification modeling using the PIMA Indians Diabetes Dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)
![numpy](https://img.shields.io/badge/numpy-1.21+-blue.svg)
![matplotlib](https://img.shields.io/badge/matplotlib-3.5+-red.svg)
![seaborn](https://img.shields.io/badge/seaborn-0.11+-blue.svg)


A complete machine learning pipeline for binary classification on the PIMA Indians Diabetes Dataset, comparing kNN with PCA and ANOVA feature selection techniques.

📊 Dataset
Source: PIMA Indians Diabetes Database (Kaggle)

Context: 768 female patients of Pima Indian heritage, age 21+

Features (8): Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Target: Outcome (0 = Non-diabetic, 1 = Diabetic)

Challenge: Class imbalance, zero values in medical fields

🛠️ Technologies Used
Language: Python 3.8+

Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

🔬 Methodology
1. Data Preprocessing
Removed zero values from medical fields (Glucose, BMI)

Dropped duplicates

Applied RandomOverSampler to handle class imbalance

Normalized features using StandardScaler

2. Model Training
Algorithm: K-Nearest Neighbors (kNN)

Hyperparameter Tuning: GridSearchCV with RepeatedStratifiedKFold (5 folds × 3 repeats)

Best Params: k=3, euclidean distance, weights='distance'

3. Dimensionality Reduction & Feature Selection
PCA: Retained 95% variance (7 components)

ANOVA SelectKBest: Selected top 6 features (excluded SkinThickness, BloodPressure)

4. Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, Specificity, MCC, ROC-AUC, PR-AUC

📈 Key Results
Metric	   Tuned kNN	  kNN+PCA	 kNN+ANOVA
Accuracy	    0.66	     0.68	     0.74
Precision	    0.57	     0.58	     0.65
Recall	      0.66	     0.65	     0.71
ROC-AUC	      0.60	     0.67	     0.73

ANOVA feature selection significantly outperformed both baseline kNN and PCA approaches, achieving the best overall performance.


