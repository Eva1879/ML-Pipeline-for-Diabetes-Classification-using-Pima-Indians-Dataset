import pandas as pd 
import numpy as np


#2. Loading the dataset from pandas library (since data in rows, columns) through saved csv file and storing in a variable db
db = pd.read_csv("diabetes.csv")

#To display shape of dataset--> rows, columns
print("Shape: ", db.shape)

#to display the data types, 
print(db.dtypes)

#to display first five rows
print(db.head())

#to get the class distribution
print(db.describe())

#3. Clean & preprocess the data if necessary.

#Counts the number of missing values in columns
print(db.isnull().sum())

#No missing values, so no need to handle

#counting the number of zeros in Glucose and BMI Column
print("Number of zeros in Glucose:  ",(db['Glucose'] == 0).sum())
print("Number of zeros in BMI: ",(db['BMI'] == 0).sum())

#Removing the zeros in Glucose, BMI column
db_clean = db[db['Glucose'] != 0] 
print(db_clean['Glucose'].head(20))

db_clean = db_clean[db_clean['BMI'] != 0]  
print(db_clean['BMI'].head(20))

#removing duplicates
db_clean = db_clean.drop_duplicates()
print(db_clean.head(10))


#Required libraries to import scaling function, KNN, and train test split from sklearn library.
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = db_clean.drop('Outcome', axis=1)  # All columns except 'Outcome' since to train the features, and Outcome is for classifying/predicting
y = db_clean['Outcome']  # Only the 'Outcome' column for target

# Split data into train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#4. Standard Scaler: Applying Normalization
# StandardScaler: z = (x - mean) / std
scaler = StandardScaler()

# Fit only on training data, to scale extreme values in X_train and X_test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set, scaled (X_train): {X_train_scaled.shape[0]} samples")
print(f"Test set, scaled (X_test): {X_test_scaled.shape[0]} samples")

# to check if the dataset is imbalanced, 
# we need to observe y_train and their binary classification of the categories of diabetic(1), non-diabetic(0)
print("\nClass distribution before oversampling:")
print(y_train.value_counts())


# it is slightly imbalanced data, so performing oversampling on the minority class
# this duplicates the related classification from y_train respectively
from imblearn.over_sampling import RandomOverSampler  #importing the req library

ros = RandomOverSampler(random_state=42)
# Perform oversampling
print("\nOverSampled Dataframe")
X_train_scaled, y_train_bal = ros.fit_resample(X_train_scaled, y_train)

print("\nClass distribution after oversampling:")
#print(X_train_scaled)
print(y_train_bal.value_counts())


# Create KNN classifier
knn = KNeighborsClassifier()

#using k=3, so that With K=3 (odd): 2 neighbors say Diabetes, 1 says No Diabetes → Diabetes ✓
# choosing odd numbers so the grouping won't be a tie
# Define parameter grid
param_grid = {
    'n_neighbors': range(3,36,2),
    'metric': ['euclidean', 'manhattan']
}

#To import the required libraries for repeated k folds and Grid search
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, GridSearchCV

repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Set up GridSearchCV with 5-fold, 3 repeat stratified Fold training and validation
grid_search = GridSearchCV(knn, 
                           param_grid, 
                           cv=repeated_cv,  #has the repeated stratified method
                           scoring='accuracy', 
                           n_jobs=-1)

	
grid_search.fit(X_train_scaled, y_train_bal)

# Best hyperparameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy score:", grid_search.best_score_)
print("Best trained model:", grid_search.best_score_)



#Step 9: Evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef, confusion_matrix,
                           roc_auc_score, average_precision_score,
                           RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)

# Predict on test
y_pred = grid_search.predict(X_test_scaled)
print("\n\nPredicted score: ", y_pred)

#An extra step in converting to probabilities of X_test
y_pred_proba = grid_search.predict_proba(X_test_scaled)  # Get ALL probabilities using Grid search, all possibilities, in 20% of data in X_test

# Now to extract individual class probabilities
y_pred_proba_class0 = y_pred_proba[:, 0]  # Class 0 probabilities
y_pred_proba_class1 = y_pred_proba[:, 1]  # Class 1 probabilities

print("\nClass Probabilities of 5 patients")

print("Class 0 probabilities (No Diabetes): ", y_pred_proba_class0[:5])
print("Class 1 probabilities (Diabetes): ", y_pred_proba_class1[:5])

# Evaluation
print("Accuracy:\n", accuracy_score(y_test, y_pred))
print("Precision:\n", precision_score(y_test,y_pred))

#Library to display simple plots
import matplotlib.pyplot as plt   

#Creating Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['No Diabetes', 'Diabetes']    #Classification
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (KNN on Diabetic and Non-Diabetic classification)")
plt.show()


rec  = recall_score(y_test, y_pred)
print("\nRecall Score: ",rec)

f1   = f1_score(y_test, y_pred)
print("\nF1 Score: ", f1)

mcc  = matthews_corrcoef(y_test, y_pred)
print("\nMatthews correlation coefficient: ",mcc)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("\nSpecificity value: ", specificity)


# ROC Curve + AUC
roc_disp = RocCurveDisplay.from_predictions(y_test, y_pred)
plt.title("ROC Curve (kNN)")
plt.show()
roc_auc = roc_auc_score(y_test, y_pred)

# Precision–Recall Curve + PR-AUC
pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
plt.title("Precision–Recall Curve (kNN)")
plt.show()
pr_auc_ap = average_precision_score(y_test, y_pred)


# Print metrics nicely, with rounded off decimal points
print("\n=== Metrics (k=5) ===")
print(f"Accuracy      : {accuracy_score(y_test,y_pred):.3f}")
print(f"Precision     : {precision_score(y_test,y_pred):.3f}")
print(f"Recall        : {rec:.3f}")
print(f"Specificity   : {specificity:.3f}")
print(f"F1 Score      : {f1:.3f}")
print(f"MCC           : {mcc:.3f}")
print(f"ROC-AUC       : {roc_auc:.3f}")
print(f"PR-AUC (AP)   : {pr_auc_ap:.3f}")



