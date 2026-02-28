import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from Part_1 import X_train_scaled, X_test_scaled, y_train, y_test, scaler, grid_search, y_train_bal, y_test, y_pred

print("\n PCA Test ")

# Keeping enough components to explain 95% variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original training shape: {X_train_scaled.shape}")  #{} in an numpy array
print(f"PCA-transformed shape: {X_train_pca.shape}")


# Number of components selected
n_components = pca.n_components_
explained_variance = pca.explained_variance_ratio_.sum() * 100
print("\nPCA Results:")
print(f"Original features: {X_train_scaled.shape[1]}")
print(f"Components to retain 95% variance: {n_components}")
print(f"Actual variance explained: {explained_variance:.2f}%")
print(f"New training data shape: {X_train_pca.shape}")
print(f"New test data shape: {X_test_pca.shape}")


# Get best parameters from Part 1
best_params = grid_search.best_params_
print(f"\n Best parameters from Part 1: {best_params}")

# 4. Train a NEW KNN with the SAME best parameters(tuned KNN, in Part_1,py) on PCA data
print(f"\nTraining KNN with {best_params} on PCA-transformed data...")

knn_pca = KNeighborsClassifier(**best_params)  # Unpack best parameters
knn_pca.fit(X_train_pca, y_train_bal)  # Train on PCA data

# 5. Evaluate on test set
y_pred_pca = knn_pca.predict(X_test_pca)
#to test probability for one class
y_pred_proba_pca = knn_pca.predict_proba(X_test_pca)[:, 1] 

# Calculate and Evaluating metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef,RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, average_precision_score)

accuracy = accuracy_score(y_test, y_pred_pca)
#used {} because vairable inside a string, and rounded to the nearest decimal value
print(f"\nAccuracy: {accuracy: .4f}")  
#used 'f' inside print statement due to a variable inside

precision = precision_score(y_test, y_pred_pca)
print(f"\nPrecision Score: {precision: .4f}")

recall = recall_score(y_test, y_pred_pca)
print(f"\nRecall Score: {recall: .4f}")

f1 = f1_score(y_test, y_pred_pca)
print(f"\nF1 Score: {f1: .4f}")

mcc = matthews_corrcoef(y_test, y_pred_pca)
print(f"\nMCC: {mcc: .4f}")

#Library to display simple plots
import matplotlib.pyplot as plt   

#Visualizing the Above Results
#Creating Confusion Matrix
cm = confusion_matrix(y_test, y_pred_pca)
class_names = ['No Diabetes', 'Diabetes']    #Classification
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("PCA Transformed Confusion Matrix (KNN on Diabetic and Non-Diabetic classification)")
plt.show()


# ROC Curve + AUC
roc_disp = RocCurveDisplay.from_predictions(y_test, y_pred_pca)
plt.title("PCA: ROC Curve (kNN)")
plt.show()
roc_auc = roc_auc_score(y_test, y_pred_pca)
print("\nAvg  pca transformed ROC: ", roc_auc)

# Precision–Recall Curve + PR-AUC
pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
plt.title("PCA: Precision–Recall Curve (kNN)")
plt.show()
pr_auc_ap = average_precision_score(y_test, y_pred_pca)
print("\nAvg pca transformed Precision-Recall Curve: ", pr_auc_ap)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("\nSpecificity value: ", specificity)

