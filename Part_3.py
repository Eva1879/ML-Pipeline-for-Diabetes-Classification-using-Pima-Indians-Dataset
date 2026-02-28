from sklearn.feature_selection import SelectKBest, f_classif
from Part_1 import X_train, X_train_scaled, X_test_scaled, y_train, y_test, scaler, grid_search, y_train_bal, y_pred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef,RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, average_precision_score)


print("\n SelectKBest with ANOVA F-test")

#We could start and test with different k values, but using k=6, by excluding 2 features
k_features = 6
print(f"\n Selecting top {k_features} features using ANOVA F-test, thus removing least 2 features...")
# Select top K features
selector = SelectKBest(score_func=f_classif, k=k_features)

# Fit on training data only
X_train_filtered = selector.fit_transform(X_train_scaled, y_train_bal)
X_test_filtered = selector.transform(X_test_scaled)

#first display original feature names
feature_names = X_train.columns.tolist()  #.tolist(), lists the columns

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
print("Selected features (ANOVA):")
print(list(selected_features))

X_train_ANOVA =X_train_filtered
X_test_ANOVA =X_test_filtered

# Get best parameters from Part 1
best_params = grid_search.best_params_
print(f"\n Best parameters from Part 1: {best_params}")

# Train a NEW KNN with the SAME best parameters on selected features by ANOVA
print(f"\nTraining KNN with {best_params} on {k_features} selected features...")
knn_selected = KNeighborsClassifier(**best_params)
knn_selected.fit(X_train_ANOVA, y_train_bal)

# Evaluate on test set
y_pred_selected = knn_selected.predict(X_test_ANOVA)
y_pred_proba_selected = knn_selected.predict_proba(X_test_ANOVA)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_selected)
print(f"\nAccuracy: {accuracy: .3f}");
precision = precision_score(y_test, y_pred_selected)
print(f"\nPrecision: {precision: .3f}");
recall = recall_score(y_test, y_pred_selected)
print(f"\nRecall: {recall: .3f}");
f1 = f1_score(y_test, y_pred_selected)
print(f"\nF1 Score: {f1: .3f}");
mcc = matthews_corrcoef(y_test, y_pred_selected)
print(f"\nMCC: {mcc: .3f}");


#Creating Confusion Matrix
cm = confusion_matrix(y_test, y_pred_selected)
class_names = ['No Diabetes', 'Diabetes']    #Classification
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("ANOVA Selected Confusion Matrix (Diabetic and Non-Diabetic classification)")
plt.show()


# ROC Curve + AUC
roc_disp = RocCurveDisplay.from_predictions(y_test, y_pred_selected)
plt.title("ANOVA Test: ROC Curve (kNN)")
plt.show()
roc_auc = roc_auc_score(y_test, y_pred_selected)
print("\nAvg  Anova test ROC: ", roc_auc)

# Precision–Recall Curve + PR-AUC
pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred_selected)
plt.title("Anova test: Precision–Recall Curve (kNN)")
plt.show()
pr_auc_ap = average_precision_score(y_test, y_pred_selected)
print("\nAvg Anova test Precision-Recall Curve: ", pr_auc_ap)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("\nSpecificity value: ", specificity)


