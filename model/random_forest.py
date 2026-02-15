import pandas as pd 
import numpy as np
from utils import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef 

X_train, X_test, y_train, y_test = load_and_preprocess()

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

#predict 
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("---- Random  Forest  Metrics----")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precison : {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
