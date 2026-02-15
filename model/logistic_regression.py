from sklearn.linear_model import LogisticRegression
from utils import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

#predict 
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("---- Logistic Regression Metrics----")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precison : {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
