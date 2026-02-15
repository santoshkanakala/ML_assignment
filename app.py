import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle

st.set_page_config(page_title = "ML Assignment -2", layout ="wide")
st.title("Student Performance Prediction Dashboard")
st.markdown("Demonstarting ML claddification models")

#Model selection
st.sidebar.header("Model selection")
model_choice = st.sidebar.selectbox(
  "Select ML model",
  ("Logistic Regression", "Decision Tree" , "kNN", "Naive Bayes", "Random Forest", "XGBoost"))

# Data set upload feature 
st.subheader("step 1: upload Test Data set")
uploaded_file = st.file_uploader("Upload 'student-por.csv' ", type ="csv")

#MFeature: Model selection

def get_trained_model(name, X_train, y_train):
  if name == "Logistic Regression" :
    model = LogisticRegression(max_iter = 1000)
  elif name == "Decision Tree":
    model = DecisionTreeClassifier(random_state = 42)
  elif name == "kNN":
    model = KNeighborsClassifier(n_neighbors= 5)
  elif name ==  "Naive Bayes":
    model = GaussianNB()
  elif name == "Random Forest":
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
  elif name == "XGBoost":
    model = XGBClassifier(use_label_encoder= False, eval_metric= 'logloss')

  model.fit(X_train, y_train)
  return model

file_name =''
if uploaded_file is not None:
  file_name = uploaded_file
else:
  file_name = "student-por.csv"
try:
  data = pd.read_csv(file_name, sep=';')
  if data.shape[1] < 2:
    data = pd.read_csv(file_name, sep=',')
except:
  data = pd.read_csv(file_name)

st.write("Preview of Test Data:", data.head())

if 'G3' in data.columns:
  data['target'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)
  X = data.drop(['G3', 'target'], axis=1)
  if 'G1' in X.columns: X = X.drop(['G1', 'G2'], axis=1)
  y = data['target']
  le = LabelEncoder()
  for col in X.select_dtypes(include='object').columns:
    X[col] = le.fit_transform(X[col])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  with st.spinner(f" Training {model_choice}.."):
    model = get_trained_model(model_choice, X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
      y_prob = model.predict_proba(X_test)[:, 1]
    else:
      y_prob = y_pred
  #Display evaluation metrics
  st.subheader(f"Results for {model_choice}")
  me1,me2, me3, me4, me5, me6 = st.columns(6)
  me1.metric("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
  try:
    me2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
  except:
    me2.metric("AUC", "N/A")
  me3.metric(f"Precison" ,f"{precision_score(y_test, y_pred):.4f}")
  me4.metric(f"Recall", f"{recall_score(y_test, y_pred):.4f}")
  me5.metric(f"F1",f"{f1_score(y_test, y_pred):.4f}")
  me6.metric(f"MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
  #Confusion matrix and report
  st.divider()
  col_left, col_right = st.columns(2)
  with col_left:
    st.write("---Confusion Matrix-----")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot = True, fmt='d', cmap= 'Greens', ax= ax)
    st.pyplot(fig)
  with col_right:
    st.write("---Classification Report -----")
    report = classification_report(y_test, y_pred, output_dict = True)
    st.table(pd.DataFrame(report).transpose())
else:
  st.error("Error: The csv must contain a 'G3' column for evaluation.")
