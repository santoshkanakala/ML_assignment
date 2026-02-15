import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 

def load_and_preprocess():
  #Loading the dataset 
  df = pd.read_csv('student-por.csv', sep =';')
  #Preprocessing 
  #Creating binary target: Pass(1) if G3 > 10, else Fail (0)
  df['target'] = (df['G3'] >= 10).astype(int)
  df = df.drop(['G1', 'G2', 'G3'], axis =1)

  #Encode categorical varibles 
  le = LabelEncoder()
  for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

  X = df.drop('target', axis = 1)
  y = df['target']
  #split 80% test and 2- 20%train"

  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =42)

  #scaling 
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled, y_train, y_test 
