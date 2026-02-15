a.Problem Statement:

The objective of this project is to implement and evaluate six different machine learning classification models to predict student passing status(Pass/ Fail). Using the UCI student Performance Dataset, the goal is to identify whether a studen will pass ( Grade >=10) or fail based on various demographic, social and academic features. The project involves building an end- to-end ML workflow, from model training and metric evaluation to deployment on Streamlit.

b. Dataset Description 
. Source: UCI Machine leanring repostitory - Student Performance(https://archive.ics.uci.edu/dataset/320/student+performance).
. Target Variable: Pass_status (Binary : 1 for pass , 0 for fail)
. Mininum feature size - 30 which meets requirement greater than 12
. Minimun instant size - 649 (> 500 satisfies the requirement)
. Features included - Student age, parental educationm weekly study timem failures, alcohol consumptionm and health status.

c. Models used and comparision tables:
Attaching the results from the executed results of respective models

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8769 | 0.8116 | 0.9160 | 0.9478 | 0.9316 | 0.3228 |
| **Decision Tree** | 0.8538 | 0.6565 | 0.9211 | 0.9130 | 0.9170 | 0.3044 |
| **kNN** | 0.8615 | 0.7362 | 0.8880 | 0.9652 | 0.9250 | 0.0530 |
| **Naive Bayes** | 0.8692 | 0.7658 | 0.9375 | 0.9130 | 0.9251 | 0.4129 |
| **Random Forest (Ensemble)** | 0.8769 | 0.7597 | 0.8898 | 0.9826 | 0.9339 | 0.1048 |
| **XGBoost (Ensemble)** | 0.8846 | 0.7548 | 0.9098 | 0.9652 | 0.9367 | 0.3083 |


d. Performance Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest AUC (0.8116), indicating superior ability to distinguish between Pass and Fail status. |
| **Decision Tree** | While providing high Precision (0.9211), it had the lowest AUC (0.6565), suggesting it is less reliable across different thresholds. |
| **kNN** | Demonstrated very high Recall (0.9652) but the lowest MCC (0.0530), indicating difficulty in identifying the minority class (Fail). |
| **Naive Bayes** | Achieved the highest MCC (0.4129), showing the best-balanced performance for both classes in this specific dataset. |
| **Random Forest (Ensemble)** | Delivered the highest Recall (0.9826), making it the most effective model for capturing all students who will pass. |
| **XGBoost (Ensemble)** | **Overall Best Performer** in terms of Accuracy (0.8846) and F1-Score (0.9367), providing the most robust general predictions. |
