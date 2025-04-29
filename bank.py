import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\New folder\Bank Customer Churn Prediction.csv')


le_country = LabelEncoder()
le_gender = LabelEncoder()
df['country'] = le_country.fit_transform(df['country'])
df['gender'] = le_gender.fit_transform(df['gender'])


X = df.drop(['churn', 'customer_id', 'credit_score'], axis=1)
y = df['churn']


sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


lr = LogisticRegression(max_iter=10000)
dt = DecisionTreeClassifier(max_depth=7, random_state=42)
rf = RandomForestClassifier(n_estimators=500, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)


lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)


def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    st.subheader(f"Results for {model_name}")
    st.write("### Classification Report")
    st.text(cr)

    st.write("### Confusion Matrix")
    st.write(cm)

    st.write("### ROC Curve")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    st.pyplot(plt)


st.title("Bank Customer Churn Prediction")
st.write("Enter the customer details:")


country = st.selectbox("Country", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0, 100000, 50000)
products_number = st.number_input("Number of Products", 1, 4, 2)
credit_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", 20000, 200000, 50000)


country = le_country.transform([country])[0]
gender = le_gender.transform([gender])[0]
credit_card = 1 if credit_card == 'Yes' else 0
active_member = 1 if active_member == 'Yes' else 0


input_data = np.array([country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]).reshape(1, -1)
input_data_scaled = sc.transform(input_data)


if st.button("Predict with Logistic Regression"):
    prediction = lr.predict(input_data_scaled)
    churn_status = "Yes" if prediction[0] == 1 else "No"
    st.write(f"### Churn Prediction (Logistic Regression): {churn_status}")
    evaluate_model(lr, "Logistic Regression")

if st.button("Predict with Decision Tree"):
    prediction = dt.predict(input_data_scaled)
    churn_status = "Yes" if prediction[0] == 1 else "No"
    st.write(f"### Churn Prediction (Decision Tree): {churn_status}")
    evaluate_model(dt, "Decision Tree")

if st.button("Predict with Random Forest"):
    prediction = rf.predict(input_data_scaled)
    churn_status = "Yes" if prediction[0] == 1 else "No"
    st.write(f"### Churn Prediction (Random Forest): {churn_status}")
    evaluate_model(rf, "Random Forest")

if st.button("Predict with XGBoost"):
    prediction = xgb_model.predict(input_data_scaled)
    churn_status = "Yes" if prediction[0] == 1 else "No"
    st.write(f"### Churn Prediction (XGBoost): {churn_status}")
    evaluate_model(xgb_model, "XGBoost")
