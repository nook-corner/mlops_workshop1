# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

# ====== CONFIG ======
#MLFLOW_TRACKING_URI = "http://139.162.39.5:5000"  # แก้เป็น IP Linode ของคุณ
EXPERIMENT_NAME = "loan-prediction"

# ====== STEP 1: Load data from DVC ======
df = pd.read_csv("data/loan_train.csv")

# ====== STEP 2: Data Cleaning ======
# แทนค่าที่หายไปด้วยค่าที่เหมาะสม
for col in ["Gender", "Married", "Self_Employed", "Dependents"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# เก็บค่าเติม missing ที่ใช้ตอน train (คำนวณจาก df ก่อน get_dummies หรือจาก df หลังทำความสะอาดก็ได้)
fill_values = {}
for col in ["Gender", "Married", "Self_Employed", "Dependents"]:
    fill_values[col] = df[col].mode()[0] if col in df.columns else None

if "LoanAmount" in df.columns:
    fill_values["LoanAmount"] = float(df["LoanAmount"].median())
if "Loan_Amount_Term" in df.columns:
    fill_values["Loan_Amount_Term"] = float(df["Loan_Amount_Term"].median())
if "Credit_History" in df.columns:
    # ใช้ mode สำหรับคอลัมน์นี้เหมือนตอน train
    fill_values["Credit_History"] = float(df["Credit_History"].mode()[0])





# ====== STEP 3: Split X, y ======
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]


for col in X.columns:
    if pd.api.types.is_integer_dtype(X[col]):
        X[col] = X[col].astype(float)


# รายชื่อคอลัมน์หลัง encode
model_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== STEP 4: Scale features ======
scaler = StandardScaler()

# เก็บชื่อ column เดิมไว้
feature_names = X_train.columns

# Scale และแปลงกลับเป็น DataFrame พร้อมชื่อ column
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)


# ทำโฟลเดอร์ model แน่ใจว่ามี
Path("model").mkdir(parents=True, exist_ok=True)

with open("model/model_columns.json", "w") as f:
    json.dump(model_columns, f)

with open("model/fill_values.json", "w") as f:
    json.dump(fill_values, f)

# ====== STEP 5: MLflow Setup ======
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# ถ้าอยู่ใน GitHub Actions  เก็บ artifact ใน workspace ของ runner
if os.getenv("GITHUB_ACTIONS") == "true":
    workspace_dir = os.getenv("GITHUB_WORKSPACE", ".")
    mlruns_path = os.path.join(workspace_dir, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
else:
    mlflow.set_tracking_uri("http://139.162.39.5:5000")  # MLflow server บน Linod



mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    # Train Model
    model = LogisticRegression(max_iter=500)  # เพิ่ม max_iter
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 500)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Input example for signature
    example_input = pd.DataFrame([X.iloc[0].to_dict()])

    # Log model with new MLflow API
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=example_input
    )
    # ===== Save artifacts locally =====
    # Save model locally
    joblib.dump(model, "model/loan_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")  # เก็บ scaler ไว้ใช้ตอน deploy

print(f"Training completed. Accuracy: {acc:.4f}")
