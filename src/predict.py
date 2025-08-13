from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib, json
import uvicorn
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# โหลด artifacts
MODEL_PATH = Path("model/loan_model.pkl")
SCALER_PATH = Path("model/scaler.pkl")
COL_PATH = Path("model/model_columns.json")
FILL_PATH = Path("model/fill_values.json")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = json.loads(Path(COL_PATH).read_text())
fill_values   = json.loads(Path(FILL_PATH).read_text())

# สร้างสคีมารับอินพุต (ใส่ฟิลด์ยอดฮิตใน dataset)
class LoanInput(BaseModel):
    Gender: str | None = None
    Married: str | None = None
    Dependents: str | None = None
    Education: str | None = None
    Self_Employed: str | None = None
    ApplicantIncome: float | None = None
    CoapplicantIncome: float | None = None
    LoanAmount: float | None = None
    Loan_Amount_Term: float | None = None
    Credit_History: float | None = None
    Property_Area: str | None = None

def preprocess_one(sample: dict) -> pd.DataFrame:
    # 1) to DataFrame
    df = pd.DataFrame([sample])

    # 2) เติม missing ตามค่าที่เราบันทึกไว้
    for col, val in fill_values.items():
        if col in df.columns and val is not None:
            df[col] = df[col].fillna(val)

    # 3) one-hot ให้เหมือนตอน train
    df_enc = pd.get_dummies(df, drop_first=True)

    # 4) จัดคอลัมน์ให้ตรงกับตอน train (คอลัมน์ที่ไม่มีให้เติม 0)
    df_enc = df_enc.reindex(columns=model_columns, fill_value=0)

    # 5) scale โดยใช้คอลัมน์ตามลำดับเดียวกับ model_columns
    df_scaled = pd.DataFrame(scaler.transform(df_enc), columns=model_columns)

    return df_scaled

@app.post("/predict")
def predict(input_data: LoanInput):
    row = input_data.dict()
    X = preprocess_one(row)
    pred = model.predict(X)[0]
    proba = getattr(model, "predict_proba", None)
    p1 = float(proba(X)[0][1]) if proba else None
    return {
        "approved": bool(pred == 1),
        "probability_approved": p1
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
