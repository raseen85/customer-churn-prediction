from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load saved model and preprocessing objects
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    # Convert incoming JSON into a dataframe
    input_df = pd.DataFrame([data])

    # One-hot encode the same way training data was encoded
    input_df = pd.get_dummies(input_df)

    # Add any missing columns (that existed in training but not in this input) as 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column order matches training exactly
    input_df = input_df[model_columns]

    # Scale the numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }