import io
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

app = FastAPI()

# Load trained ML model and feature names
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# Input validation
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)


# Home endpoint
@app.get("/")
def home():
    return {
        "msg": "California House Prediction API",
        "status": "running",
        "endpoint": "POST /prediction or POST /predict-file"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }


# Single house prediction
@app.post("/prediction")
def prediction(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([house.model_dump()])
        input_data = input_data[features]

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": (
                f"${predicted:,.2f} hundred thousand"
            ),
            "confidence_range": (
                f"${max(0, price_usd - 39000):,.0f} "
                f"to ${price_usd + 39000:,.0f}"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Multiple house predictions from CSV
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only"
        )

    # Read CSV
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read CSV file. Please check the file format."
        )

    # Required columns
    required_columns = features

    # Check missing columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_columns}"
        )

    # Check empty file
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file has no data rows"
        )

    try:
        # Select columns in the correct order
        input_data = df[required_columns].copy()

        # Validate numeric values
        input_data = input_data.apply(
            pd.to_numeric, errors="raise"
        )

        if input_data.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains missing values"
            )

        # Predict house prices
        predictions = model.predict(input_data)

        # Convert predictions to USD
        df["predicted_price_usd"] = predictions * 100000

        # Format prices for readable CSV output
        df["predicted_price_usd"] = df[
            "predicted_price_usd"
        ].map(lambda x: f"${x:,.0f}")

        # Convert DataFrame to CSV
        output = df.to_csv(index=False)

        return StreamingResponse(
            io.BytesIO(output.encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"CSV prediction failed: {str(e)}"
        )

