import pickle
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(debug=True)

class Data(BaseModel):
    """JSON string containing the 32 features (exactly 32)"""
    data: Dict[str, float] = Field(..., example={"data":{"id": 859575.0, "radius_mean": 18.94, "texture_mean": 21.31, "perimeter_mea": 123.6, "area_mean": 1130.0, "smoothness_mean": 0.09009, "compactness_mean": 0.1029, "concavity_mean": 0.108, "concave points_mean": 0.07951, "symmetry_mean": 0.1582, "fractal_dimension_mean": 0.05461, "radius_se": 0.7888, "texture_se": 0.7975, "perimeter_se": 5.486, "area_se": 96.05, "smoothness_se": 0.004444, "compactness_se": 0.01652, "concavity_se": 0.02269, "concave ponts_se": 0.0137, "symmetry_se": 0.01386, "fractal_dimension_se": 0.001698, "radius_worst": 24.86, "texture_worst": 26.58, "perimeter_worst": 165.9, "area_worst": 1.0, "smoothness_worst": 0.1193, "compactness_worst": 0.2336, "concavity_worst": 0.2687, "concave points_worst": 0.1789, "symmetry_worst": 0.2551, "fractal_dimension_worst": 0.06589}})

class Prediction(BaseModel):
    """Prediction result: Benign or Malignant"""
    prediction: Dict[str, str] = Field(..., example={"prediction": "This looks Benign"})

def load_model():
    return pickle.load(open("model.pkl", "rb"))

def pred_to_word(prediction):
    if prediction == "B":
        return "This looks Benign"
    else:
        return "This looks Malignant"

@app.get("/health")
async def health_check():
    # just a health check endpoint
    return {"status": "healthy and working like a charmmmm"}

@app.post("/predict", response_model=Prediction)
async def predict(data: Data):
    # wil make the prediction here
    feature_array = list(data.data.values())
    if len(feature_array) != 32:
        return HTTPException(status_code=400, detail="Please provide exactly 32 features.")

    prediction = model.predict([feature_array])
    final_verdict = pred_to_word(prediction[0])
    return {"prediction": final_verdict}

model = load_model()