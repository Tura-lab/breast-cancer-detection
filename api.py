import pickle
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Data(BaseModel):
    data: Dict[str, float]

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

@app.post("/predict")
async def predict(data: Data):
    # will make the prediction here
    prediction = model.predict([list(data.data.values())])
    
    final_verdict = pred_to_word(prediction[0])
    
    return {"prediction": final_verdict}

model = load_model()