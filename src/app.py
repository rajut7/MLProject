import pickle
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str


@app.get("/")
def root():
    return {"message": "Welcome to customer prediction"}


@app.post("/predict")
def predict(data: Data):
    df = pd.DataFrame([data.dict()])
    print(df)

    # Preprocess the data
    categorical = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country']

    # encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    # lb = LabelBinarizer()
    X_categorical = df[categorical]
    X_continuous = df.drop(columns=categorical)

    file_path = "./src/encoder.pkl"
    with open(file_path, 'rb') as file:
        encoder = pickle.load(file)

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    # Load the trained model
    model_path = "./src/model.pkl"
    with open(model_path, "rb") as pickle_in:
        model = joblib.load(pickle_in)

    # Perform prediction
    prediction = model.predict(X)

    return {"prediction": prediction.tolist()}
