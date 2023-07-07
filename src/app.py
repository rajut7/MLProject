import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnigt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_work: int
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
    # X, _ = process_data(X=df, categorical_features=categorical, label=None, training=False, encoder=encoder, lb=lb)
    X_categorical = df[categorical]
    X_continuous = df.drop(columns=categorical)

    with open("encoder.pkl", 'rb') as file:
        encoder = pickle.load(file)

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    # Load the trained model
    with open("model.pkl", "rb") as pickle_in:
        model = pickle.load(pickle_in)

    # Perform prediction
    prediction = model.predict(X)

    return {"prediction": prediction.tolist()}
