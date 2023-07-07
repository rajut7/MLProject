from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_get_prediction():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to customer prediction"}


def test_ml_inference():
    data = {
        "age": 30,
        "workclass": "Private",
        "fnigt": 77516,
        "education": "Bachelors",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174.0,
        "capital_loss": 0.0,
        "hours_per_work": 40,
        "native_country": "Cuba"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    prediction = response.json()["prediction"]
    assert prediction[0] == 0 or prediction[0] == 1


def test_negative_inference():
    data = {
        "age": 40,
        "workclass": "Private",
        "fnigt": 123456,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0.0,
        "capital_loss": 0.0,
        "hours_per_work": 35,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [0]}
