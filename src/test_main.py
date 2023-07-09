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
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174.0,
        "capital_loss": 0.0,
        "hours_per_week": 40,
        "native_country": "Cuba"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    prediction = response.json()["prediction"]
    assert prediction[0] == 0 or prediction[0] == 1


def test_positive_inference():
    data = {
        'age': 42,
        'workclass': 'Private',
        'fnlgt': 185900,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 50000,
        'capital_loss': 0,
        'hours_per_week': 45,
        'native_country': 'United-States'
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}


def test_negative_inference():
    data = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 9,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174.0,
        "capital_loss": 0.0,
        "hours_per_week": 40,
        "native_country": "Cuba"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": [0]}
