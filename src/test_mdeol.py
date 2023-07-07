import os
import pickle
import pytest
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import process_data


@pytest.fixture(scope='module')
def setup_data():
    df = pd.read_csv('../../data/census.csv')
    categorical = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country']
    target = 'salary'
    X, y, _, _ = process_data(X=df, categorical_features=categorical,
                              label=target,
                              training=True, encoder=None, lb=None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    fbeta = fbeta_score(y_test, y_pred, beta=1, zero_division=1)
    file_path = "./model.pkl"

    return X_train, X_test, y_train, y_test, classifier, \
        y_pred, precision, recall, fbeta, file_path


def test_train_model(setup_data):
    X_train, _, y_train, _, _, _, _, _, _, _ = setup_data
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.estimators_


def test_compute_model_metrics(setup_data):
    _, _, _, y_test, _, y_pred, precision, recall, fbeta, _ = setup_data
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_inference(setup_data):
    _, X_test, _, _, classifier, y_pred, _, _, _, _ = setup_data
    y_pred = pd.Series(y_pred)
    assert len(y_pred) == len(X_test)
    assert isinstance(y_pred, pd.Series)


def test_model_save(setup_data):
    _, _, _, _, classifier, _, _, _, _, file_path = setup_data
    with open(file_path, 'wb') as file:
        pickle.dump(classifier, file)
    assert os.path.exists(file_path)


if __name__ == '__main__':
    pytest.main()
