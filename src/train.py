import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)
    return pred


def compute_metrics_by_slice(
        data,
        model,
        categorical_features,
        target_feature):
    """
    Computes the performance metrics of a model on slices of the data
      based on categorical features.

    Inputs:
    - data: pandas DataFrame containing the data with categorical
    features and the target feature.
    - categorical_features: list of categorical feature names.
    - target_feature: name of the target feature.

    Returns:
    - slice_metrics: pandas DataFrame containing the performance
    metrics for each slice.
    """

    slice_metrics = []

    for feature in categorical_features:
        unique_values = data[feature].unique()
        for value in unique_values:
            # Create a slice of data for the current value of the feature
            slice_data = data[data[feature] == value]
            if len(slice_data) < 2:
                slice_metrics.append({
                    'Feature': feature,
                    'Value': None,
                    'Precision': None,
                    'Recall': None,
                    'F1': None
                })
                continue

            X_slice, y_slice, _, _ = process_data(
                slice_data, categorical_features=categorical_features,
                label=target_feature, training=True, encoder=None, lb=None)
            X_train, X_test, y_train, y_test = train_test_split(
                X_slice, y_slice, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)
            # Perform model inference on the slice
            y_pred = inference(model, X_test)

            # Compute performance metrics for the slice
            precision, recall, f1 = compute_model_metrics(y_test, y_pred)

            # Store the metrics for the slice
            slice_metrics.append({
                'Feature': feature,
                'Value': value,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

    # Create a DataFrame from the list of slice metrics
    slice_metrics_df = pd.DataFrame(slice_metrics)

    return slice_metrics_df


if __name__ == "__main__":
    df = pd.read_csv('../data/census.csv')
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
    X, y, encoder, lb = process_data(
        X=df, categorical_features=categorical, label=target,
        training=True, encoder=None, lb=None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    classifier = train_model(X_train, y_train)
    y_pred = inference(classifier, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    slice_metrics = compute_metrics_by_slice(
        df,
        classifier,
        categorical_features=categorical,
        target_feature=target)

    encoder_path = "./encoder.pkl"
    with open(encoder_path, 'wb') as file:
        pickle.dump(encoder, file)

    file_path = "./model.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(classifier, file)

    output_file = 'slice_output.txt'
    with open(output_file, 'w') as file:
        file.write(slice_metrics.to_string(index=False))
