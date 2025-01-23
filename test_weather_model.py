import pytest
import pandas as pd
from train import (
    load_data,
    check_missing_values,
    convert_date_column,
    label_encoding,
    preprocess_data,
    logistic_regression,
    evaluate_model
)

# Test for Data Loading
def test_load_data():
    data = load_data("./data/weather.csv")
    assert isinstance(data, pd.DataFrame), "Data is not loaded as a DataFrame"
    assert not data.empty, "Dataframe is empty"

# Test for Missing Values
def test_check_missing_values():
    data = load_data("weather.csv")
    missing_values = check_missing_values(data)
    assert missing_values.sum() == 0, "There are missing values in the dataset"

# Test for Date Conversion
def test_convert_date_column():
    data = load_data("weather.csv")
    convert_date_column(data)
    assert pd.api.types.is_datetime64_any_dtype(data['date']), "Date column was not converted to datetime"

# Test for Logistic Regression Model Accuracy
def test_logistic_regression_accuracy():
    data = load_data("weather.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    classifier = logistic_regression(X_train, y_train)
    accuracy = evaluate_model(classifier, X_test, y_test)
    
    assert accuracy > 0.5, f"Accuracy for Logistic Regression is too low: {accuracy}"