import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pytest
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Check for missing values in the dataset
def check_missing_values(data):
    return data.isnull().sum()

# Convert 'date' column to datetime
def convert_date_column(data):
    data['date'] = pd.to_datetime(data['date'])

# Visualization Functions
def plot_weather_distribution(data):
    plt.figure(figsize=(10, 5))
    sns.set_theme()
    sns.countplot(x='weather', data=data, palette="ch:start=.2,rot=-.3")
    plt.xlabel("Weather", fontweight='bold', size=13)
    plt.ylabel("Count", fontweight='bold', size=13)
    plt.show()

def plot_temperature_variation(data):
    # Plot max temperature variation
    px.line(data_frame=data, x='date', y='temp_max', title='Variation of Maximum Temperature').show()

    # Plot min temperature variation
    px.line(data_frame=data, x='date', y='temp_min', title='Variation of Minimum Temperature').show()

def plot_temp_by_weather(data):
    # Plot max temp by weather condition
    plt.figure(figsize=(10, 5))
    sns.catplot(x='weather', y='temp_max', data=data, palette="crest")
    plt.show()

    # Plot min temp by weather condition
    plt.figure(figsize=(10, 5))
    sns.catplot(x='weather', y='temp_min', data=data, palette="RdBu")
    plt.show()

# Label Encoding Function
def label_encoding(data, column_name):
    label_encoder = preprocessing.LabelEncoder()
    data[column_name] = label_encoder.fit_transform(data[column_name])
    return data[column_name].unique()

# Preprocessing and Model Training
def preprocess_data(data):
    # Drop the 'date' column
    data = data.drop('date', axis=1)
    
    # Split data into features and target variable
    X = data.drop('weather', axis=1)
    y = data['weather']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot confusion matrix
    # sns.heatmap(cm, annot=True, fmt='.3g')
    # plt.show()
    
    return accuracy

# Logistic Regression Model
def logistic_regression(X_train, y_train):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# Naive Bayes Model
def naive_bayes(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

# Hyperparameter tuning for Naive Bayes
def tune_naive_bayes(X_train, y_train):
    # Define the hyperparameters to tune
    param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

    # Create a Naive Bayes model
    nb = GaussianNB()

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    # Create 'model' directory if it does not exist
    if not os.path.exists('model'):
        os.makedirs('model')
    
    model_path = os.path.join('model', 'tuned_naive_bayes_model.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Tuned Naive Bayes model saved as '{model_path}'")

    # Return the best model
    return grid_search.best_estimator_


# SVM Model
def svm_model(X_train, y_train):
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# Main function to run the entire workflow
def main():
    # Load the data
    data = load_data("./data/weather.csv")
    
    # Data preprocessing
    check_missing_values(data)
    convert_date_column(data)
    
    # Visualizations
    # plot_weather_distribution(data)
    # plot_temperature_variation(data)
    # plot_temp_by_weather(data)
    
    # Label Encoding
    label_encoding(data, "weather")
    
    # Preprocess data for model training
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train models and evaluate
    # Logistic Regression
    print("Logistic Regression:")
    classifier = logistic_regression(X_train, y_train)
    acc_log_reg = evaluate_model(classifier, X_test, y_test)
    print(f"Accuracy score: {acc_log_reg}\n")

    # SVM
    print("Support Vector Machine (SVM):")
    classifier = svm_model(X_train, y_train)
    acc_svm = evaluate_model(classifier, X_test, y_test)
    print(f"Accuracy score: {acc_svm}\n")

    # Naive Bayes
    print("Naive Bayes:")
    classifier = naive_bayes(X_train, y_train)
    acc_naive_bayes = evaluate_model(classifier, X_test, y_test)
    print(f"Accuracy score: {acc_naive_bayes}\n")
    
    # Naive Bayes with hyperparameter tuning
    print("Tuned Naive Bayes:")
    classifier = tune_naive_bayes(X_train, y_train)
    acc_naive_bayes = evaluate_model(classifier, X_test, y_test)

if __name__ == "__main__":
    main()
