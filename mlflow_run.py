import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature

# Ensure the mlruns directory exists
mlflow.set_tracking_uri("./mlruns")
os.makedirs("./mlruns", exist_ok=True)

# Set the experiment name (either use default or custom name)
experiment_name = "weather_models_experiment"  # Custom name for your experiment
mlflow.set_experiment(experiment_name)

# Load the dataset
data = pd.read_csv("./data/weather.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.drop('date', axis=1)
x = data.drop('weather', axis=1)
y = data['weather']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to log experiment with MLflow
def log_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run() as run:
        # Log model parameters
        mlflow.log_param("model_type", model_name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log model metrics
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model with signature (input-output signature) and example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[0:1]  # Example from training data
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)
        
        # Print run details
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model Type: {model_name}")
        print(f"Accuracy: {accuracy}")
        
        # Optionally, log confusion matrix as an artifact
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_filename = f"{model_name}_confusion_matrix.png"
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

# Logistic Regression Model
log_model(LogisticRegression(random_state=0), "LogisticRegression", X_train, y_train, X_test, y_test)

# Support Vector Classifier (SVC) Model
log_model(SVC(kernel='linear', random_state=0), "SVC", X_train, y_train, X_test, y_test)

# Naive Bayes Model
log_model(GaussianNB(), "NaiveBayes", X_train, y_train, X_test, y_test)

# Fetch and print all experiments and runs
experiments = mlflow.search_experiments(order_by=["name"])
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")
    
# Fetch and print all runs
runs = mlflow.search_runs(order_by=["start_time desc"])
