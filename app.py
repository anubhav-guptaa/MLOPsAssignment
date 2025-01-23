from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Fine-tuned Naive Bayes model
model = joblib.load('model/tuned_naive_bayes_model.pkl')

@app.route('/')
def home():
    return "Model is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting a JSON payload with input data
    
    # Get input data
    input_data = np.array(data['input']).reshape(1, -1)  # Assuming input is a list of features
    
    # Predict using the model
    prediction = model.predict(input_data)
    
    # Return the result as a JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
