# AgroEdge-Edge-Computing-for-Precision-Agriculture
AgroEdge uses edge computing and ML to analyze real-time data from sensors and drones, providing farmers with actionable insights to optimize crop yield and resource management.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Mock data to simulate sensor and drone input. In practice, this should be replaced with real-time data streams.
# Sample data structure: {'temperature': value, 'humidity': value, 'soil_moisture': value, 'drone_image_analysis': {'health': value, 'growth_stage': value}}
data = [
    {'temperature': 25, 'humidity': 80, 'soil_moisture': 20, 'drone_image_analysis': {'health': 0.8, 'growth_stage': 3}},
    {'temperature': 30, 'humidity': 50, 'soil_moisture': 12, 'drone_image_analysis': {'health': 0.6, 'growth_stage': 2}},
    # Add more data as needed.
]

# Convert the structured data into a DataFrame for easier manipulation and analysis.
def prepare_data(data):
    processed_data = []
    for entry in data:
        processed_entry = {
            'temperature': entry['temperature'],
            'humidity': entry['humidity'],
            'soil_moisture': entry['soil_moisture'],
            'health': entry['drone_image_analysis']['health'],
            'growth_stage': entry['drone_image_analysis']['growth_stage'],
        }
        processed_data.append(processed_entry)
    return pd.DataFrame(processed_data)

# Example function to train a model on the given data. This is a simplification.
def train_model(data):
    df = prepare_data(data)
    X = df.drop('health', axis=1) # Using all columns except 'health' as features.
    y = df['health'] # Predicting the 'health' as labeled by drone image analysis.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model MSE: {mse}")
    
    return model

# Assume the model has been trained outside of the Flask app for simplicity.
# model = train_model(data)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = request.json
            input_df = prepare_data([input_data])
            # Assuming the model is already trained and loaded. Replace 'model' with your loaded model variable.
            # prediction = model.predict(input_df)
            
            # Mock prediction to simulate the model's output.
            prediction = np.random.rand()
            
            return jsonify({'health_prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
