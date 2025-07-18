from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)
model = None
scaler = None
region_encoder = None
crop_encoder = None
model_info = {}

def load_and_train_model():
    """Load data, train model, and save preprocessors"""
    global model, scaler, region_encoder, crop_encoder, model_info
    
    print("Loading dataset from Excel file...")
    df = pd.read_excel('alphachk.xlsx')
    
    print(f"Original dataset shape: {df.shape}")
    print("Columns in dataset:", df.columns.tolist())
    columns_to_keep = [
        'region', 'crop_type', 'soil_moisture_%', 'soil_pH', 
        'temperature_C', 'rainfall_mm', 'humidity_%', 
        'sunlight_hours', 'water_final'
    ]
    
    df_clean = df[columns_to_keep].copy()
    df_clean = df_clean.dropna()
    
    print(f"Clean dataset shape: {df_clean.shape}")
    region_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    
    df_clean['region_encoded'] = region_encoder.fit_transform(df_clean['region'])
    df_clean['crop_type_encoded'] = crop_encoder.fit_transform(df_clean['crop_type'])
    
    print("Unique regions:", region_encoder.classes_)
    print("Unique crops:", crop_encoder.classes_)
    X = df_clean[['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm', 
                  'humidity_%', 'sunlight_hours', 'region_encoded', 'crop_type_encoded']]
    y = df_clean['water_final']
    
    print(f"Features shape: {X.shape}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f} ml")
    print(f"Zero water cases: {(y == 0).sum()} out of {len(y)} total cases")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42),
        'Decision Tree': DecisionTreeRegressor(min_samples_split=10, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.1),
        'Lasso Regression': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    best_mse = float('inf')
    best_model_name = None
    
    print("\nTraining and comparing models:")
    print("=" * 50)
    
    for name, model_candidate in models.items():
        print(f"Training {name}...")
        model_candidate.fit(X_train_scaled, y_train)
        y_pred = model_candidate.predict(X_test_scaled)
        y_pred = np.maximum(0, y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name}: MSE = {mse:.2f}, R² = {r2:.4f}")
        
        if mse < best_mse:
            best_mse = mse
            best_model_name = name
            model = model_candidate
    
    model_info = {
        'name': best_model_name,
        'mse': best_mse,
        'r2': r2_score(y_test, np.maximum(0, model.predict(X_test_scaled))),
        'training_samples': len(X_train)
    }
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 MSE: {best_mse:.2f}")
    print(f"📈 R²: {model_info['r2']:.4f}")
    joblib.dump(model, 'irrigation_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(region_encoder, 'region_encoder.pkl')
    joblib.dump(crop_encoder, 'crop_encoder.pkl')
    
    print("✅ Model and preprocessors saved successfully!")

def load_saved_model():
    """Load previously saved model and preprocessors"""
    global model, scaler, region_encoder, crop_encoder, model_info
    
    try:
        model = joblib.load('irrigation_model.pkl')
        scaler = joblib.load('scaler.pkl')
        region_encoder = joblib.load('region_encoder.pkl')
        crop_encoder = joblib.load('crop_encoder.pkl')
        model_info = {
            'name': type(model).__name__,
            'mse': 'Loaded from file',
            'r2': 'Loaded from file',
            'training_samples': 'Loaded from file'
        }
        
        print("✅ Loaded existing model from files")
        return True
    except FileNotFoundError:
        print("❌ No saved model found, will train new model")
        return False

@app.route('/')
def index():
    regions = list(region_encoder.classes_) if region_encoder else []
    crops = list(crop_encoder.classes_) if crop_encoder else []
    
    return render_template('index.html', regions=regions, crops=crops, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        soil_moisture = float(data['soil_moisture'])
        soil_ph = float(data['soil_ph'])
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])
        sunlight = float(data['sunlight'])
        region = data['region']
        crop = data['crop']
        if not all([region, crop]):
            return jsonify({
                'success': False,
                'error': 'Please select both region and crop type'
            })
        try:
            region_encoded = region_encoder.transform([region])[0]
            crop_encoded = crop_encoder.transform([crop])[0]
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid region or crop selection: {str(e)}'
            })
        input_data = np.array([[soil_moisture, soil_ph, temperature, rainfall, 
                               humidity, sunlight, region_encoded, crop_encoded]])
        input_scaled = scaler.transform(input_data)

        water_needed = model.predict(input_scaled)[0]
        water_needed = max(0, round(water_needed, 2))
        if soil_moisture > 40 and rainfall >= 0 and humidity > 35:
           
            if soil_moisture > 45 and humidity > 40:
                water_needed = min(water_needed, 100)  #
        if water_needed < 100:
            recommendation = "No irrigation needed. Soil conditions are adequate."
            irrigation_level = "None"
            irrigation_color = "green"
        elif water_needed < 800:
            recommendation = "Low water requirement. Light irrigation recommended."
            irrigation_level = "Low"
            irrigation_color = "lightgreen"
        elif water_needed < 1200:
            recommendation = "Moderate water requirement. Regular irrigation needed."
            irrigation_level = "Moderate"
            irrigation_color = "orange"
        elif water_needed < 1600:
            recommendation = "High water requirement. Frequent irrigation necessary."
            irrigation_level = "High"
            irrigation_color = "red"
        else:
            recommendation = "Very high water requirement. Intensive irrigation required."
            irrigation_level = "Very High"
            irrigation_color = "darkred"
        
        
        context_notes = []
        
        if soil_moisture < 20:
            context_notes.append("⚠️ Very low soil moisture detected")
        elif soil_moisture > 50:
            context_notes.append("💧 High soil moisture - minimal irrigation needed")
            
        if temperature > 35:
            context_notes.append("🌡️ High temperature increases water evaporation")
        elif temperature < 15:
            context_notes.append("❄️ Cool temperature reduces water needs")
            
        if rainfall > 10:
            context_notes.append("🌧️ Recent rainfall reduces irrigation needs")
            
        if humidity < 40:
            context_notes.append("🏜️ Low humidity increases plant water stress")
        elif humidity > 80:
            context_notes.append("💨 High humidity reduces water loss")
        
        return jsonify({
            'success': True,
            'water_needed': water_needed,
            'recommendation': recommendation,
            'irrigation_level': irrigation_level,
            'irrigation_color': irrigation_color,
            'model_used': model_info.get('name', 'Unknown'),
            'context_notes': context_notes
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })

if __name__ == '__main__':
    print("🌱 Starting Smart Irrigation ML System...")
    
    if not load_saved_model():
        print("🔄 Training new model...")
        load_and_train_model()
    
    print("🚀 Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
