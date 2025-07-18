from flask import Flask, render_template, jsonify, request
import joblib

app = Flask(__name__)
pump_state = 'OFF'
mode = 'manual'

latest_data = {
    "rain": 0,
    "soil": 0,
    "temp": 0,
    "hum": 0,
    "predicted": "-",
    "action": "OFF"
}

# Load AI model and encoders
try:
    model = joblib.load('irrigation_model.pkl')
    scaler = joblib.load('scaler.pkl')
    region_encoder = joblib.load('region_encoder.pkl')
    crop_encoder = joblib.load('crop_encoder.pkl')
    print("✅ AI model and preprocessors loaded.")
except Exception as e:
    model = None
    print(f"❌ Failed to load model or encoders: {e}")

@app.route('/')
def dashboard():
    regions = list(region_encoder.classes_) if region_encoder else []
    crops = list(crop_encoder.classes_) if crop_encoder else []
    return render_template('index.html', regions=regions, crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        soil = data['soil_moisture']
        ph = data['soil_ph']
        temp = data['temperature']
        rain = data['rainfall']
        hum = data['humidity']
        sunlight = data['sunlight']
        region = data['region']
        crop = data['crop']

        region_enc = region_encoder.transform([region])[0]
        crop_enc = crop_encoder.transform([crop])[0]

        features = [[soil, ph, temp, rain, hum, sunlight, region_enc, crop_enc]]
        scaled = scaler.transform(features)
        prediction = round(max(0, model.predict(scaled)[0]), 2)

        recommendation = "Irrigation Required" if prediction > 800 else "No Need for Irrigation"
        context_notes = []

        if rain > 80:
            context_notes.append("⚠️ Rainfall is high. Consider delaying irrigation.")
        if soil > 80:
            context_notes.append("🌱 Soil moisture is high. Irrigation may not be necessary.")

        return jsonify(success=True, water_needed=prediction, recommendation=recommendation, context_notes=context_notes)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400

@app.route('/esp_data', methods=['POST'])
def esp_data():
    global latest_data, pump_state
    try:
        data = request.get_json()
        rain = int(data['rain'])
        soil = int(data['soil_moisture'])
        temp = float(data['temperature'])
        hum = float(data['humidity'])

        print(f"[ESP] Temp={temp}, Hum={hum}, Soil={soil}, Rain={rain}")

        predicted = "-"
        action = pump_state

        if mode == 'ai' and model:
            region = region_encoder.classes_[0]
            crop = crop_encoder.classes_[0]
            region_enc = region_encoder.transform([region])[0]
            crop_enc = crop_encoder.transform([crop])[0]

            features = [[soil, 7.0, temp, rain, hum, 5.0, region_enc, crop_enc]]
            scaled = scaler.transform(features)
            predicted = round(max(0, model.predict(scaled)[0]), 2)

            # Optional override for rain > 80
            # if rain > 80:
            #     print("🌧️ Rain > 80% — overriding AI, pump OFF.")
            #     action = "OFF"
            # else:
            action = "ON" if predicted > 800 else "OFF"

            pump_state = action

        latest_data.update({
            "rain": rain,
            "soil": soil,
            "temp": temp,
            "hum": hum,
            "predicted": predicted,
            "action": action
        })
        return jsonify(success=True)
    except Exception as e:
        print(f"❌ Error in /esp_data: {e}")
        return jsonify(success=False, error=str(e)), 400

@app.route('/esp_command')
def esp_command():
    return jsonify(pump=pump_state)

@app.route('/data')
def get_data():
    return jsonify({
        "rain": latest_data["rain"],
        "soil": latest_data["soil"],
        "temp": latest_data["temp"],
        "hum": latest_data["hum"],
        "action": latest_data["action"],
        "mode": mode.upper(),
        "predicted": latest_data["predicted"]
    })

@app.route('/control', methods=['POST'])
def control():
    global pump_state, mode
    data = request.get_json()
    cmd = data.get('command')

    if cmd == 'ON':
        pump_state = 'ON'
    elif cmd == 'OFF':
        pump_state = 'OFF'
    elif cmd == 'TOGGLE_MODE':
        mode = 'ai' if mode == 'manual' else 'manual'
    elif cmd == 'RESET' or cmd is None:
        pump_state = 'OFF'
        mode = 'manual'

    return jsonify(success=True, pump=pump_state, mode=mode)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
