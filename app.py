from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

@app.route("/")
def index():
    return "Welcome to Covid19 Prediction App".capitalize()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Prepare input data
        input_data = []
        for feature in features:
            value = request.json.get(feature, 0)
            # Convert to binary (1/0)
            if isinstance(value, str):
                value = 1 if value.upper() in ['YES', '1', 'TRUE'] else 0
            input_data.append(int(value))
        
        # Predict
        input_df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'result': 'POSITIVE' if prediction == 1 else 'NEGATIVE',
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)