from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract original 7 inputs
    step = float(data['step'])
    type_code = float(data['type'])  # already numeric from the HTML
    amount = float(data['amount'])
    oldbalanceOrg = float(data['oldbalanceOrg'])
    newbalanceOrig = float(data['newbalanceOrig'])
    oldbalanceDest = float(data['oldbalanceDest'])
    newbalanceDest = float(data['newbalanceDest'])

    # --- ✅ Feature Engineering ---
    balance_diff = oldbalanceOrg - newbalanceOrig
    transfer_ratio = amount / oldbalanceOrg if oldbalanceOrg > 0 else 0
    net_transfer = (oldbalanceOrg - newbalanceOrig) - (newbalanceDest - oldbalanceDest)

    # Create full feature vector
    features = np.array([[step, type_code, amount, oldbalanceOrg, newbalanceOrig,
                          oldbalanceDest, newbalanceDest, balance_diff, transfer_ratio, net_transfer]])

    # Load scaler and model
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('fraud_model.pkl')

    # Scale features and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Result label
    result = "❌ Illegitimate Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
