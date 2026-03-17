from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- import CORS
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # <-- allow requests from any origin

# Load model files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input data to dataframe
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)

        # Ensure all columns match training data
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        return jsonify({
            "result": "PASS" if prediction[0] == 1 else "FAIL",
            "confidence": round(float(max(probability[0]) * 100), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(port=5001, debug=True)
