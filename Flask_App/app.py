from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)

# Load Customer Data and Risk Prediction Model
data = pd.read_csv("Customer_DB.csv")
Risk_Prediction_Model = load_model('model/Risk_Prediction_Model.h5')

def predict_risk(input_data):
    # Extract the features (excluding the 'ID' column)
    input_data = input_data.drop(columns=['ID'])
    
    # Make predictions using the loaded model
    predictions = Risk_Prediction_Model.predict(input_data)
    
    # Assuming the model returns a single prediction, you can access it like this
    predicted_risk = round(float(predictions[0]), 4)

    # Define the threshold
    threshold = 0.5

    # Make a binary prediction based on the threshold
    if predicted_risk >= threshold:
        prediction = "Risk"
    else:
        prediction = "Non-Risk"
    
    return predicted_risk, prediction


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        customer_id = int(request.form.get("customer_id"))
        customer_data = data[data['ID'] == customer_id]
        
        if customer_data.empty:
            return render_template("index.html", error="ID not found")
        
        predicted_risk_value, prediction = predict_risk(customer_data)
        predicted_risk_percent = round(predicted_risk_value * 100, 2)
        repayment_probability = round((1 - predicted_risk_value) * 100, 4)

        return render_template(
            "index.html",
            predicted_risk_value=predicted_risk_percent,
            repayment_probability=repayment_probability,
            prediction=prediction
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
