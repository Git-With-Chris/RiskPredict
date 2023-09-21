from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# Specify the path to the directory you want to set as your local working directory
current_working_directory = 'C:/Users/Xiaofeng/RMIT/Case Studies in Data Science/WIL Project/WIL_Project'

# Change the current working directory to the specified path
os.chdir(current_working_directory)

app = Flask(__name__)

# Load Customer Data and Risk Prediction Model
data = pd.read_csv('./Flask_App/Customer_DB.csv')                                       # change file path to better suit the current working dir
Risk_Prediction_Model = load_model('./Flask_App/model/Risk_Prediction_Model.h5')        # change file path to better suit the current working dir

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

        # Predicted probability for the test observation
        predicted_probability = predicted_risk_value

        # Define the sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Generate x values (a range of values from -7 to 7)
        x = np.linspace(-5, 5, 200)

        # Calculate the corresponding y values using the sigmoid function
        y = sigmoid(x)

        # Create a new figure for each prediction
        plt.figure()

        # Plot the sigmoid curve above 0.5 on the y-axis in red and below 0.5 in green
        plt.plot(x[y >= 0.5], y[y >= 0.5], color='red', label='Risk')
        plt.plot(x[y <= 0.5], y[y <= 0.5], color='green', label='Non-Risk')

        # Calculate the x-coordinate on the sigmoid curve for the given y-coordinate (predicted_probability)
        x_predicted = np.interp(predicted_probability, y, x)

        # Highlight the predicted probability point
        plt.scatter(x=[x_predicted], y=predicted_probability, color='blue', label='User')

        # Label the axes and add a legend
        plt.xlabel('Input')
        plt.ylabel('Sigmoid Output')
        plt.legend()

        # Show the plot
        plt.title(f'Predicted Probability ({predicted_probability:.2f})')
        plt.grid(True)

        save_path = './Flask_App/static/sigmoid_plot.png'   # change file path to better suit the current working dir
        plt.savefig(save_path)

        return render_template(
            "index.html",
            customer_id=customer_id,
            predicted_risk_value=predicted_risk_percent,
            repayment_probability=repayment_probability,
            prediction=prediction
        )   # this code passing important parameters to the html template

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
