from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

app = Flask(__name__)

# Load Customer Data and Risk Prediction Model
data = pd.read_csv("Flask_App/Customer_DB.csv")
Risk_Prediction_Model = load_model('Flask_App/model/Risk_Prediction_Model.h5')
graphing_df = pd.read_csv('Flask_App/Graph_CSV.csv')

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
        prediction = "High-Risk"
    else:
        prediction = "Low-Risk"
    
    return predicted_risk, prediction

# Get the customer prediction information
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

"""
#######
This code was causing the website to crash for me
######
@app.route("/get_plot", methods = ['GET', 'POST'])
def graph_generator():
    if request.method == "POST":
        cust_id = request.form['cust_id'] # input has not been fixed, I was first trying to get the image to show on the index file
        sns.boxplot(data = graphing_df, y= "ExternalRiskEstimate", x = "Category")
        plt.title('Test_graph')
        plt.savefig('static/my_plot.png') # save in static for referencing
        return render_template("index.html", plot_url = 'static/my_plot.png') # URL to reference for /get_plot
    else:
        return render_template('index.html')

        
        """


"""
#####
Rough Start for a function that can return graph via ID

import seaborn as sns
def graph_generator(customer_id, graph_type, variable):
    id = customer_id
    new_df = df
    new_df['Category'][new_df['ID'] == id] = 1.5
    if graph_type == "BoxPlot":
        ax1 = plt.subplot()
        ax1 = sns.boxplot(data = new_df, x = 'Category', y = variable)
        ax1.set_xticklabels(['Low Risk', 'High Risk', 'You'])
        plt.xlabel('Category')
        plt.ylabel('Normalised External Risk Estimate')
        plt.title('Model Comparison of External Risk Estimate')
        plt.show()
        plt.clf()
    
        
    elif graph_type == "Distribution Plot":
        x_val = new_df[variable][new_df['ID'] == id]
        ax1 = plt.subplot()
        ax1 = sns.kdeplot(data=new_df, x=variable, hue='Category', fill=True)
    
        plt.vlines(x=x_val, ymin=0, ymax=2, label="You", colors='red', linestyle='dashed')  # Adjust linestyle and color as needed
        plt.xlabel('Normalized External Risk Estimate')
        plt.ylabel('Distribution')
        plt.title('Model Comparison of External Risk Estimate')
    
        # Set the legend labels using plt.legend
        plt.legend(title='Risk Category', labels=["Low Risk", "High Risk", "You"])
    
        plt.show()
        plt.clf()

        # Might need to savefig in /static for Flask to be able to send it to webpage

"""

"""
#Code Needed to Create Sigmoid Function. Does not work currently in Flask and needs to be adjusted.

# Predicted probability for the test observation
predicted_probability = predicted_probabilities

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values (a range of values from -7 to 7)
x = np.linspace(-5, 5, 200)

# Calculate the corresponding y values using the sigmoid function
y = sigmoid(x)

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
plt.show()
"""



if __name__ == "__main__":
    app.run('127.0.0.1', 5000, debug=True)
