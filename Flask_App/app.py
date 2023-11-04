import flask
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model
import seaborn as sns



app = Flask(__name__)

# Load Customer Data and Risk Prediction Model
data = pd.read_csv('Customer_DB.csv')                        
Risk_Prediction_Model = load_model('model/Risk_Prediction_Model.h5')     
df = pd.read_csv('Data_with_prediction_category.csv')

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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        customer_id = int(request.form.get("customer_id"))
        parameter_range = int(request.form.get("parameter_range")) 
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

        # Define the columns you want to plot according to the parameter_range input
        # make sure the parameter_range input not exceeding the ceilings
        if parameter_range < 1:
            parameter_range = 1
        elif parameter_range > 35:
            parameter_range = 35 
        
        # define the columns_to_plot as a list of parameters' names 
        columns_to_plot = customer_data.columns[1: parameter_range+1]

        # create financial_paramters list
        financial_parameters = []
        for i, col in enumerate(columns_to_plot):
            financial_parameters.append((i+1, col))

        # Use Matplotlib's GridSpec to create a grid of subplots with custom sizes:
        fig = plt.figure(figsize=(16, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        sigmoid_plot = plt.subplot(gs[0])
        box_plot = plt.subplot(gs[1])

        # First subplot: sigmoid curve
        # Plot the sigmoid curve above 0.5 on the y-axis in red and below 0.5 in green
        sigmoid_plot.plot(x[y >= 0.5], y[y >= 0.5], color='red', label='Risk')
        sigmoid_plot.plot(x[y <= 0.5], y[y <= 0.5], color='green', label='Non-Risk')

        # Calculate the x-coordinate on the sigmoid curve for the given y-coordinate (predicted_probability)
        x_predicted = np.interp(predicted_probability, y, x)

        # Highlight the predicted probability point
        sigmoid_plot.scatter(x=[x_predicted], y=predicted_probability, color='blue', label='Customer')   # sychronizing all user tags as 'Customer'

        # Label the axes and add a legend
        sigmoid_plot.set_xlabel('Input', fontsize=14)
        sigmoid_plot.set_ylabel('Sigmoid Output', fontsize=14)
        sigmoid_plot.legend()

        # Show the plot
        sigmoid_plot.set_title(f'Predicted Probability ({predicted_probability:.2f})')
        sigmoid_plot.grid(True) 

        # Second subplot: multi box-plot

        # Customize box and median line appearance
        box_colors = {'color': '#4B0082'}  # Indigo in hex
        median_colors = {'color': '#00FF00'}  # Bright Green in hex
        whisker_colors = {'color': '#4B0082'}  # Indigo in hex
        cap_colors = {'color': '#4B0082'}  # Indigo in hex

        # draw box plot according to data input
        boxplots = box_plot.boxplot([data[col] for col in columns_to_plot], labels=range(1, len(columns_to_plot)+1), showfliers=False, patch_artist=True, 
                                    boxprops=box_colors, medianprops=median_colors, whiskerprops=whisker_colors, capprops=cap_colors)

        # Changing the fill color of the boxes
        for box in boxplots['boxes']:
            box.set_facecolor('#D8BFD8')  # Light purple

        # Changing the color and thickness of the median lines
        for median in boxplots['medians']:
            median.set(linewidth=4)  #  linewidth=2


        # Highlight customer_data on the box-plot
        for i, col in enumerate(columns_to_plot):
            box_plot.scatter(x=i + 1, y=customer_data[col].values, color='blue', label='Customer' if i == 0 else "", s=200, zorder = 10)

        box_plot.set_title(f'Position of Customer with ID {customer_id} on the Distrubution Board of Financial Parameters')
        box_plot.set_xlabel('Financial Parameters', fontsize=14)
        box_plot.set_ylabel('Values', fontsize=14)
        box_plot.tick_params(axis='y', labelsize=8)
        box_plot.legend(loc='upper right')

        save_path = 'static/sigmoid_plot.png'   # change file path to better suit the current working dir
        plt.savefig(save_path)
        plt.clf()

        return render_template(
            "index.html",
            customer_id=customer_id,
            predicted_risk_value=predicted_risk_percent,
            repayment_probability=repayment_probability,
            prediction=prediction,
            financial_parameters = financial_parameters
        )   # this code passing important parameters to the html template

    return render_template("index.html")

@app.route("/info", methods=["GET", "POST"])
def two_chart():
    if request.method == "POST":
        customer_id = int(request.form.get("cust_id"))
        customer_data = data[data['ID'] == customer_id]

        if customer_data.empty:
            return render_template("index.html", error="ID not found")

        #external_risk_estimate
        comp = df[df['ID'] == customer_id]
        labels = ['High Risk', 'Low Risk', 'You']
        sns.kdeplot(df, x = 'ExternalRiskEstimate', hue = 'Category', fill = True)
        plt.vlines(x = comp['ExternalRiskEstimate'], ymin=0, ymax=2, color = 'red', label = 'You', linestyles=['dashed'])
        plt.legend(labels = labels)
        plt.xlabel('Normalised External Risk Estimate Score')
        plt.title('Comparison of External Risk Estimate')
        plt.savefig('static/external_risk_estimate.png')
        plt.show()
        plt.clf()
        labels = ['High Risk', 'Low Risk', 'You']
        sns.kdeplot(df, x = 'NetFractionRevolvingBurden', hue = 'Category', fill = True)
        plt.vlines(x = comp['NetFractionRevolvingBurden'], ymin=0, ymax=14, color = 'red', label = 'You', linestyles=['dashed'])
        plt.legend(labels = labels)
        plt.xlabel('Normalised Net Fraction Revolving Burdden')
        plt.title('Comparison of Net Fraction Revolving Burden')
        plt.savefig('static/percent_delinquency.png')
        plt.clf()
        ax = plt.subplot()
        sns.boxplot(df, x = 'Category', y = 'NumTotalTrades')
        plt.hlines(y = comp['NumTotalTrades'], xmin=-1, xmax=2, color = 'red', label = 'You', linestyles=['dashed'])
        plt.legend()
        ax.set_xticklabels(labels = ["Low Risk", "High Risk"])
        plt.ylabel("Normalised Number of Trades")
        plt.title('Comparison of Number of Total Trades Between Risk Groups')
        plt.savefig('static/num_trades.png')
        plt.clf()
        plt.legend()

        return render_template(
            "info.html"
        )   # this code passing important parameters to the html template

    return render_template("info.html")

if __name__ == "__main__":
    app.run(debug=True)
