from flask import Flask, request, render_template, url_for
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        
        # Create a numpy array for prediction
        data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                          free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        
        # Predict the quality
        prediction = model.predict(data)
        
        # Generate the plot
        plt.figure(figsize=(10, 6))
        features = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides', 
                    'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
        values = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
        
        plt.barh(features, values, color='skyblue')
        plt.xlabel('Value')
        plt.title('Wine Quality Prediction Inputs')
        plt.tight_layout()
        
        # Save the plot as a static file
        plot_path = 'static/prediction_plot.png'
        plt.savefig(plot_path)
        plt.close()

        # Return the result
        return render_template('index.html', prediction_text=f'Predicted Wine Quality: {prediction[0]:.2f}')
        
if __name__ == '__main__':
    app.run(debug=True)
