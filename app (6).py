from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import gzip

# Load the trained pipeline from the pickle file


with gzip.open('pipeline.pkl.gz', 'rb') as file:
    pipeline = pickle.load(file)

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = {
        'PM2.5' : float(request.form['PM2.5']),
        'City' : request.form['City'],
        'PM10' : float(request.form['PM10']),
        'NO' : float(request.form['NO']),
        'NO2' : float(request.form['NO2']),
        'NOx' : float(request.form['NOx']),
        'CO' : float(request.form['CO']),
        'SO2' : float(request.form['SO2']),
        'Toluene' : float(request.form['Toluene'])
    }
    

    # Convert the data to a DataFrame
    df = pd.DataFrame([data])


    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(df)

    # Return the result to the template
    result = f'Predicted AQI: {predictions[0]:.2f} '
    return render_template('result.html', predictions=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)