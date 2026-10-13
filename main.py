from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

coefficients = np.array([4.16486355, 47.69091621, 59.15398742, -5.27539399, -3.14124207])
intercept = 211.70683965

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    reqData = request.get_json()
    data = reqData['features']
    
    input_array = np.array(data, dtype=float)
    
    hasil = np.dot(input_array, coefficients) + intercept
    
    return jsonify({'hasil_prediksi': round(float(hasil), 4)})

if __name__ == '__main__':
    app.run(debug=True)