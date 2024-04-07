import os
from flask import Flask, redirect, render_template, request, jsonify
from Controller.controller import predict
import torch
import logging
import pandas as pd
import gunicorn


# Load disease information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def detect():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image']
        filename = image.filename

        if filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format, only PNG, JPG, JPEG allowed'}), 400

        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        pred = predict(file_path)

        if pred is None:
            return jsonify({'error': 'Failed to make prediction'}), 500

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        return render_template('predict.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred)

        
    except Exception as e:
        logging.error(f"Submission error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
