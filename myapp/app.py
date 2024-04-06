import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the model
class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out

# Load the model weights
model = Plant_Disease_Model()
model.load_state_dict(torch.load('plantDisease-resnet34.pth', map_location=torch.device('cpu')))
model.eval()

# Load disease information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor()
])

# Define prediction function
def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        return index
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

# Define routes
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

        pred = prediction(file_path)

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
    app.run(debug=True, port=8000)
