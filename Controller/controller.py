from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import logging
import torch

from Model.predict_disease import PlantDiseaseModel

model = PlantDiseaseModel()
model.load_state_dict(torch.load('plantDisease-resnet34.pth', map_location=torch.device('cpu')))
model.eval()

def predict(image_path):
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
