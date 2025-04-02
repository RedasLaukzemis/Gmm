from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim):
        super(ConvolutionalModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        reduced_size = input_size // 4
        self.first_dim = reduced_size * reduced_size * 4

        self.fc1 = nn.Linear(self.first_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3, training=self.training)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224  # Same size as used in training
hidden_dim = 32  # Match training parameters
output_dim = 3   # Number of classes

model = ConvolutionalModel(input_size, hidden_dim, output_dim)
# model.load_state_dict(torch.load('c://users/user/desktop/2_gmm_lab/modelWeights/model_weights.pth', map_location=device))
model_path = os.path.join(os.getcwd(), "modelWeights", "model_weights.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

class_labels = ["Car", "Jellyfish", "Parrot"]

# Initialize Flask API
app = Flask(__name__)

# Define transformation (must match the training transformations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure RGB mode
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            output = model(img)
            prediction_index = torch.argmax(output, dim=1).item()
        
        prediction_label = class_labels[prediction_index]
        return jsonify({'prediction': prediction_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)