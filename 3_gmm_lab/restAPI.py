import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, render_template
import os

app = Flask(__name__)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)  # Added deeper layer
        self.pool = nn.MaxPool2d(2)

        # Decoder (upsampling)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)  # Skip connection added
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)   # Skip connection added
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # 64 channels
        e2 = self.enc2(self.pool(e1))  # 128 channels
        e3 = self.enc3(self.pool(e2))  # 256 channels (new)

        # Decoder with skip connections
        d3 = self.up3(e3)        # 128 channels
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection
        d3 = self.dec3(d3)       # 128 channels

        d2 = self.up2(d3)        # 64 channels
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection
        d2 = self.dec2(d2)       # 64 channels

        return self.final(d2)

def load_model(n_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=n_classes)
    model_path = os.path.join(os.getcwd(), "modelWeights", "unet_multiclass.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    # Resize to expected input size (modify as needed)
    img = img.resize((256, 256))
    img = np.array(img)
    # Normalize and convert to tensor
    img = img / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
    return img


def mask_to_image(mask):
    # Assuming mask is [1, H, W] with class indices
    mask = mask.squeeze().cpu().numpy()
    # Create RGB image (modify colors as needed)
    color_map = {
        0: [0, 0, 0],       # Background (black)
        1: [255, 0, 0],     # Class 1 (red)
        2: [0, 255, 0],     # Class 2 (green)
        3: [0, 0, 255]      # Class 3 (blue)
    }
    
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            rgb[i, j] = color_map.get(mask[i, j], [0, 0, 0])
    
    return Image.fromarray(rgb)

model = load_model()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Empty file uploaded'}), 400
    
    try:
        # Read original image for overlay
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Preprocess for model
        img_tensor = preprocess_image(img_bytes)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            mask = torch.argmax(output, dim=1)

        # Generate mask image
        mask_img = mask_to_image(mask)

        # Resize mask to original image size
        mask_img = mask_img.resize(original_img.size, resample=Image.NEAREST)

        # Blend with original image
        mask_rgba = mask_img.convert("RGBA")
        overlay = Image.blend(original_img.convert("RGBA"), mask_rgba, alpha=0.5)

        # Save to bytes
        img_byte_arr = io.BytesIO()
        overlay.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)