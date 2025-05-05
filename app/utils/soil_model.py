import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

# Check if GPU is available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Soil type classes
soil_classes = ["Black Soil", "Cinder Soil", "Laterite Soil", "Peat Soil", "Yellow Soil"]

# Load MobileNetV2 and update classifier
from torchvision.models import MobileNet_V2_Weights

# Initialize the model
model = models.mobilenet_v2(weights=None) 
model.classifier[1] = torch.nn.Linear(model.last_channel, len(soil_classes))

# Load the trained weights
model_path = 'app/models/best_model.pth'

# Check if the model file exists
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {model_path}. Please ensure the model exists.")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict function
def predict_soil(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return f"Error processing image: {e}"

    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move tensor to the correct device
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return soil_classes[predicted.item()]

