import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.nn import functional as F
from PIL import Image
import sys

model_load_path = sys.argv[1]

# Load the saved model
loaded_model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
num_features = loaded_model.fc.in_features
loaded_model.fc = torch.nn.Linear(num_features, 2)
loaded_model.load_state_dict(torch.load(model_load_path))
loaded_model.eval()

# Move the model to the GPU (Apple Metal)
device = torch.device("mps")
loaded_model = loaded_model.to(device)

# Load and preprocess the image
def preprocess_image(image_path):
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_tensor = img_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Specify the folder containing the images
image_folder = "./dataset/mix"

# Iterate through the folder and process each image
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Predict using the loaded model
    with torch.no_grad():
        output = loaded_model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        _, prediction = torch.max(output, 1)
        result = "hotdog" if prediction.item() == 0 else "not hotdog"  # Assuming hotdog is label 0 and not hotdog is label 1
        probability = probabilities[0][prediction].item()

    print(f'Image: {image_name} | Result: {result} | Probability: {probability}')
