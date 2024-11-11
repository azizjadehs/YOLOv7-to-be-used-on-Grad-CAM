import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


from yolo_model import ModifiedYolov7  # Replace with actual class name
from classifier_output import ClassifierOutput

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision import transforms
import yaml


sys.path.append('')  # Add your system path
from models.yolo import Model, IDetect  # Ensure this is the correct import path for your Model class


# Load the model configuration from YAML
yaml_path = '.yaml'
try:
    with open(yaml_path, 'r') as f:
        model_config = yaml.safe_load(f)
    print("Model configuration loaded successfully")

except FileNotFoundError:
    print(f"Error: The file {yaml_path} was not found.")
except yaml.YAMLError as e:
    print(f"Error loading YAML file: {e}")

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(model_config).to(device)


# Load the trained weights from the .pt file
weights_path = '.pt'

try:
    checkpoint = torch.load(weights_path, map_location=device)
    print("Weights loaded successfully")
except FileNotFoundError:
    print(f"Error: The weights file {weights_path} was not found.")
except RuntimeError as e:
    print(f"Error loading weights: {e}")

if 'model' in checkpoint:
    state_dict = checkpoint['model'].state_dict()
    print("Model is in checkpoint and state_dict loaded")
else:
    state_dict = checkpoint
    print("Model is not in checkpoint, using checkpoint directly as state_dict")
model.load_state_dict(state_dict)
print("Model state_dict loaded successfully")

model.eval()

# Choose the selected layer (you already found `104.rbr_1x1.0`)
selected_layer = None
for name, layer in model.model.named_modules():
    if name == "":  # Ensure this matches exactly with the selected layer's name
        print(f"Selected layer for Grad-CAM: {name}")
        selected_layer = layer
        break

if selected_layer is None:
    raise ValueError("The selected layer was not found in the model.")

# Initialize Grad-CAM with the selected layer
cam = GradCAM(model=model, target_layers=[selected_layer])
#cam = GradCAMPlusPlus(model=model, target_layers=[selected_layer])


# Specify the input folder containing images
input_folder = ''  # Change this to your input folder

# Specify the output folder where the Grad-CAM images will be saved-
output_folder = ''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Load and preprocess your image
image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
# Process each image in the folder
for i, image_path in enumerate(image_paths):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize to match your model's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #Standard for Imagenet
    ])
    input_image = preprocess(image).unsqueeze(0).to(device)

    # Perform a forward pass to get the model output
    outputs = model(input_image)


# Define the target class index
    target_class_idx = 0  # Replace with the desired class index from your dataset
    targets = [ClassifierOutputTarget(target_class_idx)]

# Generate the Grad-CAM for the first target
    grayscale_cam = cam(input_tensor=input_image, targets=targets)[0] 
    print(grayscale_cam)
    # Convert image for visualization (undo normalization for display)
    rgb_img = np.array(image)
    rgb_img = np.float32(rgb_img) / 255  # Convert to float32 for visualization

    # Resize the Grad-CAM mask to match the input image size
    grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))

    # Convert the grayscale CAM to 3D (RGB)
    grayscale_cam_resized = np.stack([grayscale_cam_resized] * 3, axis=-1)

    # Visualize the result
    visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True, image_weight=0.7)

    # Ensure the visualization is scaled between 0 and 255
    if visualization.max() > 1.0:
        visualization = np.uint8(visualization)
    else:
        visualization = np.uint8(visualization * 255)

    # Save the visualization image with a unique name
    base_name = Path(image_path).stem  # Get the image file name without extension
    save_path = os.path.join(output_folder, f'{base_name}_grad_cam_{i}.png')
    Image.fromarray(visualization).save(save_path)
    print(f"Grad-CAM visualization saved as '{save_path}'")
