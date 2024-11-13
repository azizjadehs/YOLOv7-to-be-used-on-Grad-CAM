import argparse
import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import yaml

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from models.yolo import Model, IDetect  # or if modifications directly on models.yolo.py change into from models.yolo import Model, IDetect
from src.classifier_output_target import ClassifierOutputTarget

# Argument parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for YOLOv7")
    parser.add_argument('--yaml-path', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--weights-path', type=str, required=True, help="Path to the trained model weights (.pt)")
    parser.add_argument('--input-folder', type=str, required=True, help="Path to the input folder containing images")
    parser.add_argument('--output-folder', type=str, required=True, help="Path to the output folder for Grad-CAM images")
    parser.add_argument('--target-class-idx', type=int, default=0, help="Target class index for Grad-CAM visualization")
    parser.add_argument('--resize-dim', type=int, nargs=2, default=[416, 416], help="Resize dimensions for the input image (width, height)")
    parser.add_argument('--selected-layer', type=str, default=None, help="Name of selected layer for Grad-CAM")

    
    return parser.parse_args()

def main():
    args = parse_args()
    
    yaml_path = args.yaml_path
    # Load the model configuration from YAML
    try:
        with open(args.yaml_path, 'r') as f:
            model_config = yaml.safe_load(f)
        print("Model configuration loaded successfully")
    except FileNotFoundError:
        print(f"Error: The file {args.yaml_path} was not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {e}")
        return

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(model_config).to(device)

    # Load the trained weights
    weights_path = args.weights_path
    try:
        checkpoint = torch.load(args.weights_path, map_location=device)
        print("Weights loaded successfully")
    except FileNotFoundError:
        print(f"Error: The weights file {args.weights_path} was not found.")
        return
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    if 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
        print("Model is in checkpoint and state_dict loaded")
    else:
        state_dict = checkpoint
        print("Model is not in checkpoint, using checkpoint directly as state_dict")
    model.load_state_dict(state_dict)
    print("Model state_dict loaded successfully")

    model.eval()

    # Choose the selected layer    
    for name, layer in model.model.named_modules():
        if name == args.selected_layer:  # Ensure this matches the selected layer's name
            print(f"Selected layer for Grad-CAM: {name}")
            selected_layer = layer
            break

    if selected_layer is None:
        raise ValueError("The selected layer was not found in the model.")

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[selected_layer])

    input_folder = args.input_folder  # Change this to your input folder

    # Specify the output folder where the Grad-CAM images will be saved-
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
       os.makedirs(output_folder)


    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Get image paths from the input folder
    image_paths = [os.path.join(args.input_folder, fname) for fname in os.listdir(args.input_folder)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Process each image in the folder
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(args.resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = preprocess(image).unsqueeze(0).to(device)

        # Perform a forward pass
        outputs = model(input_image)

        # Define the target class index
        targets = [ClassifierOutputTarget(args.target_class_idx)]

        # Generate the Grad-CAM
        grayscale_cam = cam(input_tensor=input_image, targets=targets)[0]

        # Prepare for visualization
        rgb_img = np.array(image) / 255.0  # Normalize to [0, 1] range
        grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
        visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True, image_weight=0.7)

        # Ensure the visualization is in [0, 255] range
        visualization = (visualization * 255).astype(np.uint8) if visualization.max() <= 1.0 else visualization

        # Save the output image
        base_name = Path(image_path).stem
        save_path = os.path.join(args.output_folder, f'{base_name}_grad_cam_{i}.png')
        Image.fromarray(visualization).save(save_path)
        print(f"Grad-CAM visualization saved as '{save_path}'")

if __name__ == "__main__":
    main()
