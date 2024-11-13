# YOLOv7-to-be-used-on-Grad-CAM
This repos provides modifications on YOLOv7 model and add classes to make it useable with pytorch Grad-CAM

This repository extends the [YOLOv7 model](https://github.com/WongKinYiu/yolov7), originally created by WongKinYiu and contributors, by integrating Grad-CAM for enhanced explainability of object detection results. YOLOv7 is known for its high performance in real-time object detection tasks and is licensed under the GNU General Public License (GPL).


# About This Project
The original YOLOv7 is an outstanding real-time object detection model. However, when integrating Grad-CAM for explainability, the base version of YOLOv7 generated various error messages and compatibility issues. This project was created to address those issues and make the model compatible with Grad-CAM, enabling users to visualize and understand which parts of an image contribute most to the model's detection results.

## Errors when using Grad-CAM with YOLOv7:
-Mismatch error when loading configuration of the model.
->Solution: change the number of classes "nc" in the .yaml data into the number of trained classes.

-RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation.
->Solution: Removing all inplace operations from the IDetect class in the .py data of the model (Modifications done on YOLOv7 here already).

-AttributeError: 'tuple' object has no attribute 'shape' from Grad-CAM library.
->Solution: Extract the first tensor [0] from the Output tuple of YOLOv7 (IDetect).

-RuntimeError: grad can be implicitly created only for scalar outputs.
->Solution: making the class: _ClassifierOutputTarget_. This class can handel different types of model's outputs and extract the class score.


## Key Modifications:

-Adjustments and additions to the original YOLOv7 code to ensure seamless integration with Grad-CAM.
-Creation of new classes and methods to enable Grad-CAM functionality without compromising YOLOv7's detection performance.

Why This Repository? This repository provides a ready-to-use, modified version of YOLOv7 that supports Grad-CAM, making it easier for researchers and developers to implement explainable AI techniques in object detection tasks without running into compatibility issues.

## Compatibility and Tested Environment

This project was tested with the following versions:

- **PyTorch**: 2.5.0+cu121
- **Torchvision**: 0.20.0+cu121
- **CUDA**: 12.1

While these versions were tested and verified, the code may work with other compatible versions of PyTorch and Torchvision. If you encounter any compatibility issues, consider aligning with these versions.

## Acknowledgments
- Thanks to the creators of the original [YOLOv7](https://github.com/WongKinYiu/yolov7) for providing a robust base model that this project builds upon.
- And thanks to the creators of Grad-CAM code [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).

## Running the Script with Parameters
To run the script, use the following command:
```bash
python src/main_script.py --yaml-path path/to/config.yaml --weights-path path/to/weights.pt --input-folder path/to/input/images --output-folder path/to/output/images --target-class-idx 1 --resize-dim 416 416 --selected-layer 104.rbr_dense.0
```
Or simply copy paste the main_code and run it in your IDE.

## Uploaded a colab notebook [_Using_Grad_CAM_on_YOLOv7_Tutorial_](https://github.com/azizjadehs/YOLOv7-to-be-used-on-Grad-CAM/blob/main/Using_Grad_CAM_on_YOLOv7_Tutorial.ipynb) to help run the main code and the script.
