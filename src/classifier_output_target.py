class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # Check if the output is a list and extract the first element
        if isinstance(model_output, list):
            print(f"Model output is a list with length {len(model_output)}")
            model_output = model_output[0]  # Adjust this based on what part of the output you need

        print(f"ClassifierOutputTarget called with model_output of type: {type(model_output)} and shape: {model_output.shape}")

        if len(model_output.shape) == 5:
         # Handle 5D tensor of shape [batch_size, anchors, grid_y, grid_x, outputs]
         # Assuming the last dimension contains the class scores after the bounding box coordinates
            class_scores = model_output[:, :, :, :, self.category + 5]
            print(f"Extracted class scores for category {self.category} from 5D tensor")
            return class_scores.sum()


        if len(model_output.shape) == 3:
            # Assuming the last 4 values in the third dimension are class scores
            class_scores = model_output[:, :, self.category + 5]
            return class_scores.sum()

        elif len(model_output.shape) == 2:
            # Handle the case with shape [25200, 9]
            class_scores = model_output[:, self.category + 5]
            return class_scores.sum()

        else:
            raise ValueError("Unexpected model output shape: " + str(model_output.shape))
