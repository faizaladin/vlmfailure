import torch
import torch.nn as nn
from torchvision import models, transforms

class ResNetModel(nn.Module):
    """
    A ResNet model wrapper for image classification.
    Allows specifying the ResNet version and the number of output classes.
    """
    def __init__(self, model_name='resnet50', num_classes=1000, pretrained=True):
        """
        Args:
            model_name (str): The name of the ResNet model to use (e.g., 'resnet18', 'resnet50').
            num_classes (int): The number of output classes for the final layer.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(ResNetModel, self).__init__()
        
        # Load the specified pretrained model
        self.model = models.__dict__[model_name](pretrained=pretrained)
        
        # Get the number of input features for the classifier
        num_ftrs = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Set up standard transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.model(x)

    def preprocess(self, image):
        """
        Preprocesses a single PIL image to a tensor.
        """
        return self.transform(image).unsqueeze(0)

    def predict(self, image):
        """
        Takes a PIL image, preprocesses it, and returns the model's prediction.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess(image)
            outputs = self.forward(inputs)
            if self.model.fc.out_features == 1:
                # For binary classification, return the sigmoid probability
                return torch.sigmoid(outputs).item()
            else:
                # For multi-class, return the class with the highest score
                _, predicted = torch.max(outputs, 1)
                return predicted.item()

