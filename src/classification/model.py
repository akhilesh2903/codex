import torch
import torch.nn as nn
import torchvision.models as models

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet_b0', pretrained=True):
        super(DRClassifier, self).__init__()
        
        if model_name.startswith('efficientnet'):
            self.model = getattr(models, model_name)(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name.startswith('resnet'):
            self.model = getattr(models, model_name)(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif model_name.startswith('densenet'):
            self.model = getattr(models, model_name)(pretrained=pretrained)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
            
    def forward(self, x):
        return self.model(x)

def get_model(num_classes, model_name):
    return DRClassifier(num_classes=num_classes, model_name=model_name)
