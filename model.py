from torchvision import  models
import torch
import torch.nn as nn

def mymodel(model_name = "resnet50"):
    n_classes = 1103  
    model = models.resnet50()
    model.load_state_dict(torch.load("../input/model/"+"resnet50"+ ".pth" ))
    n_filters = model.fc.in_features
    model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(n_filters, n_classes),
                nn.Sigmoid()
    )
    
    return model
