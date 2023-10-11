import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class MobileNetV3Classifier(nn.Module):
    def __init__(self, opt):
        super(MobileNetV3Classifier, self).__init__()
        self.num_classes = opt.num_classes

        self.weights = None
        if opt.pretrained:
            self.weights    = torchvision.models.MobileNet_V3_Large_Weights
            self.preprocess = self.weights.transforms
            self.model = torchvision.models.mobilenet_v3_large(weights = self.weights, progress=True, num_classes=self.num_classes)
        else:
            self.model = torchvision.models.mobilenet_v3_large(weights = None, progress = True, num_classes=self.num_classes)

        self.softmax = nn.Softmax(dim=0)

    def forward(self,x):
        res = self.model(x)
        return res
    
    def transforms(self):
        return self.preprocess
    
class ViTransformerClassifier(nn.Module):

    def __init__(self, opt):
        super(ViTransformerClassifier, self).__init__()
        self.num_classes = opt.num_classes

        self.model = torchvision.models.vit_b_16(num_classes=self.num_classes)

    def forward(self,x):
        res = self.model(x)
        return res
    

class MobileNetV2Classifier(nn.Module):
    def __init__(self, opt):

        super(MobileNetV2Classifier, self).__init__()
        self.num_classes = opt.num_classes
        self.in_res = opt.in_res

        if opt.pretrained:
            self.weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.preprocess = self.weights.transforms
            self.model = torchvision.models.mobilenet_v2(weights = self.weights, dropout = 0.3)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, self.num_classes),
            )
            
            for param in self.model.features.parameters():
                param.requires_grad = False

        else:
            self.model = torchvision.models.mobilenet_v2(num_classes=self.num_classes)
            self.preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(self.in_res),
                                                              torchvision.transforms.Grayscale(num_output_channels=3)])

    def transforms(self):
        return self.preprocess

    def forward(self,x):
        res = self.model(x)
        return res
    

class ResNet18(nn.Module):

    def __init__(self, opt):

        super(ResNet18, self).__init__()
        self.num_classes = opt.num_classes

        self.model = torchvision.models.resnet18(num_classes=self.num_classes)

    def forward(self,x):
        res = self.model(x)
        return res