import torch
import torch.nn as nn
import cv2
from traditionalClassifierLocalStats import TraditionalClassifier
import numpy as np

class TorchTraditionalClassifier(nn.Module):

    def __init__(self, inw, inh):
        super(TorchTraditionalClassifier, self).__init__()

        self.inw = inw
        self.inh = inh
        self.weights = nn.Parameter(torch.FloatTensor([0 for i in range(8)]))

    def get_initial_params(self):

        initial_params = [0 for i in range(8)]
        
        params = cv2.SimpleBlobDetector_Params()

        initial_params[0] = params.minThreshold
        initial_params[1] = params.maxThreshold
        initial_params[2] = params.thresholdStep
        initial_params[3] = 1
        initial_params[4] = 1000
        initial_params[5] = 0.1
        initial_params[6] = 1
        initial_params[7] = 0.1
        initial_params[8] = 1
        initial_params[9] = 0.1
        initial_params[10] = 1
        initial_params[11] = params.minDistBetweenBlobs

        return initial_params
    
    def transforms(self):

        def torch_transform(im):

            cv2_transform = TraditionalClassifier().transforms()
            im = cv2_transform(im)
            im = cv2.resize(im, (self.inw, self.inh))
            im = torch.from_numpy(im)
            im = im.type(torch.FloatTensor)
            return im
        
        return torch_transform

    
    def forward(self, x):

        classifier = TraditionalClassifier()
        weights = [self.weights[i].item() for i in range(len(self.weights))]
        classifier.set_params(weights)

        pred = classifier.predict_batch(x)
        pred = torch.from_numpy(np.array(pred))

        return pred
    
