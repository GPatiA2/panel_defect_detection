from pannel_detector import PannelDetector
import cv2
import numpy as np
from pannel_chopper import PannelChopper
from model import PannelClassifier
from random import randint
from argparse import Namespace
import torchvision
from torch.nn import Softmax
import json
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import pylab
import torch
from sklearn.preprocessing import OneHotEncoder


class ReportDataGenerator():

    CV2_COLORS = [
        (0,255,0),
        (255,0,0),
        (0,0,255),
        (255,255,0),
        (0,255,255),
        (255,0,255)
    ]

    PLT_COLORS = [
        (0,1,0),
        (1,0,0),
        (0,0,1),
        (1,1,0),
        (0,1,1),
        (1,0,1)
    ]

    opt = Namespace(
        num_classes = 6,
        model       = 'MobileNetV2',
        pretrained  = False,
        criterion   = 'CE',
        init_method = 'none',
        in_res      = (70,100)
    )

    def __init__(self, detector_weights, classifier_weights, tag_path, res, imgs_out_dir, test = False):

        self.detector = PannelDetector(weights_file = detector_weights)

        self.test = test

        self.out_dir = imgs_out_dir

        with open(tag_path, 'r') as f:
            self.classes = json.load(f)

        self.chopper = PannelChopper(res)

        self.classifier = PannelClassifier.load_from_checkpoint(classifier_weights, opt = self.opt)
        self.classifier.freeze()

        self.rects = [patches.Patch(color=self.PLT_COLORS[i], 
                                label=self.classes[str(i)][0]) for i in range(len(self.CV2_COLORS))]
    

    def read_images(self,path):
        
        images = []
        
        k = len(os.listdir(path)) if not self.test else 10

        i = 0
        for f in os.listdir(path):
            if f.endswith('_T.JPG') and i < k:

                im = cv2.imread(os.path.join(path,f))
                images.append((f, im))
                i += 1


        return images
    
    def save_legend_image(self):

        figure = pylab.figure()
        figlegend = pylab.figure(figsize=(3,2))
        ax = figure.add_subplot(111)
        lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
        figlegend.legend(handles=self.rects, loc='center')
        figure.show()
        figlegend.show()
        path = os.path.join(self.out_dir, 'legend.png')
        figlegend.savefig(path)

        return path

    def generate_report_data(self, path_to_images):

        report_images = []
        defect_count  = {}

        images = self.read_images(path_to_images)

        os.makedirs(self.out_dir, exist_ok=True)

        for im in images:

            defect_in_image = { k[1][0] : 0  for k in self.classes.items()}

            im2 = im[1].copy()

            detections = self.detector.detect(im[1])

            crops   = self.chopper.chop(im[1], detections)

            i = 0
            for c in crops:

                if self.test:
                    pred = torch.randint(0, 6, (1,1)    )
                else:
                    img = torchvision.transforms.ToTensor()(c[1])
                    img = img[None, : , : , :]
                    img = self.classifier.transforms()(img)
                    pred = self.classifier.predict_step(img, i)
                    pred = np.argmax(Softmax(dim = 0)(pred))
                
                cv2.drawContours(im2, c[0], -1, self.CV2_COLORS[pred], 2)
                i += 1

                if self.classes[str(pred.item())][0] not in defect_count.keys():
                    defect_count[self.classes[str(pred.item())][0]] = 0

                defect_count[self.classes[str(pred.item())][0]] += 1

                defect_in_image[self.classes[str(pred.item())][0]] += 1

            pth = os.path.join(self.out_dir, im[0])

            cv2.imwrite(pth, im2)

            report_images.append((pth, defect_in_image))

        return report_images, defect_count