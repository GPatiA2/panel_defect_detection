from pannel_detector import PannelDetector
import cv2
import numpy as np
from pannel_chopper import PannelChopper
from models.model import PannelClassifier
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
from sklearn.preprocessing import LabelBinarizer

class ReportDataGenerator():

    def __init__(self, detector, classifier, chopper, tag_path, imgs_out_dir, test = False):

        self.test = test

        self.out_dir = imgs_out_dir

        with open(tag_path, 'r') as f:
            self.classes = json.load(f)

        # self.plt_colors = self.labels_as_colors(self.classes)
        # self.cv2_colors = [[it[i] * 255 for i in range(len(it))] for it in self.plt_colors]
        self.plt_colors = [(0,1,0), (1,0,0)]
        self.cv2_colors = [(0,255,0), (0,0,255)]
        self.rects = [patches.Patch(color=self.plt_colors[i], 
                                label=self.classes[i]) for i in range(len(self.cv2_colors))]

        self.detector   = detector
        self.classifier = classifier
        self.chopper    = chopper

    def labels_as_colors(self, lb):

        lb_c = [bin(idx + 1) for idx, val in enumerate(lb)]
        lb_c = [lb_c[idx][2:] for idx, val in enumerate(lb_c)]
        
        max_len = max(lb_c, key = lambda x : len(x))
        lb_color_corrected = []
        for it in lb_c:
            while len(it) < len(max_len):
                it = '0' + it
            lb_color_corrected.append(it)

        lb_c = [[int(it[i]) for i in range(len(it))] for it in lb_color_corrected]
        return lb_c

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

    def generate_report_data(self, path_to_images, show_detections = False, show_crops = False):

        report_images = []
        defect_count  = {}

        images = self.read_images(path_to_images)

        os.makedirs(self.out_dir, exist_ok=True)

        for im in images:

            defect_in_image = { k : 0  for k in self.classes}

            im2 = im[1].copy()

            detections = self.detector.detect(im[1])

            conts = [np.array(d['segmentation'], dtype=np.int32) for d in detections]
    
            if show_detections:
                print(len(conts))
                print(conts[0].dtype)
                print(conts[0])

                im3 = im2.copy()
                im3 = cv2.drawContours(im3, conts, -1, (0,255,0), 2)
                cv2.imshow("im3", im3)
                cv2.waitKey(0)

            crops   = self.chopper.efficient_chop(im[1], detections)

            if show_crops:
                for c in crops:
                    cv2.imshow("crop", c[1])
                    cv2.waitKey(0)

            i = 0
            for c in crops:

                if self.test:
                    pred = torch.randint(0, 6, (1,1))
                else:
                    pred = self.classifier.predict_step(np.uint8(c[1]))
                    pred = torch.from_numpy(np.array(pred))
                    # img = torchvision.transforms.ToTensor()(c[1])
                    # img = img[None, : , : , :]
                    # img = self.classifier.transforms()(img)
                    # pred = self.classifier.predict_step(img, i)
                    # pred = np.argmax(Softmax(dim = 0)(pred))
                
                im2 = cv2.drawContours(im2, c[0], -1, self.cv2_colors[pred], 2)
                i += 1

                if self.classes[pred.item()] not in defect_count.keys():
                    defect_count[self.classes[pred.item()]] = 0

                defect_count[self.classes[pred.item()]] += 1

                defect_in_image[self.classes[pred.item()]] += 1

            pth = os.path.join(self.out_dir, im[0])

            cv2.imwrite(pth, im2)

            report_images.append((pth, defect_in_image))

        return report_images, defect_count