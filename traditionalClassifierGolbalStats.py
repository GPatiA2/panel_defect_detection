import cv2
import numpy as np
import os
import argparse
import json
from copy import deepcopy
from sklearn import cluster
from functools import partial

class TraditionalClassifier():

    def __init__(self, dict_params):    
        params = cv2.SimpleBlobDetector_Params()
        params = self.set_params(params, dict_params)
        self.detector = cv2.SimpleBlobDetector_create(params)
        
    def set_params(self, params, dict_params):

        params.minThreshold = dict_params['minThreshold'] if 'minThreshold' in dict_params.keys() else params.minThreshold
        params.maxThreshold = dict_params['maxThreshold'] if 'maxThreshold' in dict_params.keys() else params.maxThreshold
        params.thresholdStep = dict_params['thresholdStep'] if 'thresholdStep' in dict_params.keys() else params.thresholdStep

        params.filterByArea = dict_params['filterByArea'] if 'filterByArea' in dict_params.keys() else params.filterByArea
        params.minArea = dict_params['minArea'] if 'minArea' in dict_params.keys() else params.minArea 
        params.maxArea = dict_params['maxArea'] if 'maxArea' in dict_params.keys() else params.maxArea

        params.filterByCircularity = dict_params['filterByCircularity'] if 'filterByCircularity' in dict_params.keys() else params.filterByCircularity
        params.minCircularity = dict_params['minCircularity'] if 'minCircularity' in dict_params.keys() else params.minCircularity
        params.maxCircularity = dict_params['maxCircularity'] if 'maxCircularity' in dict_params.keys() else params.maxCircularity

        params.filterByConvexity = dict_params['filterByConvexity'] if 'filterByConvexity' in dict_params.keys() else params.filterByConvexity
        params.minConvexity = dict_params['minConvexity'] if 'minConvexity' in dict_params.keys() else params.minConvexity
        params.maxConvexity = dict_params['maxConvexity'] if 'maxConvexity' in dict_params.keys() else params.maxConvexity

        params.filterByInertia = dict_params['filterByInertia'] if 'filterByInertia' in dict_params.keys() else params.filterByInertia
        params.minInertiaRatio = dict_params['minInertiaRatio'] if 'minInertiaRatio' in dict_params.keys() else params.minInertiaRatio
        params.maxInertiaRatio = dict_params['maxInertiaRatio'] if 'maxInertiaRatio' in dict_params.keys() else params.maxInertiaRatio

        params.minDistBetweenBlobs = dict_params['minDistBetweenBlobs'] if 'minDistBetweenBlobs' in dict_params.keys() else params.minDistBetweenBlobs

        return params

    def predict_step(self, im):

        keypoints = self.detector.detect(image = im)

        rgb_im = np.stack((im,)*3, axis=-1)

        im_with_keypoints = cv2.drawKeypoints(rgb_im, keypoints, (0,255,0))
        
        return (im_with_keypoints, keypoints)
    
    def transforms(self, mean, std):

        def preprocess(img, mean, std):
            # img  = cv2.equalizeHist(img)
            # img  = cv2.GaussianBlur(img, (5,5), 0)
            img = cv2.ximgproc.rollingGuidanceFilter(img, numOfIter= 10 )
            img  = cv2.threshold(img, mean + std, 255, cv2.THRESH_BINARY_INV)[1]
            # mean = np.mean(img)
            # std  = np.std(img)
            # img  = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
            # img  = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            cv2.namedWindow("ROLLING + TH", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROLLING + TH", 800, 800)
            cv2.imshow("ROLLING + TH", img)
            return img
        
        return partial(preprocess, mean=mean, std=std)
    

    
    
def load_dataset(im_path):

    dataset = []
    with open(os.path.join(im_path, 'tags.json'), 'r') as f:
        tags = json.load(f)

    tags_d = {}
    for it in tags:
        tags_d[it['name']] = it['label']

    for f in os.listdir(im_path):
        if f.endswith(".JPG"):
            t = 'bad' if tags_d[f] != 'healthy' else 'healthy'
            dataset.append((os.path.join(im_path, f), t))
        
    return dataset

def options():

    parser = argparse.ArgumentParser(description='Traditional Detector')
    parser.add_argument('--image_path', type=str, default='images', help='Path to dataset')
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = options()

    blob_params = {

        'minDistBetweenBlobs' : 5,

        'filterByArea' : True,
        'minArea' : 20,
        'maxArea' : 1000,

        'filterByCircularity' : False,
        'minCircularity' : 0.1,
        'maxCircularity' : 1,
        
        'filterByConvexity' : False,
        'minConvexity' : 0.1,
        'maxConvexity' : 1,
        
        'filterByInertia' : False,
        'minInertiaRatio' : 0.1,
        'maxInertiaRatio' : 1,
        
        # 'minThreshold' : 0,
        # 'maxThreshold' : 255,
        # 'thresholdStep' : 10
    }

    detector = TraditionalClassifier(blob_params)

    dataset = load_dataset(args.image_path)
    imgs = [cv2.imread(d[0]) for d in dataset]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    imgs = [cv2.resize(im, (35,50)) for im in imgs]
    imgs = np.stack(imgs, axis = -1)
    mean = np.mean(imgs[imgs > 0])
    std  = np.std(imgs[imgs > 0])

    print("Dataset loaded")
    print(len(dataset))

    for im_path, tag in dataset:

        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im2 = detector.transforms(mean, std)(deepcopy(im))

        im_with_keypoints, kp = detector.predict_step(im2)

        pred = 'healthy' if len(kp) == 0 else 'bad'
        name = 'TAG = ' + tag + " PREDICTED = " + pred
        orig = "ORIGINAL WITH KEYPOINTS"

        print("Detected keypoints: ", len(kp))
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 800)
        cv2.namedWindow(orig, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(orig, 800, 800)
        cv2.imshow(name, im_with_keypoints)
        cv2.imshow(orig, cv2.drawKeypoints(im, kp, (0,255,0)))
        cv2.waitKey(0)

 