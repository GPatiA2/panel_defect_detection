import cv2
import numpy as np
import os
import argparse
import json
from copy import deepcopy
from sklearn import cluster

class ClusteringClassifier():

    def __init__(self):
        self.classifier = cluster.KMeans(n_clusters=2, init = 'k-means++')
    
    def train_step(self, images):
        self.classifier.fit(images)

    def predict_step(self, images):
        return self.classifier.predict(images)
    
    def transforms(self):

        def preprocess(img):
            mean = np.mean(img)
            std  = np.std(img)
            img  = cv2.GaussianBlur(img, (5,5), 0)
            img  = cv2.threshold(img, mean + std, 255, cv2.THRESH_BINARY_INV)[1]
            # img  = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
            # img  = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            return img
        
        return preprocess
    
def load_dataset(im_path, transforms):

    dataset = []
    with open(os.path.join(im_path, 'tags.json'), 'r') as f:
        tags = json.load(f)

    tags_d = {}
    for it in tags:
        tags_d[it['name']] = it['label']

    for f in os.listdir(im_path):
        if f.endswith(".JPG"):
            im = cv2.imread(os.path.join(im_path, f))
            im = cv2.resize(im, (35,50))
            im_orig = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = transforms(im_orig)
            im2 = np.ravel(im)
            dataset.append((im, im2, im_orig))

    return dataset

def options():

    parser = argparse.ArgumentParser(description='Traditional Detector')
    parser.add_argument('--train_image_path', type=str, default='images', help='Path to train dataset')
    parser.add_argument('--test_image_path', type=str, default='images', help='Path to test dataset')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    opt = options()

    classifier = ClusteringClassifier()

    train_dataset = load_dataset(opt.train_image_path, classifier.transforms())
    train_imgs = np.stack([i[1] for i in train_dataset], axis=0)
    print(train_imgs.shape)

    test_dataset = load_dataset(opt.test_image_path, classifier.transforms())
    test_imgs = np.stack([i[1] for i in test_dataset], axis=0)
    print(test_imgs.shape)

    classifier.train_step(train_imgs)
    
    tags = classifier.predict_step(test_imgs)

    cv2.namedWindow('healthy', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('healthy', 900,900)

    cv2.namedWindow('bad', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bad', 900,900)

    for i in range(len(test_imgs)):

        blend_img =  test_dataset[i][2] 

        if tags[i] == 0:
            cv2.imshow('healthy', blend_img)
        else:
            cv2.imshow('bad', blend_img)

        cv2.waitKey(0)
