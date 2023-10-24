
import cv2
import json
import os
import numpy as np
import argparse
from copy import deepcopy
from matplotlib import pyplot as plt

class ContourClassifier():

    def __init__(self, params):
        self.rolling_guidance_iters = params['rolling_iters']

    def transforms(self):

        def preprocess(img):
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img  = cv2.ximgproc.rollingGuidanceFilter(img, numOfIter= 10)
            cv2.namedWindow('rgf', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('rgf', 800, 800)
            cv2.imshow('rgf', img)
            # img  = cv2.boxFilter(img, -1, (5,5))
            # cv2.namedWindow('box', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('box', 800, 800)
            # cv2.imshow('box', img)
            # img  = cv2.Canny(img, 100, 200)
            mean = np.mean(img[img > 0])
            std  = np.std(img[img > 0])
            img = cv2.threshold(img, mean+ std, 255, cv2.THRESH_BINARY)[1]

            return (img, mean, std)
        
        return preprocess

    def predict(self, image):

        im = deepcopy(image)
        im, mean, std = self.transforms()(im)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        br = maxVal > mean + std

        return im, maxLoc, br

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

    parser = argparse.ArgumentParser(description='Contour Classifier')
    parser.add_argument('--images_dir', type=str, default='data/contour_classifier/images', help='path to images')
    parser.add_argument('--rolling_iters', type=int, default=5, help='number of iterations for rolling guidance filter')
    
    opt = parser.parse_args()
    print(opt)

    return opt

def straighten_img(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    maxArea = 0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxArea :
            maxArea = area
            best = contour

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    #crop image inside bounding box
    scale = 1  # cropping margin, 1 == no margin
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    rotated = False
    if angle > 45:
        angle -= 90
        rotated = True

    center = (int((x1+x2)/2), int((y1+y2)/2))
    size = (int(scale*(x2-x1)), int(scale*(y2-y1)))

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if rotated else H
    croppedH = H if rotated else W

    image = cv2.getRectSubPix(cropped, (int(croppedW*scale), int(croppedH*scale)), (size[0]/2, size[1]/2))

    return image


if __name__ == "__main__":

    opt = options()
    params = vars(opt)

    dataset    = load_dataset(params['images_dir'])
    ds         = [cv2.imread(it[0]) for it in dataset]

    ds_g       = [cv2.cvtColor(it, cv2.COLOR_BGR2GRAY) for it in ds]

    ds_str     = [straighten_img(it) for it in ds_g]

    for im in ds_str:
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('im', 800, 800)
        cv2.imshow('im', im)
        cv2.waitKey(0)

    ds_r       = [cv2.resize(it, (35,50)) for it in ds_g]

    stacked    = np.stack(ds_r)

    median_img = np.median(stacked, axis=0)
    print(median_img.shape)

    print(ds_r[0].shape)

    print(median_img)
    cv2.namedWindow("med", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("med", 800, 800)
    cv2.imshow("med",np.uint8(median_img))
    cv2.waitKey(0)

    ds_sus_median = [cv2.subtract(np.uint8(it), np.uint8(median_img)) for it in ds_r]

    def get_kernel(size):
        k = np.ones((size,size),np.uint8)
        k[int(size/2), int(size/2)] = 0
        print(k) 
        return k


    t1 = lambda x : cv2.GaussianBlur(x, (7,7), 3)
    t2 = lambda x : x 

    t = lambda x : t2(t1(x))
    ds_t = [t(it) for it in ds_sus_median]

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 800, 800)

    cv2.namedWindow("filtered", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("filtered", 800, 800)

    for i in range(len(ds_t)):
        cv2.imshow("filtered", ds_t[i])
        cv2.imshow("original", ds_r[i])
        cv2.waitKey(0)
