
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

    th_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # rect_img = cv2.drawContours(np.stack((th_img.copy(),)*3, axis = -1), [box], 0, (0,0,255),1)
    # cv2.namedWindow('orig', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('orig', 800, 800)
    # cv2.imshow('orig', img)

    # cv2.namedWindow('rect', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('rect', 800, 800)
    # cv2.imshow('rect', rect_img)

    x_coords = [b[0] for b in box]
    y_coords = [b[1] for b in box]

    h = max(x_coords) - min(x_coords)
    w = max(y_coords) - min(y_coords)

    center = (min(x_coords) + h/2, min(y_coords) + w/2)

    orig_angle = rect[2]
    if orig_angle != 90:
        if orig_angle > 45:
            angle = -90 + orig_angle
        else:
            angle = orig_angle
    else:
        return img

    rot_m = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(img, rot_m, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)

    # cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('rotated', 800, 800)
    # cv2.imshow('rotated', image)
    # cv2.waitKey(0)

    return image


if __name__ == "__main__":

    # opt = options()
    # params = vars(opt)

    dataset    = load_dataset('dataset/all')
    
    ds         = [cv2.imread(it[0]) for it in dataset]

    ds_g       = [cv2.cvtColor(it, cv2.COLOR_BGR2GRAY) for it in ds]

    ds_str     = [straighten_img(it) for it in ds_g]

    ds_r       = [cv2.resize(it, (35,50)) for it in ds_str]

    stacked    = np.stack(ds_r, axis=0)

    median_img = np.median(stacked, axis=0)

    cv2.namedWindow("med", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("med", 800, 800)
    cv2.imshow("med", cv2.equalizeHist(np.uint8(median_img)))
    cv2.waitKey(0)

    ds_sus_median = [cv2.subtract(np.uint8(it), np.uint8(median_img)) for it in ds_r]

    ds_t = ds_sus_median

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 800, 800)

    cv2.namedWindow("filtered", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("filtered", 800, 800)
    
    ds_t = []
    for img in ds_sus_median:
        cv2.imshow("original", img)
        maxVal = np.max(img)
        minVal = np.min(img[np.nonzero(img)])
        median = np.median(img[np.nonzero(img)])
        print(maxVal, " ",  minVal, " ", median)
        print(maxVal - minVal)
        print(maxVal - median)
        print("=======")
        median = np.median(img)

        mul = np.multiply(np.float32(img),img/50, dtype=np.float32)
        # img = cv2.boxFilter(img, -1, (11,11))
        img = np.uint8(np.clip(mul, 0, 255 ))
        # img = cv2.bitwise_not(img)
        # img = cv2.convertScaleAbs(img, alpha=.5, beta=0)


        # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, (5,5))
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5,5))
        # img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow("filtered", img)
        ds_t.append(img)
        cv2.waitKey(0)




    # for i in range(len(ds_t)):
    #     cv2.imshow("filtered", ds_t[i])
    #     cv2.imshow("original", ds_r[i])
    #     cv2.waitKey(0)
