import cv2
import json
import os
import numpy as np
import argparse
from copy import deepcopy
import sklearn.model_selection as sk
import random

def options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type = str)
    parser.add_argument('--dst_file', type = str)
    parser.add_argument('--test_frac', type = float)
    args = parser.parse_args()
    return args

def transform(img):

    mean = np.mean(img[img > 0])
    std  = np.std(img[img > 0])
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img  = cv2.ximgproc.rollingGuidanceFilter(img, numOfIter= 10 )
    img  = cv2.threshold(img, mean + std, 255, cv2.THRESH_BINARY_INV)[1]
    img  = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    img  = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    return img

if __name__ == '__main__':

    opt = options()

    img_list = []
    for f in os.listdir(opt.src_dir):
        if f.endswith(".JPG"):
            path = os.path.join(opt.src_dir, f)
            img  = cv2.imread(path)
            img_list.append((f, img))

    random.shuffle(img_list)

    x = []
    y = []
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800,800)
    cv2.imshow('image', deepcopy(img_list[0][1]))

    cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('filtered', 800,800)
    cv2.imshow('filtered', transform(deepcopy(img_list[0][1])))
    
    k = cv2.waitKey(0)
    i = 0

    while i < len(img_list) and k != ord('f'):

        if k == ord('y'):
            x.append(img_list[i][0])
            y.append(1)
            i += 1
            cv2.imshow('filtered', transform(deepcopy(img_list[i][1])))
            cv2.imshow('image', deepcopy(img_list[i][1]))
            k = cv2.waitKey(0)

        elif k == ord('n'):
            x.append(img_list[i][0])
            y.append(0)
            i += 1
            cv2.imshow('filtered', transform(deepcopy(img_list[i][1])))
            cv2.imshow('image', deepcopy(img_list[i][1]))
            k = cv2.waitKey(0)

        elif k != ord('y') and k != ord('n') and k != ord('f'):
            print("Invalid key pressed. Press 'y' for yes and 'n' for no.")
            k = cv2.waitKey(0)
        
        else:
            pass



    x_train, x_test, y_train, y_test = sk.train_test_split(x, y, test_size=opt.test_frac, shuffle=True, stratify=y)

    train = {x_train[i] : y_train[i] for i in range(len(x_train))}
    test  = {x_test[i]  : y_test[i]  for i in range(len(x_test))}

    ds = {}
    ds["train"] = train
    ds["test"]  = test

    with open(opt.dst_file, 'w') as f:
        json.dump(ds, f, indent=4)


    