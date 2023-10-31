
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
        self.median_img = cv2.imread(params['median_img'], cv2.IMREAD_GRAYSCALE)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.maxCircularity = 1
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False
        params.minDistBetweenBlobs = 1
        self.detector = cv2.SimpleBlobDetector_create(params)

    def transforms(self):

        def preprocess(img):
            cpy = deepcopy(img)
            cpy_g = cv2.cvtColor(cpy, cv2.COLOR_BGR2GRAY)
            # cpy_str = self.straighten_img(cpy_g)
            cpy_r = cv2.resize(cpy_g, (35,50))
            cpy_sus_median = cv2.subtract(np.uint8(cpy_r), np.uint8(self.median_img))
            cpy_t = cpy_sus_median
            cpy_med = cv2.boxFilter(cpy_t, -1, (3,3))
            cpy_mul = np.multiply(np.float32(cpy_med),cpy_med/60, dtype=np.float32)
            cpy_rol = cv2.ximgproc.rollingGuidanceFilter(cpy_mul, numOfIter= self.rolling_guidance_iters)
            cpy = np.uint8(np.clip(cpy_rol, 0, 255 ))

            return cpy
        
        return preprocess

    def predict_step(self, image):

        im = deepcopy(image)
        im_t = self.transforms()(im)

        keypoints = self.detector.detect(im_t)
        if len(keypoints) == 0:
            return 0
        else:
            return 1

    def predict_with_kp(self, image):

        im = deepcopy(image)
        im_t = self.transforms()(im)

        keypoints = self.detector.detect(im_t)
        if len(keypoints) == 0:
            return 0, [], im_t
        else:
            return 1, keypoints, im_t
         

    def straighten_img(self, img):

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

        return image
    
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
    parser.add_argument('--median_img', type=str, default='median.png', help='path to median image')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')
    opt = parser.parse_args()
    print(opt)

    return opt

def debug(ds, detector):

    ds_g       = [cv2.cvtColor(it, cv2.COLOR_BGR2GRAY) for it in ds]

    ds_str     = [detector.straighten_img(it) for it in ds_g]

    ds_r       = [cv2.resize(it, (35,50)) for it in ds_str]

    stacked    = np.stack(ds_r, axis=0)

    median_img = np.median(stacked, axis=0)

    cv2.imwrite("median.png", np.uint8(median_img))

    cv2.namedWindow("med", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("med", 800, 800)
    cv2.imshow("med", cv2.equalizeHist(np.uint8(median_img)))
    cv2.waitKey(0)

    ds_sus_median = [cv2.subtract(np.uint8(it), np.uint8(median_img)) for it in ds_r]

    ds_t = ds_sus_median

    cv2.namedWindow("filtered", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("filtered", 800, 800)

    cv2.namedWindow("processed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("processed", 800, 800)
    
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 800, 800)

    cv2.namedWindow("rgf", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rgf", 800, 800)

    ds_t = []
    i = 0
    for img in ds_sus_median:
        cv2.imshow("original", ds_str[i])
        i += 1
        
        cv2.imshow("filtered", img)
        maxVal = np.max(img)
        minVal = np.min(img[np.nonzero(img)])
        median = np.median(img[np.nonzero(img)])
        print(maxVal, " ",  minVal, " ", median)
        print(maxVal - minVal)
        print(maxVal - median)
        print("=======")
        median = np.median(img)

        mul = cv2.ximgproc.rollingGuidanceFilter(img, numOfIter= 10)
        cv2.imshow("rgf", mul)
        mul = np.multiply(np.float32(mul),mul//50, dtype=np.float32)
        # img = cv2.boxFilter(img, -1, (11,11))
        img = np.uint8(np.clip(mul, 0, 255 ))

        cv2.imshow("processed", img)
        ds_t.append(img)
        cv2.waitKey(0)


if __name__ == "__main__":

    opt = options()
    detector = ContourClassifier(vars(opt))

    dataset    = load_dataset('dataset/all')
    
    ds         = [cv2.imread(it[0], cv2.IMREAD_COLOR ) for it in dataset]

    if opt.debug:
        debug(ds, detector)
        exit()
    
    else:
        cv2.namedWindow("ORIGINAL", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ORIGINAL", 800, 800)

        cv2.namedWindow("FILTERED", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FILTERED", 800, 800)

        cv2.namedWindow("DEFECT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEFECT", 800, 800)

        cv2.namedWindow("HEALTHY", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HEALTHY", 800, 800)

        for it in ds:
            it_2 = it.copy()
            it_2 = detector.straighten_img(cv2.cvtColor(it_2, cv2.COLOR_BGR2GRAY))
            it_2 = cv2.cvtColor(it_2, cv2.COLOR_GRAY2BGR)
            pred, kp, imt_t = detector.predict_with_kp(it)
            cv2.imshow("ORIGINAL", it)
            cv2.imshow("FILTERED", imt_t)
            if pred == 1:
                print(len(kp))
                for p in kp:
                    it_2 = cv2.circle(it_2, (int(p.pt[0]), int(p.pt[1])), int(p.size) + 4, (0,0,255), 1)
                cv2.imshow("DEFECT", it_2)
            else:
                cv2.imshow("HEALTHY", it)     

            cv2.waitKey(0)   


    