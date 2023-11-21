from pannel_detector import PannelDetector
import cv2
import argparse
import os
import numpy as np
import json

def options():
    parser = argparse.ArgumentParser(description='Pannel detector')
    parser.add_argument('--input_dir', type=str, default='rtkimgs/reduced_calamocha_sector7', help='Path to input images')
    parser.add_argument('--result_dir', type=str, default='rtkimgs/reduced_calamocha_sector7', help='Path to output images')
    opt = parser.parse_args()
    return opt

def read_images(path):
    images = []
    for img_path in os.listdir(path):
        if img_path.endswith('.JPG'):
            img = cv2.imread(os.path.join(path, img_path))
            images.append((img, img_path))
    return images

opt = options()

os.makedirs(opt.result_dir, exist_ok=True)

detector = PannelDetector('weights/model_final_rgb.pth')

images = read_images(opt.input_dir)

results = {}

for img in images:

    detections = detector.detect(img[0])
    print(detections)
    conts = [d['segmentation'] for d in detections]
    results[img[1]] = conts

    # im3 = img[0].copy()
    # im3 = cv2.drawContours(im3, conts, -1, (0,255,0), 2)

    # print(len(conts))

    # cv2.namedWindow(img[1], cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(img[1], (1920, 1080))
    # cv2.imshow(img[1], im3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

with open(opt.result_dir + '/results.json', 'w') as f:
    json.dump(results, f, indent=4)
