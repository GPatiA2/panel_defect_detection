import cv2
import numpy as np
import os 
import argparse
import json
from thermal import Thermal 

class DefectClassifier():

    def __init__(self, params):

        self.params = params

    def load(path):

        with open(path, 'r') as f:
            params = json.load(f)

        return DefectClassifier(params)
    
    def get_classes(self):
            
        l = list(self.params['criteria'].keys())
        l = sorted(l, key = lambda x: self.params['criteria'][x])

        return l

    def local_min_filter(self, img, intensity):

        cpy      = img.copy()
        dilation = cv2.erode(cpy, np.ones((intensity, intensity), np.uint8), iterations = 1) 

        filtered = np.zeros_like(img)
        filtered[np.logical_and(dilation == img, dilation > 0, img > 0)] = 255

        return filtered

    def apply_local_min_filter(self, img):

        initial_image = img.copy()
        
        lmf0 = self.local_min_filter(initial_image, 3)
        iters = 1000
        for i in range(5,iters, 2):
            lmfi = self.local_min_filter(initial_image, i)

            if np.count_nonzero(lmfi) == np.count_nonzero(lmf0):
                break

            else:
                lmf0 = lmfi.copy()

        return lmfi

    def local_max_filter(self, thermal_crop, intensity):

        crop = thermal_crop.copy()
        dilation = cv2.dilate(crop, np.ones((intensity, intensity), np.uint8), iterations=1)

        highlight = np.zeros_like(crop)
        highlight[np.logical_and(dilation == thermal_crop, dilation != 0, thermal_crop > 23)] = 255

        return highlight

    def apply_local_max_filter(self, img):
        
        initial_image = img.copy()
        
        lmf0 = self.local_max_filter(initial_image, 3)
        iters = 1000
        for i in range(5,iters, 2):
            lmfi = self.local_max_filter(initial_image, i)

            if np.count_nonzero(lmfi) == np.count_nonzero(lmf0):
                break

            else:
                lmf0 = lmfi.copy()

        return lmfi

    def detect(self, thermal_crop):
 
        local_mins_mask = cv2.erode(thermal_crop, np.ones((5, 5), np.uint8), iterations = 1)
        local_mins_mask = np.where(np.logical_and(local_mins_mask == thermal_crop, thermal_crop != 0), 255, 0)
        local_mins_mask = np.uint8(local_mins_mask)
        local_maxs_mask = self.apply_local_max_filter(thermal_crop)

        local_mins_idx = np.argwhere(local_mins_mask == 255)
        local_maxs_idx = np.argwhere(local_maxs_mask == 255)

        cv2.waitKey(0)  

        t = thermal_crop.copy()
        t = np.int32(t)

        local_mins_idx = sorted(local_mins_idx, key = lambda x: -t[x[0], x[1]])
        local_maxs_idx = sorted(local_maxs_idx, key = lambda x: -t[x[0], x[1]])

        criteria = list(self.params['criteria'].items())
        criteria.append(("NO DEFECT", 0))
        criteria = sorted(criteria, key = lambda x: x[1], reverse = True)
        
        min_idx = 0
        max_idx = 0
        defect  = criteria[-1][0] 
        defect_idx = len(criteria) - 1
        dist    = 0

        # print("CRITERIA = ", criteria)
        # print("LEN LOCAL MAXS = ", len(local_maxs_idx))
        # print("LEN LOCAL MINS = ", len(local_mins_idx))

        while defect != criteria[0][0] and max_idx < len(local_maxs_idx):
            while defect != criteria[0][0] and min_idx < len(local_mins_idx):
                
                if t[local_mins_idx[min_idx][0], local_mins_idx[min_idx][1]] < self.params['low_limit']:
                    break

                max_val = t[local_maxs_idx[max_idx][0], local_maxs_idx[max_idx][1]]
                min_val = t[local_mins_idx[min_idx][0], local_mins_idx[min_idx][1]]

                dif = max_val - min_val
                dist = np.sqrt((local_maxs_idx[0][0] - local_mins_idx[min_idx][0])**2 + (local_maxs_idx[0][1] - local_mins_idx[min_idx][1])**2)
                
                if dif > 20:
                    defect = "HIGH"
                elif dif <= 20 and dif > 10 and defect != "HIGH":
                    defect = "MEDIUM"
                elif dif <= 10 and dif > 5 and defect != "HIGH" and defect != "MEDIUM":
                    defect = "LOW"
                else:
                    defect = "NO DEFECT"
                
                min_idx += 1

            max_idx += 1

        return defect, dist, (local_maxs_idx[max_idx - 1][0], local_maxs_idx[max_idx - 1][1]), (local_mins_idx[min_idx - 1][0], local_mins_idx[min_idx - 1][1])
    
    def classify(self, thermal_crop):

        defect, dist, max_loc, min_loc = self.detect(thermal_crop)

        return defect