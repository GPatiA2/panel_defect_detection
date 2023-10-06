import numpy as np
import os
import cv2

class PannelChopper():

    def __init__(self, chop_shape):
        self.out_shape = chop_shape

    def chop(self, image, detections):

        samples = []

        for d in detections:
            coords = d["segmentation"] 
            coords = [np.int32(np.array(p)) for p in coords]

            x,y,w,h = cv2.boundingRect(np.array(coords))

            coords = np.int32([coords])

            mask_cont = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
            mask_cont = cv2.fillPoly(mask_cont, pts = coords, color = (255,255,255))

            only_cont   = cv2.bitwise_and(mask_cont, image)

            masked_chop = only_cont[y:y+h, x:x+w ] 

            masked_chop = cv2.resize(masked_chop, self.out_shape, cv2.INTER_AREA)

            samples.append((coords, masked_chop)) 

        return samples           
    

    def efficient_chop(self, image, detections):

        samples = []

        for d in detections:

            coords = d["segmetnation"]
            coords = [np.int32(np.array(p)) for p in coords]

            x,y,w,h = cv2.boundingRect(np.array(coords))

            mask = np.zeros((h,w,3), np.uint8)

            unmasked_chop = image[y:y+h, x:x+w]

            warped_coords = [np.array([p[0] - h, p[1] - w]) for p in coords]

            mask = cv2.fillPoly(mask, pts = warped_coords, color = (255,255,255))

            masked_chop = cv2.bitwise_and(mask, unmasked_chop)

            if self.out_shape is not None:
                masked_chop = cv2.resize(masked_chop, self.out_shape, cv2.INTER_AREA)

            samples.append((coords, masked_chop))

        return samples