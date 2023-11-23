import cv2
import numpy as np
import os 
import argparse
import json
from thermal import Thermal 

class DefectClassifier():

    NOT_VISITED = 0
    IN_QUEUE    = 1
    VISITED     = 2

    def __init__(self, params):

        self.params = params

    def load(path):

        with open(path, 'r') as f:
            params = json.load(f)

        return DefectClassifier(params)

    def local_max_filter(self, img, intensity):

        cpy      = img.copy()
        dilation = cv2.dilate(cpy, np.ones((intensity, intensity), np.uint8), iterations = 1) 

        filtered = np.zeros_like(img)
        filtered[np.logical_and(dilation == img, dilation > 0, img > 0)] = 255

        return filtered
    
    def local_min_filter(self, img, intensity):

        cpy      = img.copy()
        dilation = cv2.erode(cpy, np.ones((intensity, intensity), np.uint8), iterations = 1) 

        filtered = np.zeros_like(img)
        filtered[np.logical_and(dilation == img, dilation > 0, img > 0)] = 255

        return filtered
    
    def get_classes(self):
        
        l = ["UNKNOWN"]
        for def_class in self.params['criteria'].items():
            l.append(def_class[0])

        return l

    def apply_local_max_filter(self, img):

        initial_image = img.copy()
        
        lmf0 = self.local_max_filter(initial_image, self.params['int0'])
        iters = self.params['int0'] + self.params['lmf_iter']*self.params['int_step']
        for i in range(self.params['int0'] + self.params['int_step'],iters, self.params['int_step']):
            lmfi = self.local_max_filter(initial_image, self.params['int0'] + i*self.params['int_step'])

            if np.count_nonzero(lmfi) == np.count_nonzero(lmf0):
                break

            else:
                lmf0 = lmfi.copy()

        return lmfi
    
    def apply_local_min_filter(self, img):

        initial_image = img.copy()
        
        lmf0 = self.local_max_filter(initial_image, self.params['int0'])
        iters = self.params['int0'] + self.params['lmf_iter']*self.params['int_step']
        for i in range(self.params['int0'] + self.params['int_step'],iters, self.params['int_step']):
            lmfi = self.local_min_filter(initial_image, self.params['int0'] + i*self.params['int_step'])

            if np.count_nonzero(lmfi) == np.count_nonzero(lmf0):
                break

            else:
                lmf0 = lmfi.copy()

        return lmfi
    
    def get_lm_contours(self, lm_img):

        contours, _ = cv2.findContours(lm_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cont_mask = []
        for cont in contours:
            cont_mask.append(cv2.fillPoly(np.zeros_like(lm_img), [cont], 255))

        return contours, cont_mask
    
    def calculate_neighbours(self, point, shape):

        NEIGH_X = [-1, 0, 1]
        NEIGH_Y = [-1, 0, 1]

        neighbours = []

        for x in NEIGH_X:
            for y in NEIGH_Y:
                px = point[0] + x
                py = point[1] + y
                if x == 0 and y == 0:
                    continue
                elif px >= 0 and px < shape[0] and py >= 0 and py < shape[1]:
                    neighbours.append((px, py))

        return neighbours
    
    def breadth_walk(self, cont_masks, contours, thermal_crop):

        # Matrix to store whether a pixel has been visited or not
        visited     = np.zeros_like(thermal_crop, dtype=np.uint8)

        # Matrix to store the blob which a pixel belongs to
        blobs       = np.zeros_like(thermal_crop, dtype=np.uint8)

        # Matrix to store the maximum temperature of the pixel within the blob it is contained
        temp_proc   = np.zeros_like(thermal_crop, dtype=np.uint8)

        # Blob type list
        blob_types = ["UNKNOWN" for i in range(len(contours))]
        
        visited.fill(self.NOT_VISITED)
        blobs.fill(self.NOT_VISITED)
        temp_proc.fill(self.NOT_VISITED)

        queue      = []

        # Initialization
        for i in range(len(contours)):

            cont = contours[i]
            
            # Find max value in the area contained in a contour
            max_in_cont = np.max(thermal_crop[cont_masks[i] == 255])
            
            # Add all the points in the contour to the queue
            # The points are stored along with the blob they belong to
            # 
            # The blob is to be changed while the point is in the queue
            #     so that the point is assigned to the blob with the closest
            #     temperature

            for p in cont:
                p_ = (p[0][1], p[0][0])

                temp_proc[p_[0], p_[1]] = max_in_cont
                queue.append(p_)

                visited[p_[0], p_[1]] = self.IN_QUEUE
                blobs[p_[0], p_[1]] = i + 1

            visited[cont_masks[i] == 255] = self.VISITED
            blobs[cont_masks[i] == 255] = i + 1
            temp_proc[cont_masks[i] == 255] = max_in_cont 

        while len(queue) > 0:

            # Pop the point from the queue, along with the definitive blob it belongs to
            point = queue.pop(0)

            # Mark it as visited, so that the blob it belongs to becomes definitive
            visited[point[0], point[1]] = self.VISITED

            # Calculate the neighbours of the point and iterate over them        
            neighbours = self.calculate_neighbours(point, thermal_crop.shape)

            for n in neighbours: 
                
                # Calculate the difference in temperature between the point and the hottest
                #   pixel in the blob it belongs to
                dif = temp_proc[point[0], point[1]] - thermal_crop[n[0], n[1]]
                
                # If the neighbour has not been visited
                if visited[n[0], n[1]] == self.NOT_VISITED and thermal_crop[n[0], n[1]] > 0:

                    # If the difference in temperature is less than the threshold, assign it to the blob
                    if dif < self.params['threshold']:
                        queue.append((n[0], n[1]))
                        temp_proc[n[0], n[1]] = temp_proc[point[0], point[1]]
                        visited[n[0], n[1]] = self.IN_QUEUE
                        blobs[n[0], n[1]] = blobs[point[0], point[1]]

                    else:
                        # Sort the criteria in ascending order of upper bound
                        criteria = list(self.params['criteria'].items())
                        criteria.sort(key = lambda x : x[1])

                        # Mark the whole blob to which the point belongs as the type corresponding to the
                        #   type in which the difference in temperature falls
                        n = 0
                        for n in range(len(criteria)):
                            min_bound = self.params['threshold'] if n == 0 else criteria[n-1][1]
                            max_bound = criteria[n][1]
                            if dif >= min_bound and dif < max_bound:
                                blob_types[blobs[point[0], point[1]] - 1] = criteria[n][0]
                                break

                # If the point is in the queue, update the temperature of the hottest pixel in the blob
                #       it belongs to with the lowest temperature of:
                #
                #           - The hottest point in the blob of the neighbour which first added the point to the queue
                #                   or last updated this value
                # 
                #           -  The temperature of the hottest point in the blob of the neighbour that is currently adding
                #                   the point to the queue
                #
                # Also, if the temperature of the hottest point within the blob to which the point who added the current point
                #   to the queue is updated, also update the blob to which the current point belongs to the new value
                elif visited[n[0], n[1]] == self.IN_QUEUE:

                    if temp_proc[point[0], point[1]] < temp_proc[n[0], n[1]]:
                        temp_proc[n[0], n[1]] = temp_proc[point[0], point[1]]
                        blobs[n[0], n[1]] = blobs[point[0], point[1]]

        return (blobs, blob_types)
    
    def classify(self, thermal_crop):

        lmax_img = self.apply_local_max_filter(thermal_crop)
        lmin_img = self.apply_local_min_filter(thermal_crop)

        contours, cont_masks = self.get_lm_contours(lm_img)

        blobs, blob_types = self.breadth_walk(cont_masks, contours, thermal_crop)

        return blobs, blob_types