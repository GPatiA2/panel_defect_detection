import cv2
import numpy as np
import os 
import argparse
import json
from thermal import Thermal 
import random

def options():

    parser = argparse.ArgumentParser(description='Local Maxima Based Detector')
    parser.add_argument('--im_path', type=str, help='Path to imgs')
    parser.add_argument('--dpan_idx', type = int, help='Index of the defective pannel')
    parser.add_argument('--json_file', type=str, default = 'dataset/bad_pannels.json' , help='Path to json file')
    args = parser.parse_args()
    print(args)
    return args

def get_thermal(path) -> np.array:

    thermal = Thermal(
        dirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libdirp.so',
        dirp_sub_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_dirp.so',
        iirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_iirp.so',
        exif_filename='plugins/exiftool-12.35.exe',
        dtype=np.float32,
    )

    temperature = thermal.parse_dirp2(path)

    return np.uint8(temperature)

def crop_bbox(img : np.array, idx : int, pannels) -> np.array:

    rect = cv2.boundingRect(np.array(pannels[idx]["bbox"]))
    rect = np.intp(rect)
    img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    return img, rect

def apply_local_max_filter(thermal_crop, intensity):

    crop = thermal_crop.copy()
    dilation = cv2.dilate(crop, np.ones((intensity, intensity), np.uint8), iterations=1)
    dilation = cv2.ximgproc.rollingGuidanceFilter(dilation, 3)

    highlight = np.zeros_like(crop)
    highlight[np.logical_and(dilation == thermal_crop, dilation != 0, thermal_crop > 23)] = 255

    return highlight

def calculate_neighbours(point):

    # ----------------------------------------
    # TODO: check if point is within the image
    # ----------------------------------------

    NEIGH_X = [-1, 0, 1]
    NEIGH_Y = [-1, 0, 1]

    neighbours = []

    for x in NEIGH_X:
        for y in NEIGH_Y:
            if x == 0 and y == 0:
                continue
            else:
                neighbours.append((point[0] + x, point[1] + y))

    return neighbours

def breadth_walk(cont_masks, contours, thermal_crop):

    NOT_VISITED = 0
    IN_QUEUE    = 1
    VISITED     = 2

    visited    = np.zeros_like(thermal_crop, dtype=np.uint8)
    blobs      = np.zeros_like(thermal_crop, dtype=np.uint8)
    temp_proc  = np.zeros_like(thermal_crop, dtype=np.uint8)
    
    visited.fill(NOT_VISITED)
    blobs.fill(NOT_VISITED)
    temp_proc.fill(NOT_VISITED)

    queue      = []

    for cont in contours:

        for p in cont:
            p_ = (p[0][0], p[0][1])

            max_in_cont = np.max(thermal_crop[cont_masks[contours.index(cont)] == 255])

            temp_proc[p_[0], p_[1]] = max_in_cont
            queue.append((p_, contours.index(cont) + 1))

            visited[p_[0], p_[1]] = IN_QUEUE
            blobs[p_[0], p_[1]] = contours.index(cont) + 1

    random.shuffle(queue)

    print("[info] Starting breadth walk")
    while len(queue) > 0:

        point, procedence = queue.pop(0)

        visited[point[0], point[1]] = VISITED
        blobs[point[0], point[1]] = procedence
        
        neighbours = calculate_neighbours(point)

        for n in neighbours: 
            
            dif = temp_proc[point[0], point[1]] - thermal_crop[n[0], n[1]]
            
            if visited[n[0], n[1]] == NOT_VISITED:
                if dif < 3:
                    queue.append(((n[0], n[1]), procedence))
                    temp_proc[n[0], n[1]] = temp_proc[point[0], point[1]]
                    visited[n[0], n[1]] = VISITED
                else:
                    print("Not appending point (" + str(n[0]) + ", " + str(n[1]) + ") with value = " + str(thermal_crop[point[0], point[1]]) + " and dif = " + str(dif))
            
            elif visited[n[0], n[1]] == IN_QUEUE:
                print("updating temp proc")
                temp_proc[n[0], n[1]] = min(temp_proc[n[0], n[1]], temp_proc[point[0], point[1]])

            else:
                print("Already visited")

    return blobs



if __name__ == '__main__':

    args = options()

    im_path = args.im_path
    json_file = args.json_file
    im_name = os.path.basename(im_path)[:-4]
    idx = args.dpan_idx

    with open(json_file, 'r') as f:
        all_pannels = json.load(f)

    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    thermal = get_thermal(im_path)

    pannels = all_pannels[im_name]

    rgb_crop = crop_bbox(img, idx, pannels)[0]
    thermal_crop = crop_bbox(thermal, idx, pannels)[0]

    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rgb', 800,800)
    cv2.imshow('rgb', rgb_crop)

    cv2.namedWindow('thermal', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thermal', 800,800)
    cv2.imshow('thermal', thermal_crop)

    cv2.waitKey(0)

    local_max_filter = apply_local_max_filter(thermal_crop, 3)
    cv2.imshow('thermal', np.stack([thermal_crop, thermal_crop, local_max_filter], axis = -1))
    cv2.imshow('rgb', np.stack([rgb_crop, thermal_crop, local_max_filter], axis = -1))
    cv2.waitKey(0)

    for i in range(5, 10000, 2):
        local_max_filter_new = apply_local_max_filter(thermal_crop, i)
        if np.count_nonzero(local_max_filter) == np.count_nonzero(local_max_filter_new):
            print("[info] Final filter intensity = " , i)
            cv2.imshow('thermal', np.stack([thermal_crop, thermal_crop, local_max_filter_new], axis = -1))
            cv2.imshow('rgb', np.stack([rgb_crop, thermal_crop, local_max_filter_new], axis = -1))
            cv2.waitKey(0)
            break

        else:
            local_max_filter = local_max_filter_new.copy()
            continue

    contours, hierarchy = cv2.findContours(local_max_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thermal_rgb = np.stack([thermal_crop, thermal_crop, thermal_crop], axis = -1)
    cont_img = cv2.drawContours(thermal_rgb, contours, -1, (255,0,0), 1)
    cv2.namedWindow('thermal_conts', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thermal_conts', 800,800)
    cv2.imshow('thermal_conts', cont_img)
    cv2.waitKey(0)

    cont_mask = []
    for cont in contours:
        cont_mask.append(cv2.fillPoly(np.zeros_like(local_max_filter), [cont], 255))
        
    print("[info] Found " + str(len(contours)) + " blobs")
    print("[info] Generated " + str(len(cont_mask)) + " masks")
    walks = breadth_walk(cont_mask, contours, thermal_crop)

    unique_vals, repetitions = np.unique(walks, return_counts=True)
    for i in range(len(unique_vals)):
        if i == 0:
            print("[info] There are " + str(repetitions[i]) + " pixels not belonging to any blob")
        else:
            print("[info] Blob " + str(unique_vals[i]) + " has " + str(repetitions[i]) + " pixels")

    blobs = []
    for i in range(1, len(contours) + 1):

        blob = np.zeros_like(thermal_crop)
        blob = np.where(walks == i, 255, 0)
        blob = np.uint8(blob)
        blobs.append(blob)
        
    k = 1
    for b in blobs:
        cv2.namedWindow('blobs ' + str(k), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blobs ' + str(k), 800,800)
        cv2.imshow('blobs ' + str(k), np.stack([b, thermal_crop, thermal_crop], axis = -1)) 
        cv2.waitKey(0)
        k += 1
    