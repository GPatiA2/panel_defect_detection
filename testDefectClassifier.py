import cv2
import numpy as np
import os 
import argparse
import json
from thermal import Thermal 
import random
from pannel_detector import PannelDetector
from pannel_chopper import PannelChopper

NOT_VISITED = 0
IN_QUEUE    = 128
VISITED     = 255

UNKNOWN     = 0
LOW         = 1
MEDIUM      = 2
HIGH        = 3

C_UK        = (0,47,189)
C_LOW       = (189,156,0)
C_MEDIUM    = (189,99,0)
C_HIGH      = (189,0,0)

BLOB_COLORS = [C_UK, C_LOW, C_MEDIUM, C_HIGH]

THRESHOLD   = 5

def options():

    parser = argparse.ArgumentParser(description='Local Maxima Based Detector')
    parser.add_argument('--im_path', type=str, help='Path to imgs')
    parser.add_argument('--mode', type=str, default=False, help='Use full image')
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
        dtype=np.int16,
    )

    temperature = thermal.parse_dirp2(path)

    return np.uint8(temperature)

def crop_bbox(img : np.array, idx : int, pannels) -> np.array:

    rect = cv2.boundingRect(np.array(pannels[idx]["bbox"]))
    rect = np.intp(rect)
    img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    return img, rect

def crop_cont(img : np.array, cont) -> np.array:

    cpy = img.copy()
    rect = cv2.boundingRect(cont)
    rect = np.intp(rect)
    cpy = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    return cpy, rect

def apply_local_max_filter(thermal_crop, intensity):

    crop = thermal_crop.copy()
    dilation = cv2.dilate(crop, np.ones((intensity, intensity), np.uint8), iterations=1)
    # dilation = cv2.ximgproc.rollingGuidanceFilter(dilation, 3)

    highlight = np.zeros_like(crop)
    highlight[np.logical_and(dilation == thermal_crop, dilation != 0, thermal_crop > 23)] = 255

    return highlight

def calculate_neighbours(point, shape):

    # ----------------------------------------
    # TODO: check if point is within the image
    # ----------------------------------------

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

def highlight_in(highlight, img):

    img = np.uint8(np.stack([highlight, img, img], axis = -1))
    img[0] = img[0] * 255
    img[1] = img[1] * 255
    img[2] = img[2] * 255
    return img

def show_state(thermal_crop, temp_proc, visited, blobs, blob_types):

    cv2.imshow('temp_proc', temp_proc)
 
    cv2.imshow('visited', visited)

    cv2.imshow('blobs', blobs)

    blob_mat = [np.zeros((thermal_crop.shape[0], thermal_crop.shape[1], 3), np.uint8) for _ in range(len(blob_types))]
    
    for i in range(len(blob_types)):
        cv2.namedWindow('blob ' + str(i), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blob ' + str(i), 800,800)
        blob_mat[i][blobs == i + 1] = BLOB_COLORS[blob_types[i]]
        blob_mat[i] = np.uint8(blob_mat[i])
        cv2.imshow('blob ' + str(i), blob_mat[i])

    cv2.waitKey(10)

def breadth_walk(cont_masks, contours, thermal_crop):

    # Matrix to store whether a pixel has been visited or not
    visited     = np.zeros_like(thermal_crop, dtype=np.uint8)

    # Matrix to store the blob to which a pixel belongs
    blobs       = np.zeros_like(thermal_crop, dtype=np.uint8)

    # Matrix to store the maximum temperature of the pixel within the blob it is contained
    temp_proc   = np.zeros_like(thermal_crop, dtype=np.uint8)

    # Blob type list
    blob_types = [UNKNOWN for _ in range(len(contours))]

    cv2.namedWindow('temp_proc', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('temp_proc', 800,800)
    cv2.imshow('temp_proc', temp_proc)

    cv2.namedWindow('visited', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('visited', 800,800)
    cv2.imshow('visited', visited)

    cv2.namedWindow('blobs', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('blobs', 800,800)
    cv2.imshow('blobs', blobs)
    
    visited.fill(NOT_VISITED)
    blobs.fill(NOT_VISITED)
    temp_proc.fill(NOT_VISITED)

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

            visited[p_[0], p_[1]] = IN_QUEUE
            blobs[p_[0], p_[1]] = i + 1

        visited[cont_masks[i] == 255] = VISITED
        blobs[cont_masks[i] == 255] = i + 1
        temp_proc[cont_masks[i] == 255] = max_in_cont 

    cv2.imshow('temp_proc', temp_proc)
    cv2.imshow('visited', visited)
    cv2.imshow('blobs', blobs)
    cv2.waitKey(0)

    random.shuffle(queue)

    print("[info] Starting breadth walk")
    while len(queue) > 0:

        # Pop the point from the queue, along with the definitive blob it belongs to
        point = queue.pop(0)

        # Mark it as visited, so that the blob it belongs to becomes definitive
        visited[point[0], point[1]] = VISITED

        # Calculate the neighbours of the point and iterate over them        
        neighbours = calculate_neighbours(point, thermal_crop.shape)

        for n in neighbours: 
            
            # Calculate the difference in temperature between the point and the hottest
            #   pixel in the blob it belongs to
            dif = temp_proc[point[0], point[1]] - thermal_crop[n[0], n[1]]
            
            # If the neighbour has not been visited
            if visited[n[0], n[1]] == NOT_VISITED and thermal_crop[n[0], n[1]] > 0:

                # If the difference in temperature is less than 5ºC, assign it to the blob
                if dif < THRESHOLD:
                    queue.append((n[0], n[1]))
                    temp_proc[n[0], n[1]] = temp_proc[point[0], point[1]]
                    visited[n[0], n[1]] = IN_QUEUE
                    blobs[n[0], n[1]] = blobs[point[0], point[1]]

                # Mark the whole blob to which the point belongs as LOW
                elif dif > THRESHOLD and dif <= 10:
                    blob_types[blobs[point[0], point[1]] - 1] = max(LOW, blob_types[blobs[point[0], point[1]] - 1])
                    print("FOUND LOW ON POINT ", point, " WITH MAXIMUM ", temp_proc[point[0], point[1]], " AND NEIGHBOUR ", n, " WITH ", thermal_crop[n[0], n[1]], " WITH DIFF ", dif)

                # Mark the whole blob to which the point belongs as MEDIUM
                elif dif > 10 and dif <= 20:
                    blob_types[blobs[point[0], point[1]] - 1] = max(MEDIUM, blob_types[blobs[point[0], point[1]] - 1])
                    print("FOUND MEDIUM ON POINT ", point, " WITH MAXIMUM ", temp_proc[point[0], point[1]], " AND NEIGHBOUR ", n, " WITH ", thermal_crop[n[0], n[1]], " WITH DIFF ", dif)

                # Mark the whole blob to which the point belongs as HIGH
                elif dif > 20:
                    blob_types[blobs[point[0], point[1]] - 1] = max(HIGH, blob_types[blobs[point[0], point[1]] - 1])
                    print("FOUND HIGH ON POINT ", point, " WITH MAXIMUM ", temp_proc[point[0], point[1]], " AND NEIGHBOUR ", n, " WITH ", thermal_crop[n[0], n[1]], " WITH DIFF ", dif)

            # If the point is in the queue, update the temperature of the hottest pixel in the blob
            #       it belongs to with the lowest temperature of:
            #
            #           - The hottest point in the blob of the neighbour which first added the point to the queue
            #                   or last updated this value
            # 
            #           -  The temperature of the hottest point in the blob of the neighbour that is currently adding
            #                   the point to the queue
            #
            #
            # Also, if the temperature of the hottest point within the blob to which the point who added the current point
            #   to the queue is updated, also update the blob to which the current point belongs to the new value
            elif visited[n[0], n[1]] == IN_QUEUE:

                if temp_proc[point[0], point[1]] < temp_proc[n[0], n[1]]:
                    temp_proc[n[0], n[1]] = temp_proc[point[0], point[1]]
                    blobs[n[0], n[1]] = blobs[point[0], point[1]]

        show_state(thermal_crop, temp_proc, visited, blobs, blob_types)

    return blobs, blob_types

def detect(rgb_crop, thermal_crop):

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
    walks, types = breadth_walk(cont_mask, contours, thermal_crop)

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
        cv2.namedWindow('blobs rgb ' + str(k) + ' type ' + str(types[k])+ ' thermal', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blobs rgb ' + str(k) + ' type ' + str(types[k])+ ' thermal', 800,800)
        cv2.imshow('blobs rgb ' + str(k) + ' type ' + str(types[k])+ ' thermal', np.stack([b, thermal_crop, thermal_crop], axis = -1)) 

        cv2.namedWindow('blobs rgb ' + str(k) + ' type ' + str(types[k]) , cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blobs rgb ' + str(k) + ' type ' + str(types[k]) , 800,800)
        
        rgbcrop = np.uint8(np.stack([rgb_crop, rgb_crop, rgb_crop], axis = -1))
        rgbcrop[b == 255] = [0, 0, 255]

        cv2.imshow('blobs rgb ' + str(k) + ' type ' + str(types[k]), rgbcrop)
        cv2.waitKey(0)
        k += 1

def test_on_pannel(args):

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

    detect(rgb_crop, thermal_crop)

def test_on_full(args):

    im_path = args.im_path
    im_name = os.path.basename(im_path)[:-4]


    # -------------------------- PANNEL DETECTION --------------------------
    img = cv2.imread(im_path)
    detector   = PannelDetector('weights/model_final_thermal.pth')
    pannels = detector.detect(img)
    # -------------------------- PANNEL DETECTION --------------------------
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = get_thermal(im_path)

    # -------------------------- PANNEL CHOPPING --------------------------
    chopper = PannelChopper(None)
    rgb_crops = chopper.efficient_gs_chop(img, pannels)
    thermal_crops = chopper.efficient_gs_chop(thermal, pannels)
    # -------------------------- PANNEL CHOPPING --------------------------

    for i in range(len(rgb_crops)):

        rgb_crop = rgb_crops[i][1]
        thermal_crop = thermal_crops[i][1]

        detect(rgb_crop, thermal_crop)

        cv2.destroyAllWindows()

def draw_rectangle(thermal_img):

    def mouse_callback(event, x, y, flags, params):
        global img

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(coords) > 1:
                coords.append((x,y))
                print(coords)

            else:
                coords.append((x,y))
                print(coords)
                
    img = np.uint8(thermal_img.copy())

    cv2.namedWindow('thermal', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thermal', 800,800)
    cv2.imshow('thermal', img)

    coords = []


    cv2.setMouseCallback('thermal', mouse_callback)

    cv2.imshow("thermal", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return coords

def test_interactive(args):

    thermal = get_thermal(args.im_path)
    rgb_img = cv2.imread(args.im_path, cv2.IMREAD_GRAYSCALE)

    coords  = np.array(draw_rectangle(thermal))

    thermal_crop = crop_cont(thermal, coords)[0]
    cv2.imwrite('thermal_test.png', thermal_crop)
    rgb_crop     = crop_cont(rgb_img, coords)[0]

    print(thermal_crop.shape)
    print(rgb_crop.shape)

    detect(rgb_crop, thermal_crop)


if __name__ == '__main__':

    args = options()

    if args.mode == 'full':
        test_on_full(args)
    elif args.mode == 'pannel':
        test_on_pannel(args)
    elif args.mode == 'interactive':
        test_interactive(args)

    
    