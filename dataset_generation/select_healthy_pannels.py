import os
import cv2
import json
import numpy as np
import argparse

# 925
#
# This script 
#
#
#
#

maskrcnn_res = (1024, 767)
real_res     = (640, 512)

def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('real_images_dir', type=str, help='path to dir to load images from')
    parser.add_argument('detections_json_path', type=str, help='path to json containing detections in each image')
    parser.add_argument('defects_json_path', type=str, help='path to json file containing contours of bad pannels')
    return parser.parse_args()

def getMousePos(event,x,y,flags,params):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x,y

def normalize(coords, res):
    normalized_coords = []
    for p in coords:
        point = p[0]
        px = point[0] / res[0]
        py = point[1] / res[1]
        normalized_coords.append([px,py])

    return normalized_coords

def transfer_to_real(coords, shape):
    real_coords = []
    for p in coords:

        px = p[0] * shape[0]
        py = p[1] * shape[1]
        real_coords.append([px,py])

    return real_coords

def bad_pannel(defects, coords):

    for d in defects:
        bbox = d["bbox"]
        if bbox == '':
            print("Bbox not valid, go to another photo")
            return False

        elif cv2.pointPolygonTest(np.int32(np.array(bbox)), coords, False) == 1:
            print("BAD PANNEL SELECTED AS HEALTHY")
            return True
        
        print(f"Coords {coords} not in bbox {bbox}")

    return False


if __name__ == '__main__':

    args = options()

    real_images = os.listdir(args.real_images_dir)

    with open(args.detections_json_path, 'r') as det_json:
        detections = json.load(det_json)

    with open(args.defects_json_path, 'r') as defects_json:
        defects_json = json.load(defects_json)

    healthy_pannels = {}

    for it in defects_json.items():

        image_name   = it[0]
        defects      = it[1]

        if len(defects) > 0:

            print(f"Name = {image_name}")
            
            defects_bbox = [d["bbox"] for d in defects]

            real_detections = [d["segmentation"] for d in detections[image_name + '.JPG']["annotations"]]
            det = []
            for d in real_detections:
                det.append(d[0])

            normalized = [normalize(p, maskrcnn_res) for p in det]
            scaled     = [transfer_to_real(p, real_res) for p in normalized]

            healthy_detections = [np.array(p, np.int32) for p in scaled]

            image = cv2.imread(os.path.join(args.real_images_dir, image_name + '.JPG'))
            image = cv2.drawContours(image, healthy_detections, -1, (0,255,0), 1)

            selected_bboxes = [] 

            k = None

            while k != ord('b') : 
                cv2.imshow('Pannels', image)
                cv2.setMouseCallback('Pannels', getMousePos)
                k = cv2.waitKey(0)
                if k == ord(' '):
                    
                    healthy_bbox = False
                    i = 0
                    while (i < len(healthy_detections)) and not healthy_bbox:
                        if cv2.pointPolygonTest(healthy_detections[i], (mouseX,mouseY), False) == 1 and not bad_pannel(defects, (mouseX, mouseY)):
                            healthy_bbox = True
                            selected_bboxes.append(healthy_detections[i].tolist())
                            break
                        i += 1
                
                elif k == ord('b'):
                    break
                else:
                    pass

            if len(selected_bboxes) > 0:
                healthy_pannels[image_name] = selected_bboxes
                print("_________________________")

    
    with open('healthy_detections.json', 'w') as healthy_json:
        json.dump(healthy_pannels, healthy_json, indent = 1)






