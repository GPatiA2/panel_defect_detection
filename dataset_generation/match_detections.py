import cv2
import numpy as np
import json
import argparse
import os

#
# This scripts shows each pdf photo upsampled to the resolution of the real ones
# for the user to choose the detections that match the highlighted ones in the
# pdf images by double-clicking inside the desired contour and pressing space
#
# To skip the image just press any other key than space
# 


def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path_detections', type=str, help = 'path to json detections')
    parser.add_argument('path_approx_bboxes', type=str, help = 'path to json containing approximated bboxes')
    parser.add_argument('path_deffects_per_photo', type=str, help = 'path to json containing the deffect in each photo')
    parser.add_argument('output_file', type=str, help = 'path to output json file')
    return parser.parse_args()

def search_number(image_deffects, number):
    i = 0
    print(image_deffects)
    print(number)
    print("___")
    while i < len(image_deffects) and i != int(number):
        i += 1

    if i == len(image_deffects):
        print(len(image_deffects))
        print(i)
        assert False , "Image number not found in deffect list"
    else:
        return image_deffects[i]["defect"]
    
def inside(bbox, approx):
    ret = True
    for point in bbox:
        ret &= approx[0][0] <= point[0] and point[0] <= approx[1][0]
        ret &= approx[0][1] <= point[1] and point[1] <= approx[1][1]

    return ret

def getMousePos(event,x,y,flags,params):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x,y

def select_bbox(image_name, number, possible_bboxes):

    full_name = image_name + '_IR.JPG' if number == 0 else image_name + '_IR(' + str(number) + ').JPG'

    image = cv2.imread(os.path.join('datadron_pdf', image_name, full_name))
    image = cv2.resize(image, (640,512))
    possible_bboxes_2 = [np.array(bbox) for bbox in possible_bboxes]
    image = cv2.drawContours(image, possible_bboxes_2, -1, (0,255,0), 1)

    cv2.imshow('Pannels', image)
    cv2.setMouseCallback('Pannels', getMousePos)
    k = cv2.waitKey(0)
    if k == ord(' '):
        pass
    else:
        return None

    i = 0
    found = -1
    bbox = None
    while i < len(possible_bboxes_2) and found != 1:
        found  = cv2.pointPolygonTest(possible_bboxes_2[i], (mouseX,mouseY), False)
        if found == 1:
            bbox = possible_bboxes_2[i] 
        i += 1 

    return bbox.tolist()


if __name__ == '__main__':

    args = options()

    with open(args.path_detections) as f:
        detections = json.load(f)

    with open(args.path_approx_bboxes) as f2:
        approx_bboxes = json.load(f2)

    with open(args.path_deffects_per_photo) as f3:
        deffects = json.load(f3)

    with open(args.output_file, 'x') as out_file:    
        bad_pannels = {}

        for image in approx_bboxes.items():

            image_name = image[0]
            bad_pannels[image_name] = []

            if len(image[1]) != 0:
                
                for approx_area in image[1]['coords'].items():

                    number          = 0 if approx_area[0] == "n" else approx_area[0]
                    approx_bbox     = approx_area[1]
                    deffect         = search_number(deffects[image_name], number)
                    possible_bboxes = [bbox for bbox in detections[image_name + '.JPG'] if inside(bbox, approx_bbox)] 
                
                    bbox = []
                    if len(possible_bboxes) == 1:
                        bbox = possible_bboxes[0]
                    else:
                        bbox = select_bbox(image_name, number, possible_bboxes)

                    bad_pannels[image_name].append({
                        "bbox"   : bbox,
                        "defect" : deffect
                    })

        json.dump(bad_pannels, out_file, indent = 1)