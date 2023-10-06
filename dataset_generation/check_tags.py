import os
import cv2
import json
import argparse
import numpy as np

# 
# This script shows each real photo with the contours of the bad pannels highlighted
#
#


def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('source', type=str, help = 'choose between pdf images or real images', choices = ['pdf', 'real'])
    parser.add_argument('real_images_dir', type=str, help='path to dir to load images from')
    parser.add_argument('defect_json', type=str, help='path to json containing defects in each image')
    parser.add_argument('detection_json_path', type=str, help='path to json file containing real bboxes')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = options()

    with open(args.detection_json_path, 'r') as json_file:
        detections = json.load(json_file)

    with open(args.defect_json,'r') as defect_file:
        defects = json.load(defect_file)

    for image in detections.items():

        if len(image[1] ) > 0 and image[1][0]['bbox'] != "":

            name = image[0] 
            bboxes = [np.array(b['bbox']) for b in image[1]]

            if args.source == 'pdf':
                img = cv2.imread(os.path.join(args.real_images_dir, name, name + '_IR.JPG'))
            else:
                img = cv2.imread(os.path.join(args.real_images_dir, name + '.JPG'))

            img = cv2.resize(img, (640,512))

            # if len(bboxes) == 1:
            #     bboxes = [bboxes]

            print(name)
            print(type(bboxes))
            print(type(bboxes[0]))
            print(type[bboxes[0][0]])
            assert len(defects[name]) == len(image[1])

            # img = cv2.drawContours(img, bboxes, -1, (0,255,0), 1)

            # cv2.imshow(name, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

