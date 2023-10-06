import os
import cv2
import json
import argparse
import numpy as np

#
# This script takes pannel detections over a real image and transfers the coordinates of the contours 
# detected to a given resolution image.
#
# The coordinates are then saved to a json file
#

def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('real_images_dir', type=str, help = 'path to real images')
    parser.add_argument('detection_json_path', type=str, help='path to json file containing detection data')
    parser.add_argument('output_path', type=str, help='path to output_json')
    parser.add_argument('-s', '--show', type=bool, help='show detections')
    return parser.parse_args()

maskrcnn_res = (1024, 767)

def normalize(coords, res):
    normalized_coords = []
    for p in coords[0]:
        point = p[0]
        px = point[0] / res[0]
        py = point[1] / res[1]
        normalized_coords.append([px,py])

    return normalized_coords

def transfer_to_real(coords, shape):
    real_coords = []
    for p in coords:

        px = p[0] * shape[1]
        py = p[1] * shape[0]
        real_coords.append([px,py])

    return real_coords

if __name__ == '__main__':

    args = options()

    json_coords = {}

    with open(args.detection_json_path) as f:
        data = json.load(f)

    for im_data in data.items():

        im_name = im_data[0]
        image   = cv2.imread(os.path.join('ImagesWithDefects', im_name))
        im_shape = image.shape

        annotations = im_data[1]['annotations']

        json_coords[im_name] = []

        for detection in annotations:

            poly = np.array(detection['segmentation'])

            poly = normalize(poly, maskrcnn_res)
            
            poly = transfer_to_real(poly, im_shape)

            poly = np.array(poly, np.int32)

            json_coords[im_name].append(poly.tolist())

            image = cv2.drawContours(image, [poly], 0, (0,255,0), 1)


        if args.show :
            image = cv2.resize(image, (im_shape[1], im_shape[0]))
            cv2.imshow(im_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    with open(args.output_path, 'x') as dest_file:
        json.dump(json_coords, dest_file, indent= 1)



