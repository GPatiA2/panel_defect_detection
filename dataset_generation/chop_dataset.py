import json
import cv2
import argparse
import os
from tqdm import tqdm
import numpy as np

#
# This script takes a json file describing detections of either healty or bad pannels
# and crops real images to extract the bounding rectangle of the contour containing
# the detection
#

def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('json_path', type=str, help='Path to the json file containing tags')
    parser.add_argument('imgs_path', type=str, help='Path to the images dataset')
    parser.add_argument('dest_dir', type=str, help='Name of the dataset', default='dataset')
    parser.add_argument('subdir', type=str, help='type of pannel to be chopped (either healthy or bad). Conditions json formnat when reading from file', choices = ['healthy', 'bad'])
    return parser.parse_args()

def chop_bad_pannels(imgs_path, data):

    samples = []

    for s in tqdm(data.items()):

        if len(s[1]) > 0 and os.path.isfile(os.path.join(imgs_path, s[0]+'.JPG')):

            name_img = s[0]
            img = cv2.imread(os.path.join(imgs_path, name_img + '.JPG')) 

            i = 0
            
            for d in s[1]:

                if len(d["bbox"]) > 0:

                    coords = d["bbox"] 
                    coords = [np.int32(np.array(p)) for p in coords]

                    x,y,w,h = cv2.boundingRect(np.array(coords))

                    coords = np.int32([coords])

                    mask_cont = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                    mask_cont = cv2.fillPoly(mask_cont, pts = coords, color = (255,255,255))

                    only_cont   = cv2.bitwise_and(mask_cont, img)


                    masked_chop = only_cont[y:y+h, x:x+w ] 

                    chop_sample = {
                        'name'  : name_img + '_bad_' + str(i) + '.JPG',
                        'chop'  : masked_chop,
                        'label' : d["defect"] 
                    }

                    samples.append(chop_sample)
                    i += 1

        else:

            print(f"There is no useful defect in {s[0]} or there is not a real image with that name")

    return samples

def chop_healty_pannels(imgs_path, data):

    samples = []
    for s in tqdm(data.items()):

        name_img = s[0]
        img = cv2.imread(os.path.join(imgs_path, name_img + '.JPG')) 

        i = 0
        
        for d in s[1]:
 
                coords = [np.int32(np.array(p)) for p in d]
                print(name_img)
                print(coords)
                x,y,w,h = cv2.boundingRect(np.array(coords))

                coords = np.int32([coords])

                mask_cont = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                mask_cont = cv2.fillPoly(mask_cont, pts = coords, color = (255,255,255))

                only_cont   = cv2.bitwise_and(mask_cont, img)


                masked_chop = only_cont[y:y+h, x:x+w] 

                chop_sample = {
                    'name'  : name_img + '_healthy_' + str(i) + '.JPG',
                    'chop'  : masked_chop,
                    'label' : 'healthy' 
                }

                samples.append(chop_sample)
                i += 1

    return samples

if __name__ == "__main__":

    args = options()

    with open(args.json_path) as f:
        data = json.load(f)

    os.makedirs(args.dest_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dest_dir, args.subdir), exist_ok=True)

    if args.subdir == 'bad':
        panel_samples = chop_bad_pannels(args.imgs_path, data)
    else:
        panel_samples = chop_healty_pannels(args.imgs_path, data)

    json_data = []

    for s in tqdm(panel_samples):
        
        json_data.append({
            'name'  : s['name'],
            'label' : s['label']
        })

        cv2.imwrite(os.path.join(args.dest_dir,args.subdir,s['name']), s['chop'])

    with open(os.path.join(args.dest_dir, args.subdir, 'tags.json'), 'w') as jfile:
        json.dump(json_data, jfile, indent = 4)

    







    