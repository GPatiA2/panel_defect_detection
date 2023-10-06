import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os 
import json

#
# This script takes low res images extracted from the pdf and applies to them an HSV filter to
# approximate the area contained inside the blue square shown in the pdf images
#
# After that, it finds the coordinates of the filtered region and transfers those coordinates to
# the real image
#

scharr_dark_to_bright_X = [[-3,  0, +3],
                    [-10, 0, +10],
                    [-3,  0, +3]]

scharr_bright_to_dark_X = [[+3,  0, -3],
                    [+10, 0, -10],
                    [+3,  0,  -3]]

scharr_dark_to_bright_Y = [[-3, -10, -3],
                           [0,    0,  0],
                           [+3, +10, +3]]

scharr_bright_to_dark_Y = [[+3, +10, +3],
                           [0 ,   0,  0],
                           [-3, -10, -3]]

sobel_dark_to_bright_X = [[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]]

sobel_bright_to_dark_X = [[+1, 0, -1],
                            [+2, 0, -1],
                            [+1, 0, -1]]

sobel_dark_to_bright_Y = [[-1, -2, -1],
                          [0,   0,  0],
                          [+1, +2, +1]]

sobel_bright_to_dark_Y = [[+1, +2, +1],
                          [0,   0,  0],
                          [-1, -2, -1]]

def options():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('pdf_image_path', type=str, help='Path to the images extracted from the pdf')
    parser.add_argument('real_image_path', type=str, help='Path to the directory containing real images')
    parser.add_argument('threshold', type=int, help='Size used to extend to the detected bounding box')
    parser.add_argument('-d','--debug', type=bool, help='Toggle debug execution with pnly 1 image')
    return parser.parse_args()


def mask_bbox(pdf_image, blur_intensity):

    upsampled_img = cv2.resize(pdf_image, (640,512))
    hsv_image = upsampled_img
    hsv_image = cv2.cvtColor(pdf_image.copy(), cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)
    hsv_split = np.concatenate((h,s,v),axis=1)
    # cv2.imshow("Split HSV",hsv_split)
    # cv2.waitKey(0)

    lowsat = 25
    h, s, v = cv2.split(hsv_image)

    # s = cv2.dilate(s, np.ones((3,3)), iterations = 1)
    # s = cv2.GaussianBlur(s, (blur_intensity, blur_intensity), 0)
    # cv2.imshow("Blurred s", s)
    # cv2.waitKey(0)

    # s = cv2.Canny(s,100,200)
    # cv2.imshow("Canny s", s)
    # cv2.waitKey(0)
    # s_sobely = cv2.Sobel(s,0,0,1,ksize=5)

    darken = np.where(s <= lowsat)
    s[darken] = 0
    
    hsv_image = cv2.merge([h,s,v])

    h,s,v = cv2.split(hsv_image)
    lowval = 30
    highlight = np.where(np.logical_and(v > 0, v <= lowval))
    darken    = np.where(v > lowval)
    v[highlight] = 255
    v[darken]    = 0

    hsv_image = cv2.merge([h,s,v])
    hsv_conc = np.concatenate((h,s,v), axis=1)
    # cv2.imshow("hsv_split", hsv_conc)
    # cv2.imshow("hsvmerge", hsv_image)

    # kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype = np.uint8)
    # val_pannel = cv2.dilate(v, kernel, iterations = 3)
    # # val_pannel = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("val pannel", v)
    # cv2.waitKey(0)

    return s
    

def find_rect(pdf_img, masked_img):
    contours,hierarchy = cv2.findContours(masked_img, method = cv2.RETR_LIST, mode = cv2.CHAIN_APPROX_NONE)
    cont_img = pdf_img.copy()
    max_area_cont = max(contours, key=cv2.contourArea)

    mac_image = cv2.drawContours(cont_img, [max_area_cont], -1 ,(0,255,0), 1)
    # cv2.imshow("mac_image", cv2.resize(mac_image, (640,512)))
    # cv2.waitKey(0)

    x,y,w,h = cv2.boundingRect(max_area_cont)

    starting_point = (x+2, y+2)
    end_point = (x+w-2, y+h-2)

    bbox_image = cv2.rectangle(cont_img, starting_point, end_point, (255,0,0), 1)
    # cv2.imshow("boundingRect", cv2.resize(bbox_image, (640,512)))
    # cv2.waitKey(0)

    return (starting_point, end_point), (w,h), (w*h)


def find_inliers(pdf_image):

    print(pdf_image.shape)
    h = pdf_image.shape[0]
    w = pdf_image.shape[1]
    pixels = pdf_image.reshape([-1,1])
    print(pixels.shape)

    kmeans = KMeans(2, max_iter=500).fit(pixels)
    labels = kmeans.labels_

    print(labels.shape)
 
    img_labels = np.array([(0,0,0) if l == 1 else (0,255,0) for l in labels], np.uint8)
    img_labels = img_labels.reshape([h,w,3])

    cv2.imshow("labels", img_labels)
    cv2.waitKey(0)


def find_bbox_coords(pdf_image):
    
    area = 0

    blur_intensity = 3

    only_bbox = mask_bbox(pdf_image, blur_intensity)

    bbox_coords, (w,h), area = find_rect(pdf_image, only_bbox)

    print("BBOX COORDS = ", bbox_coords)

    return bbox_coords, (w,h)



def checkborders(pdf_image, pdf_bbox_coords, size):

    x = pdf_bbox_coords[0][0]
    y = pdf_bbox_coords[0][1]
    w = size[0]
    h = size[1]  

    # cpy = pdf_image.copy()
    # rect = cv2.rectangle(cpy, pdf_bbox_coords[0], (x+w+1,y+h+1), (0,255,0), 2)
    # cv2.imshow("rect", rect)
    # cv2.waitKey(0)

    blank = np.zeros(pdf_image.shape)

    blank[y:y+h+1,x:x+w+1]  = 1

    cpy = pdf_image.copy()
    cpy[blank != 1] = 0

    _ , cpy_s, _ = cv2.split(cpy)

    _, cpy_s = cv2.threshold(cpy_s, 45, 255, cv2.THRESH_BINARY)
    
    dark_to_bright_X = np.array(scharr_dark_to_bright_X)
    bright_to_dark_X = np.array(scharr_bright_to_dark_X)
    
    dark_to_bright_Y = np.array(scharr_dark_to_bright_Y)
    bright_to_dark_Y = np.array(scharr_bright_to_dark_Y)

    dtbX = cv2.filter2D(cpy_s, -1, dark_to_bright_X)
    btdX = cv2.filter2D(cpy_s, -1, bright_to_dark_X)

    dtbY = cv2.filter2D(cpy_s, -1, dark_to_bright_Y)
    btdY = cv2.filter2D(cpy_s, -1, bright_to_dark_Y)

    sobel_X = np.concatenate([dtbX, btdX, dtbX + btdX], axis=1)
    # cv2.imshow("sobelX", sobel_X)
    # cv2.waitKey(0)

    sobel_Y = np.concatenate([dtbY, btdY, dtbY + btdY], axis = 1)
    # cv2.imshow("sobelY", sobel_Y)
    # cv2.waitKey(0)

    sobel_X_Y = sobel_X + sobel_Y
    # cv2.imshow("AND", sobel_X_Y)
    # cv2.waitKey(0)

    laplacian = cv2.Laplacian(cpy_s, 0, 7)
    # cv2.imshow("Laplacian", laplacian)
    # cv2.waitKey(0)
    
    laplacian_blur = cv2.GaussianBlur(laplacian, (3,3), 0)
    # cv2.imshow("Laplacian blur", laplacian_blur)
    # cv2.waitKey(0)

    laplacian_close = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, np.ones((5,5)))
    # cv2.imshow("Laplacian close", laplacian_close)
    # cv2.imshow("original_image", cpy)
    # cv2.waitKey(0)
  

    

def warp_to_upsampled_image(pdf_bbox_coords, pdf_shape, real_shape, th):
    
    empty_image     = np.zeros(pdf_shape, np.uint8)
    bbox_pdf_image  = cv2.rectangle(empty_image, pdf_bbox_coords[0], pdf_bbox_coords[1], (255,0,0), 1)

    resized_to_real = cv2.resize(bbox_pdf_image, (real_shape[1], real_shape[0]), cv2.INTER_AREA) 
    resized_to_real = cv2.cvtColor(resized_to_real, cv2.COLOR_BGR2GRAY)

    cont,hierarchy  = cv2.findContours(resized_to_real, method = cv2.RETR_LIST, mode = cv2.CHAIN_APPROX_NONE)
    x,y,h,w         = cv2.boundingRect(cont[0])

    top_left     = (x-th,y-th)
    bottom_right = (x+h+th,y+w+th)

    return (top_left, bottom_right)

def warp_bbox_to_real(pdf_image_path, real_image_path, th):

    pdf_image = cv2.imread(pdf_image_path)
    real_image = cv2.imread(real_image_path)

    print(real_image.shape)

    pdf_bbox_coords, (w,h) = find_bbox_coords(pdf_image)

    size = (w,h)

    checkborders(pdf_image, pdf_bbox_coords, size)

    real_bbox_coords = warp_to_upsampled_image(pdf_bbox_coords, pdf_image.shape, real_image.shape, th)

    print(real_bbox_coords)

    real_cpy = real_image.copy()
    rect_real = cv2.rectangle(real_cpy, real_bbox_coords[0], real_bbox_coords[1], (0,255,0), 1)
    # cv2.imshow("REAL RECT", rect_real)
    # cv2.imshow("PDF RECT", cv2.resize(cv2.imread(pdf_image_path), (640,512)))
    # cv2.waitKey(0)  
    
    return real_bbox_coords


def debug():

    pdf_image  = cv2.imread("DJI_20220610131900_0034_T_IR(1).JPG")
    real_image = cv2.imread('DJI_20220610131900_0034_T.JPG')

    print(real_image.shape)

    pdf_bbox_coords = find_bbox_coords(pdf_image)

    real_bbox_coords = warp_to_upsampled_image(pdf_bbox_coords, pdf_image.shape, real_image.shape, 5)

    print(real_bbox_coords)
    real_cpy = real_image.copy()

    rect_real = cv2.rectangle(real_cpy, real_bbox_coords[0], real_bbox_coords[1], (0,255,0), 1)
    cv2.imshow("REAL RECT", rect_real)
    cv2.waitKey(0)





if __name__ == '__main__':

    args = options()

    if args.debug:
        debug()
        exit()

    real_image_dir = args.real_image_path

    pdf_diretories = os.listdir(args.pdf_image_path)

    threshold      = args.threshold

    bbox_coords = {}

    os.makedirs("ImagesWithDefects", exist_ok=True)

    saved_def_images = set()

    for dirname in pdf_diretories:
        if os.path.isdir(os.path.join(args.pdf_image_path,dirname)):
            
            real_img_path = os.path.join(real_image_dir, dirname + '.JPG')
            
            if dirname not in bbox_coords.keys():
                bbox_coords[dirname] = {}
            
            if os.path.isfile(real_img_path) and 'T' in real_img_path:
                
                for pdf_image in os.listdir(os.path.join(args.pdf_image_path, dirname)):

                    if '_IR' in pdf_image:
                        pdf_image_path = os.path.join(args.pdf_image_path, dirname, pdf_image)
                        
                        real_bbox_coords = warp_bbox_to_real(pdf_image_path, real_img_path, threshold)

                        op_par_idx = str.find(pdf_image, '(')
                        close_par_idx = str.find(pdf_image, ')')
                        
                        if op_par_idx == -1 or close_par_idx == -1:
                            key = 'n'
                        else:
                            key = pdf_image[op_par_idx + 1:close_par_idx]

                        if 'coords' not in bbox_coords[dirname].keys():
                            bbox_coords[dirname]['coords'] = {}

                        bbox_coords[dirname]['coords'][key] = real_bbox_coords

                        cv2.destroyAllWindows()

                saved_def_images.add(dirname + '.JPG')


    for im in saved_def_images:

        real_img = cv2.imread(os.path.join(real_image_dir, im))
        cv2.imwrite(os.path.join("ImagesWithDefects", im), real_img)


    with open('real_approx_bboxes.json', 'x') as json_file:
        json.dump(bbox_coords, json_file, indent = 1)
