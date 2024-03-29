import cv2
import numpy as np
import matplotlib.patches as patches
import os
import pylab
import torch
from thermal import Thermal
import matplotlib.colors as mpltcolors

class ReportDataGenerator():

    def __init__(self, pannel_detector, defect_detector, defect_classifier, chopper, imgs_out_dir, test = False):

        self.test = test

        self.out_dir = imgs_out_dir

        self.classes = defect_classifier.get_classes()
        print(self.classes)

        self.plt_colors = [mpltcolors.to_rgb(c) for c in list(mpltcolors.TABLEAU_COLORS)][:len(self.classes)]
        self.cv2_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in self.plt_colors]
        self.rects = [patches.Patch(color=self.plt_colors[i], 
                                label=self.classes[i]) for i in range(len(self.cv2_colors))]

        
        self.panel_detector   = pannel_detector
        self.chopper    = chopper
        self.defect_detector = defect_detector
        self.defect_classifier = defect_classifier

    def labels_as_colors(self, lb):

        lb_c = [bin(idx + 1) for idx, _ in enumerate(lb)]
        lb_c = [lb_c[idx][2:] for idx, _ in enumerate(lb_c)]
        
        max_len = max(lb_c, key = lambda x : len(x))
        lb_color_corrected = []
        for it in lb_c:
            while len(it) < len(max_len):
                it = '0' + it
            lb_color_corrected.append(it)

        lb_c = [[int(it[i]) for i in range(len(it))] for it in lb_color_corrected]
        return lb_c

    def read_images(self,path):
        
        images = []

        thermal = Thermal(
            dirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libdirp.so',
            dirp_sub_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_dirp.so',
            iirp_filename='plugins/dji_thermal_sdk_v1.1_20211029/linux/release_x64/libv_iirp.so',
            exif_filename='plugins/exiftool-12.35.exe',
            dtype=np.int16,
        )
        
        k = len(os.listdir(path)) if not self.test else 10

        i = 0
        for f in os.listdir(path):
            if f.endswith('_T.JPG') and i < k:

                im   = cv2.imread(os.path.join(path,f))
                temp = thermal.parse_dirp2(os.path.join(path,f))
                images.append((f, im, temp))
                i += 1

        return images
    
    def save_legend_image(self):

        figure = pylab.figure()
        figlegend = pylab.figure(figsize=(3,2))
        ax = figure.add_subplot(111)
        lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))
        figlegend.legend(handles=self.rects, loc='center')
        figure.show()
        figlegend.show()
        path = os.path.join(self.out_dir, 'legend.png')
        figlegend.savefig(path)

        return path

    def generate_report_data(self, path_to_images, show_detections = False, show_crops = False):

        report_images = []
        defect_count  = {}

        images = self.read_images(path_to_images)

        os.makedirs(self.out_dir, exist_ok=True)

        total_detections = [] 

        for im in images:

            defect_in_image = { k : 0  for k in self.classes}

            im2 = im[1].copy()

            detections = self.panel_detector.detect(im[1])

            conts = [np.array(d['segmentation'], dtype=np.int32) for d in detections]
            
            if show_detections:
                print(len(conts))
                print(conts[0].dtype)
                print(conts[0])

                im3 = im2.copy()
                im3 = cv2.drawContours(im3, conts, -1, (0,255,0), 2)
                cv2.imshow("im3", im3)
                cv2.waitKey(0)

            rgbcrops = self.chopper.efficient_chop(im[1], detections)
            thermalcrops = self.chopper.efficient_gs_chop(im[2], detections)
            total_detections.append((im[0], im[1], im[2], detections, rgbcrops, thermalcrops))

        rgbcrops = [np.stack(d[5]) for d in total_detections]
        median_img = np.median(rgbcrops, axis=0)

        for det in total_detections:
            detections = det[3]
            rgb_temp = det[1]
            themal = det[2]
            name = det[0]

            rgb_crops   = det[4]
            thermal_crops = det[5] 

            im2 = rgb_temp.copy()

            if show_crops:
                for c in rgb_crops:
                    cv2.imshow("crop", c[1])
                    cv2.waitKey(0)

            i = 0
            for k in range(len(rgb_crops)):
                c = rgb_crops[k]
                # Defect detection and classification
                pred = self.defect_detector.predict_step(np.uint8(rgb_crops[k][1]))

                if pred == 0:
                    continue
            
                else:
                    def_type = self.defect_classifier.classify(np.uint8(thermal_crops[k][1]))
                    if def_type != "NO DEFECT":
                        def_idx  = self.classes.index(def_type)
                        
                        im2 = cv2.drawContours(im2, c[0], -1, self.cv2_colors[def_idx], 2)
                        i += 1

                        if self.classes[def_idx] not in defect_count.keys():
                            defect_count[self.classes[def_idx]] = 0

                        defect_count[self.classes[def_idx]] += 1

                        defect_in_image[self.classes[def_idx]] += 1
                    else: 
                        im2 = cv2.drawContours(im2, c[0], -1, (255,0,255), 2)

            pth = os.path.join(self.out_dir, name)

            cv2.imwrite(pth, im2)

            report_images.append((pth, defect_in_image))

        return report_images, defect_count
