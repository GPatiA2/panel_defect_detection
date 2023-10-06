from __future__ import annotations
import os, os.path
import numpy as np
import cv2

##Facebook Detectron2 utilities
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

def get_color(j, on_gpu=None):
    color_idx = j * 5 % len(COLORS)
    return COLORS[color_idx]

class PannelDetector():
    
    thing_classes = ["panel"]

    def __init__(self, weights_file, model = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'):
        
        self.model = model
        self.cfg = self.config_detectron()
        self.weights_file = weights_file
        self.cfg.MODEL.WEIGHTS = weights_file
        
        self.predictor = DefaultPredictor(self.cfg)

    def rectify(self, image):
        mtx = np.array([[3318.711547549526, 0.0, 2012.5166933305093], [0.0, 3318.8343662555144, 1498.7763399576656], [0.0, 0.0, 1.0]])
        dist = np.array([0.12448971199214658, -0.2087535646698689, 0.08327957438286628,
                0.0004888305392246547, -0.00010785065050375144, -0.00265596810217254, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dist = [0.12448971199214658, -0.2087535646698689, 0.08327957438286628,
                0.0004888305392246547, -0.00010785065050375144, -0.00265596810217254, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        dist = np.array([dist[0], dist[1], dist[2], dist[3], dist[4], dist[6], dist[7]])
        h,  w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        #map_x, map_y= cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        #dst = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
    
    def bounding_box(self, points):
        x_coordinates, y_coordinates = [], []

        for p in points:
            p=p[0]
            x_coordinates.append(p[0])
            y_coordinates.append(p[1])

        return [float(min(x_coordinates)), float(min(y_coordinates)), float(max(x_coordinates))-float(min(x_coordinates)), float(max(y_coordinates))-float(min(y_coordinates))]
          
        

    def config_detectron(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model))
        cfg.DATASETS.TRAIN = ('train',)
        cfg.DATASETS.TEST = ('val',)   
        cfg.TEST.EVAL_PERIOD = 150
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.000125
        cfg.SOLVER.MAX_ITER = 500
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.RPN.IOU_THRESHOLDS = [0.7, 1]
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.8
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000  # originally 1000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000  # originally 1000
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8

        #nms
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.7
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        return cfg

    def detect(self, image):
        outputs = self.predictor(image) 
        detections = []

        for i, box in enumerate(outputs['instances'].pred_masks):
            #pred_boxes = outputs['instances'].pred_boxes.tensor[i]
            
            if outputs['instances'].scores[i] < 0.9:
                # print(outputs['instances'].scores[i])
                continue
            
            pred_mask = box
            
            m = pred_mask.size()[0]
            n = pred_mask.size()[1]

            mascara = np.zeros([m,n])
            mascara = pred_mask.cpu().detach().numpy()      

            mascara = 255*mascara
            mascara = mascara.astype(np.uint8)

            cnts, hierarchy = cv2.findContours((mascara).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
            epsilon = 0.01*cv2.arcLength(cnts[0],True)
            approx = cv2.approxPolyDP(cnts[0],epsilon,True).astype(float)
            
            bbox= self.bounding_box((approx).reshape(-1,1,2))
            color = get_color(i)

            obj = {
                "bbox" : bbox,
                "segmentation" : approx.tolist()
            }

            detections.append(obj)

        return detections
    