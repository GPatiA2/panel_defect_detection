import cv2
import numpy as np
import os 
import argparse
import json
from thermal import Thermal 

class DefectClassifier():

    def __init__(self, params):

        self.rolling_guidance_iters = params['rolling_iters']
        self.max_dilation_iters     = params['max_dilation_iters']

    def 