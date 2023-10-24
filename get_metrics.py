import sklearn.metrics as metrics
import sklearn as sk
import numpy as np
import torch
from models.lightingTraditionalModel import TraditionalClassifierModel
import os
from options import OptionParserTraditionalClassifier
from BlobDetectorClassifier import BlobTraditionalClassifier 
from torch.utils.data import DataLoader
import json
import cv2
from matplotlib import pyplot as plt

opt = OptionParserTraditionalClassifier()

ckpt_path = 'results/simpleconv1/logs/lightning_logs/version_0/checkpoints/epoch=331-step=1328.ckpt'

model = TraditionalClassifierModel(opt)

ds = []
with open(os.path.join(opt.labels_file), 'r') as f:
    tags = json.load(f)

for it in tags['test'].items():
    path = os.path.join(opt.images_dir, it[0])
    im = cv2.imread(path)
    ds.append((im, it[1]))


model = TraditionalClassifierModel.load_from_checkpoint(ckpt_path, opt = opt)
model.freeze()

y_true = []
y_pred = []

i = 0
for batch in ds:
    with torch.no_grad():
        x, y = batch
        y_true.append(y)
        pred = model.predict_step(x, i)
        pred = torch.nn.Sigmoid()(pred)
        pred = round(pred.item())
        y_pred.append(pred)
        i += 1

        cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.namedWindow('t_im', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('im', 800, 800)
        cv2.resizeWindow('t_im', 800, 800)
        
        if y == 1 and pred == 0:
            print("FALSE NEGATIVE")
            cv2.imshow('im', x)
            cv2.imshow('t_im', BlobTraditionalClassifier().transforms()(x))
            cv2.waitKey(0)
        if y == 0 and pred == 1:
            print("FALSE POSITIVE")
            cv2.imshow('im', x)
            cv2.imshow('t_im', BlobTraditionalClassifier().transforms()(x))
            cv2.waitKey(0)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))

precission = metrics.precision_score(y_true, y_pred, average='binary')
recall     = metrics.recall_score(y_true, y_pred, average='binary')
f1         = metrics.f1_score(y_true, y_pred, average='binary')
ac         = metrics.accuracy_score(y_true, y_pred)

print("True positives: ", tp)
print("True negatives: ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)

print("Precission: ", precission)
print("Recall: ", recall)
print("F1: ", f1)
print("Accuracy: ", ac)

conf_mat   = metrics.confusion_matrix(y_true, y_pred)
metrics.ConfusionMatrixDisplay(conf_mat).plot(xticks_rotation='vertical')
plt.show()



