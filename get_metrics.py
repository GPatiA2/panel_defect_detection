import sklearn.metrics as metrics
import sklearn as sk
import numpy as np
import torch
from models.lightingTraditionalModel import TraditionalClassifierModel
import os
from options import OptionParserTraditionalClassifier
from data import BinaryPannelClassificationDataset   
from torch.utils.data import DataLoader
import json
import cv2

opt = OptionParserTraditionalClassifier()

model = TraditionalClassifierModel(opt)

ds = []
with open(os.path.join(opt.labels_file), 'r') as f:
    tags = json.load(f)

for it in tags['test']:
    path = os.path.join(opt.images_dir, it[0])
    im = cv2.imread(path)
    im_t = model.transforms()(im)
    ds.append((im_t, it[1]))


data_loader = DataLoader(ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

model.load_checkpoint(opt.ckpt_path)
model.eval()

y_true = []
y_pred = []

for batch in data_loader:
    with torch.no_grad():

        x, y = batch
        y_true.append(y)
        y_pred.append(np.argmax(torch.nn.Softmax(model.predict(x))))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

precission = metrics.precision_score(y_true, y_pred, average='micro')
ac         = metrics.accuracy_score(y_true, y_pred)
recall     = metrics.recall_score(y_true, y_pred, average='micro')
f1         = metrics.f1_score(y_true, y_pred, average='micro')

print("Precission: ", precission)
print("Recall: ", recall)
print("F1: ", f1)
print("Accuracy: ", ac)

conf_mat   = metrics.confusion_matrix(y_true, y_pred)
sk.metrics.plot_confusion_matrix(conf_mat, normalize=True)



