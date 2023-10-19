import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
from traditionalClassifierLocalStats import TraditionalClassifier
import cv2
import numpy as np
from copy import deepcopy

class BlobDetectionBCELoss(nn.Module):

    PENALTIES = {
        'minThreshold'       : lambda x : x ** 2 if x < 0 else 0,
        'maxThreshold'       : lambda lth, hth : (lth - hth + 1) ** 2 if lth >= hth else 0, 
        'thresholdStep'      : lambda x : (x+1) ** 2 if x <= 0 else 0,
        'minArea'            : lambda x : (x+1) ** 2 if x <= 0 else 0,
        'maxArea'            : lambda lar, har : (lar - har + 1) ** 2 if lar >= har else 0,
        'minCircularity'     : lambda x : (x+1) ** 2 if x <= 0 else 0,
        'maxCircuiarity'     : lambda lci, hci : (lci - hci + 1) ** 2 if lci >= hci else 0,
        'minConvexity'       : lambda x : (x+1) ** 2 if x <= 0 else 0,
        'maxConvexity'       : lambda lco, hco : (lco - hco + 1) ** 2 if lco >= hco else 0,
        'minInertiaRatio'    : lambda x : (x+1) ** 2 if x <= 0 else 0,
        'maxInertiaRatio'    : lambda lin, hin : (lin - hin + 1) ** 2 if lin >= hin else 0,
        'minDistBetweenBlobs': lambda x : (x+1) ** 2 if x <= 0 else 0
    }

    PARAMS = [
        'minThreshold',
        'maxThreshold', 
        'thresholdStep',
        'minArea',
        'maxArea',
        'minCircularity',
        'maxCircuiarity',
        'minConvexity',
        'maxConvexity',
        'minInertiaRatio',
        'maxInertiaRatio',
        'minDistBetweenBlobs'
    ]

    RESET_PARAMS = {
        'minThreshold' :  0,
        'maxThreshold' : .1,
        'thresholdStep': .1,
        'minArea'      : .1,
        'maxArea'      : .2,
        'minCircularity': .1,
        'maxCircuiarity': .2,
        'minConvexity'  : .1,
        'maxConvexity'  : .2,
        'minInertiaRatio': .1,
        'maxInertiaRatio': .2,
        'minDistBetweenBlobs': .1
    }

    MIN_IDX = [0, 3, 5, 7, 9]
    MAX_IDX = [1, 4, 6, 8, 10]
    LT0     = [2, 11]

    def __init__(self):
        super(BlobDetectionBCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, x ,pred, target):

        classifier = TraditionalClassifier()
        tp = pred.clone().detach()
        tp = np.squeeze(tp.numpy())
        penalty, new_params = self.apply_penalty(tp)
        classifier.set_params(new_params)

        x = np.uint8(x.numpy())
        c_pred = classifier.predict_batch(x)
        c_pred = np.array(c_pred)
        c_pred = torch.from_numpy(c_pred)
        c_pred = c_pred.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        loss = self.criterion(c_pred, target) + penalty
        loss.requires_grad = True

        return loss

    def apply_penalty(self, params):

        new_params = [0 for i in range(12)]
        penalty = 0

        for ind in self.MIN_IDX:
            p = self.PENALTIES[self.PARAMS[ind]](params[ind])
            penalty += p
            new_params[ind] = self.RESET_PARAMS[self.PARAMS[ind]] if p > 0 else params[ind]

        for ind in self.MAX_IDX:
            minval = new_params[self.MIN_IDX[self.MAX_IDX.index(ind)]]
            p = self.PENALTIES[self.PARAMS[ind]](minval, params[ind])
            penalty += p
            new_params[ind] = new_params[self.MIN_IDX[self.MAX_IDX.index(ind)]] + .1 if p > 0 else params[ind]

        for ind in self.LT0:
            p = self.PENALTIES[self.PARAMS[ind]](params[ind])
            penalty += p
            new_params[ind] = self.RESET_PARAMS[self.PARAMS[ind]] if p > 0 else params[ind]

        return penalty, new_params

class TraditionalClassifierModel(pl.LightningModule):

    def __init__(self, opt):
        super(TraditionalClassifierModel, self).__init__()
        self.opt = opt
        self.inw = opt.wdim
        self.inh = opt.hdim
        self.indim = self.inh * self.inw
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.indim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
        )

        self.criterion = BlobDetectionBCELoss()

    def configure_optimizers(self):

        dict = {}

        if self.opt.optimizer == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2), weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == 'radam':
            optim = torch.optim.RAdam(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2), weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2), weight_decay=self.opt.weight_decay)

        dict["optimizer"] = optim
        dict["monitor"]   = "train/loss"

        if self.opt.sched:
            lr_sched = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.001, 
                                                      threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08)
            dict["lr_scheduler"] = lr_sched
        
        return dict
    
    def transforms(self):

        def torch_transform(im):

            cv2_transform = TraditionalClassifier().transforms()
            im = cv2_transform(im)
            im = cv2.resize(im, (self.inw, self.inh))
            im = torch.from_numpy(im)
            im = im.type(torch.FloatTensor)
            return im
        
        return torch_transform

    
    def training_step(self, batch, batch_idx):

        x, y   = batch

        y_pred = self.model(x)
        loss = self.criterion(x, y_pred, y)

        self.log('train/loss', loss.detach())
        self.log('epoch', self.current_epoch)

        return loss
    
    def test(self, batch, batch_idx, pref):

        x, y   = batch
        y_pred = self.model(x)
        loss = self.criterion(x, y_pred, y)

        self.log(f'{pref}/loss', loss.detach())

    def test_step(self, batch, batch_idx):
        self.test(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        self.test(batch, batch_idx, 'val')
         
    def predict_step(self, batch, batch_idx):
        x = batch
        x = self.transforms()(x)
        y_pred = self.model(x)
        return y_pred
