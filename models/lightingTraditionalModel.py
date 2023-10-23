import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
from traditionalClassifierLocalStats import TraditionalClassifier
import cv2
import numpy as np
from copy import deepcopy
import torchvision


class TraditionalClassifierModel(pl.LightningModule):

    def __init__(self, opt):
        super(TraditionalClassifierModel, self).__init__()
        self.opt = opt
        self.inw = 50
        self.inh = 35
        self.indim = self.inh * self.inw
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride = 2, padding = 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 6, kernel_size=3, stride = 2, padding = 1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=1, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride = 2, padding = 1),
            nn.Flatten(),
            nn.Linear(35, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()

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
            im = torch.unsqueeze(im, 0)
            return im
        
        return torch_transform

    
    def training_step(self, batch, batch_idx):

        x, y   = batch
        y_pred = self.model(x)
        y_pred = torch.squeeze(y_pred)
        y_pred = y_pred.type(torch.FloatTensor)
        y      = y.type(torch.FloatTensor  )
        loss = self.criterion(y_pred, y)

        self.log('train/loss', loss.detach())
        self.log('epoch', self.current_epoch)

        return loss
    
    def test(self, batch, batch_idx, pref):

        x, y   = batch
        y_pred = self.model(x)
        y_pred = torch.squeeze(y_pred)
        y_pred = y_pred.type(torch.FloatTensor)
        y      = y.type(torch.FloatTensor  )
        loss = self.criterion(y_pred, y)

        self.log(f'{pref}/loss', loss.detach())

        gr_x = torchvision.utils.make_grid(x)
        self.logger.experiment.add_image(f'{pref}/x', gr_x, self.current_epoch)

    def test_step(self, batch, batch_idx):
        self.test(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        self.test(batch, batch_idx, 'val')
         
    def predict_step(self, batch, batch_idx):
        x = batch
        x = self.transforms()(x)
        y_pred = self.model(x)
        return y_pred
