import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
from traditionalClassifierLocalStats import TraditionalClassifier
import cv2
import numpy as np

class BlobDetectionBCELoss(nn.Module):

    def __init__(self):
        super(BlobDetectionBCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, target):
        classifier = TraditionalClassifier()
        pred2 = np.squeeze(pred.numpy())
        classifier.set_params(pred2)

        pred = classifier.predict(target)
        loss = self.criterion(pred, target)

        return loss

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

        loss = self.criterion(y_pred, y)

        self.log('train/loss', loss.detach())
        self.log('epoch', self.current_epoch)

        return loss
    
    def test(self, batch, batch_idx, pref):

        x, y   = batch
        y_pred = self.model(x)
 
        loss = self.criterion(y_pred, y)

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
