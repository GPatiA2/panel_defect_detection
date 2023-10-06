import torch
import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import nets
import torch.optim.lr_scheduler as lr_scheduler
from argparse import Namespace
import json

class NeuralClassifierLoader():

    def __init__(self, options_file, weights):
        
        with open(options_file, 'r') as f:
            options = json.load(f) 
        
        self.opt = Namespace(options)
        self.weight_file = weights

    def load_classifier(self):
        return PannelClassifier.load_checkpoint(self.weight_file, self.opt)

class PannelClassifier(pl.LightningModule):

    def __init__(self, opt):
        super(PannelClassifier, self).__init__()
        self.opt = opt

        self.num_classes = opt.num_classes
        
        if opt.model == 'MobileNetV3':
            self.model = nets.MobileNetV3Classifier(opt)
        elif opt.model == 'ResNet18':
            self.model = nets.ResNet18(opt)
        elif opt.model == 'MobileNetV2':
            self.model = nets.MobileNetV2Classifier(opt)
        elif opt.model == 'ViTransformer16':
            self.model = nets.ViTransformerClassifier(opt)
        else:
            raise NotImplementedError('[!] Model %s is not implemented.' % opt.model)
        

        if opt.criterion == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError('[!] Criterion [%s] is not implemented.' % opt.criterion)
        
        if opt.init_method == 'xavier':
            for name,param in self.model.named_parameters():
                if name.endswith('weight'):
                     nn.init.xavier_uniform_(param)
        elif self.opt.init_method == 'he':
            for name, param in self.model.named_parameters():
                if name.endswith('weight') and len(param.shape) > 1:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
        else:
            # No initialization
            print(f"[info] Since {opt.init_method} is not a valid initialization method, no initialization is performed.")
            pass

    def load_checkpoint(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

    def transforms(self):
        return self.model.transforms()

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

        # gr_y = torchvision.utils.make_grid(y)
        gr_x = torchvision.utils.make_grid(x)

        # self.logger.experiment.add_image(f'{pref}/y', gr_y, self.current_epoch())
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