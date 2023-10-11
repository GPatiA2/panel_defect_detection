from gc import callbacks
from torch.utils import data
from options import OptionParserClassifier
import cv2
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from model import PannelClassifier
from data import PannelClassificationDataset
from torch.utils.data import DataLoader
import os
import json

opt = OptionParserClassifier()

results_dir  = os.path.join('results',opt.results_dir)
ckpt_dir = os.path.join(results_dir, 'checkpoints')
log_dir  = os.path.join(results_dir, 'logs')

os.makedirs(results_dir, exist_ok=False)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("[info] Checkpoint directory: ", ckpt_dir)
print("[info] Log directory: ", log_dir)
_ = input("[info] Press enter to continue")

with open(os.path.join(results_dir, 'opt.json'), 'w') as f:
    json.dump(vars(opt), f, indent = 4) 

callback_list = []
ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, monitor='val/loss',
                                              save_top_k=4, mode='min')
callbacks.append(ckpt_callback)

if opt.sched:
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_callback)

tensorboard_log = pl_loggers.TensorBoardLogger(save_dir=log_dir, name='lightning_logs')
tensorboard_log.experiment.add_text('opt', str(vars(opt)), 0)

model = PannelClassifier(opt)

dataset = PannelClassificationDataset(opt, model.transforms())

with open(os.path.join(results_dir, 'labels.json'), 'w') as f:
    labels_corrected, tags_corrected = dataset.get_tag_assignment()
    classes = { idx : (val, labels_corrected[idx])  for idx, val in enumerate(tags_corrected)}
    json.dump(classes, f, indent = 4)

train_ds, val_ds = dataset.split()
train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
val_dl   = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

_ = input("[info] Press enter to start training")

trainer = pl.Trainer(devices = [0], max_epochs=opt.epochs, callbacks=callback_list, logger = tensorboard_log ,accelerator='gpu', log_every_n_steps=3)
trainer.fit(model = model, train_dataloaders=train_dl, val_dataloaders=val_dl)


