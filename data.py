import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
import cv2
from random import shuffle
from math import ceil, floor
import json
from random import random
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import hflip, vflip
import cv2
import sklearn.model_selection as sk
import torchvision.models


class BinaryPannelClassificationDataset(Dataset):

    def __init__(self, opt):

        self.opt = opt
        self.images_dir = opt.images_dir
        self.labels_file = opt.labels_file
        self.training_frac = opt.training_frac

        self.hdim = opt.hdim
        self.wdim = opt.wdim

        self.dataset = []

        with open(self.labels_file, 'r') as f:
            labels = json.load(f)

        for it in labels['train'].items():
            im = cv2.imread(os.path.join(self.images_dir, it[0]))
            im = cv2.resize(im, (self.wdim, self.hdim))
            label = it[1]
            self.dataset.append((im, label))

        print("[info] Loaded dataset ")
        print("[info] Number of samples: ", len(self.dataset))

    def __len__(self):
        return len(self.dataset)
    
    def split(self):

        shuffle(self.dataset)

        x = [it[0] for it in self.dataset]
        y = [it[1] for it in self.dataset]

        X_train, X_val, y_train, y_val = sk.train_test_split(x, y, test_size= 1 - self.training_frac, shuffle=True, stratify=y)

        train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        val_set   = [(X_val[i], y_val[i]) for i in range(len(X_val))]

        print("[info] Split dataset ")
        print("[info] Training set: ", len(train_set))
        print("[info] Validation set: ", len(val_set))

        return SimpleDataset(train_set), SimpleDataset(val_set)
    
class SimpleDataset(Dataset):

    def __init__(self, data):

        self.dataset = data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def apply(self, function):
        self.dataset = [function(it) for it in self.dataset]



class PannelClassificationDataset(Dataset):

    def __init__(self, opt, transforms):

        self.opt        = opt

        self.dataset_dir = opt.dataset_dir
        self.tags_dir = opt.dataset_dir
        self.dataset_dir = os.path.join(self.dataset_dir, 'train')


        with open(os.path.join(self.tags_dir, 'labels.json'), 'r') as f:
            # Defects es el json que contiene los defectos que se van a detectar
            # Se debe obtener al generar el dataset
            self.defects = json.load(f)

        self.training_frac = opt.training_frac
        self.validation_frac = opt.validation_frac 

        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers

        self.pretrained = opt.pretrained

        with open(os.path.join(self.dataset_dir, 'tags.json'), 'r') as f:
            pannels_data = json.load(f) 

        self.dataset = []

        self.labels_corrected = []
        self.tags_corrected   = []

        for it in pannels_data:
            im    = cv2.imread(os.path.join(self.dataset_dir, it['name']))
            im    = cv2.imread(os.path.join(self.dataset_dir, it['name']))
            im    = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t_im  = torchvision.transforms.ToTensor()(im)

            t_im = transforms(t_im)

            label = self.defects.index(it['label'])

            if label not in self.labels_corrected:
                self.labels_corrected.append(label)

            if it['label'] not in self.tags_corrected:
                self.tags_corrected.append(it['label'])

            self.dataset.append((t_im, self.labels_corrected.index(label)))

        print("[info] Loaded dataset ")
        print("[info] Number of samples: ", len(self.dataset))
        print("[info] Number of classes: ", len(self.labels_corrected))

    def __len__(self):
        return len(self.dataset)
    
    def get_tag_assignment(self):
        return (self.labels_corrected, self.tags_corrected)

    def split(self):

        shuffle(self.dataset)

        x = [it[0] for it in self.dataset]
        y = [it[1] for it in self.dataset]

        X_train, X_val, y_train, y_val = sk.train_test_split(x, y, test_size= 1 - self.training_frac, shuffle=True, stratify=y)

        train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        val_set   = [(X_val[i], y_val[i]) for i in range(len(X_val))]

        print("[info] Split dataset ")
        print("[info] Training set: ", len(train_set))
        print("[info] Validation set: ", len(val_set))

        return TrainingDataset(train_set, self.opt), TrainingDataset(val_set, self.opt)
    
class TestDataset(Dataset):

    def __init__(self, opt):

        self.opt = opt

        self.dataset = []

        with open(os.path.join(self.opt.dataset_dir, 'tags.json'), 'r') as f:
            pannels_data = json.load(f)

        for it in pannels_data:
            im    = cv2.imread(os.path.join(self.opt.dataset_dir, it['name']))
            im    = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t_im  = cv2.resize(im, (35, 50))
            t_im  = torchvision.transforms.ToTensor()(t_im)
            t_im  = torchvision.transforms.Grayscale(num_output_channels=3)(t_im)
            label = PannelClassificationDataset.defects[it['label']]
            self.dataset.append((t_im, label))

        print("[info] Loaded dataset ")
        print("Number of samples: ", len(self.dataset))

    def __len__(self):
        return len(self.dataset)   
    

class TrainingDataset(Dataset):

    def __init__(self, data, opt):
        
        self.dataset = data

        self.horiz_flip_chance = opt.hflip_chance
        self.verti_flip_chance = opt.vflip_chance

        self.saturation_chance = opt.sat_chance
        self.saturation_factor = opt.sat_factor
        self.sat_transform     = ColorJitter(saturation=self.saturation_factor)

        self.brightness_chance = opt.bri_chance
        self.brightness_factor = opt.bri_factor
        self.bri_transform     = ColorJitter(brightness=self.brightness_factor)

        self.contrast_chance   = opt.con_chance
        self.contrast_factor   = opt.con_factor
        self.con_transform     = ColorJitter(contrast=self.contrast_factor)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        img = self.dataset[index][0]
        label = self.dataset[index][1]

        sat_random = random()
        bri_random = random()
        con_random = random()
        vflip_random = random()
        hflip_random = random()

        if sat_random < self.saturation_chance:
            img = self.sat_transform(img)
        
        if bri_random < self.brightness_chance:
            img = self.bri_transform(img)

        if con_random < self.contrast_chance:
            img = self.con_transform(img)

        if vflip_random < self.verti_flip_chance:
            img = vflip(img)

        if hflip_random < self.horiz_flip_chance:
            img = hflip(img)

        return (img, label)


    





