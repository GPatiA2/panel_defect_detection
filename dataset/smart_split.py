import os
import json
import cv2
import sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sklearn.model_selection as sk
import argparse

def options():

    parser = argparse.ArgumentParser(description='Split dataset into train and test')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='Fraction of the dataset to be used as test set')
    parser.add_argument('--output_dir', type=str, default='dataset/smart_merged', help='Output directory')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = options()

    # Take the first argument as test fraction
    test_fraction = opt.test_fraction
    out_dir = opt.output_dir

    with open(os.path.join('bad', 'tags.json'), 'r') as f:
        bad_tags = json.load(f)

    with open(os.path.join('healthy', 'tags.json'), 'r') as f2:
        healthy_tags = json.load(f2)

    not_enough_samples_of = ["dirt", "module", "missing", "multicell high", "cell high", "bypass"]

    bad_tags_filtered = list(filter(lambda x: x["label"] not in not_enough_samples_of, bad_tags))

    shuffle(healthy_tags)

    healthy_tags = healthy_tags[:200]

    merged = []
    for it in bad_tags_filtered:
        merged.append(it)

    for it in healthy_tags:
        merged.append(it)

    shuffle(merged)

    X = [it["name"] for it in merged]
    Y = [it["label"] for it in merged]

    print("LEN X "  + str(len(X)))
    print("LEN Y "  + str(len(Y)))

    X_train, X_test, y_train, y_test = sk.train_test_split(X, Y, test_size=test_fraction, shuffle=True, stratify=Y)

    train_ds = [{"name" : X_train[i], "label" : y_train[i]} for i in range(len(X_train))]
    test_ds   = [{"name" : X_test[i], "label" : y_test[i]} for i in range(len(X_test))]

    os.makedirs(os.path.join(out_dir), exist_ok=True)

    with open(os.path.join(out_dir ,'labels.json'), 'w') as f:
        labels_dict = {k : v for k,v in enumerate(bad_tags_filtered)}
        labels_dict = list(set([it[1]['label'] for it in labels_dict.items()]))
        labels_dict.append('healthy')
        json.dump(labels_dict, f, indent=4)

    os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)

    with open(os.path.join(out_dir ,'train','tags.json'), 'w') as f:
        json.dump(train_ds, f, indent=4)

    for di in train_ds:
        path = os.path.join('bad', di["name"]) if 'bad' in di["name"] else os.path.join('healthy', di["name"])
        cv2.imwrite(os.path.join(out_dir + '/train', di["name"]), cv2.imread(path))

    os.makedirs(os.path.join(out_dir , 'test'), exist_ok=True)

    with open(os.path.join(out_dir,'test','tags.json'), 'w') as f:
        json.dump(test_ds, f, indent=4)

    for di in test_ds:
        path = os.path.join('bad', di["name"]) if 'bad' in di["name"] else os.path.join('healthy', di["name"])
        cv2.imwrite(os.path.join(out_dir + '/test', di["name"]), cv2.imread(path))