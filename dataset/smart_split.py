import os
import json
import cv2
import sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import sklearn.model_selection as sk

# Take the first argument as test fraction
test_fraction = float(sys.argv[1])

with open(os.path.join('dataset', 'bad', 'tags.json'), 'r') as f:
    bad_tags = json.load(f)

with open(os.path.join('dataset','healthy', 'tags.json'), 'r') as f2:
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

os.makedirs('dataset/smart_merged', exist_ok=True)

os.makedirs('dataset/smart_merged/train', exist_ok=True)
with open(os.path.join('dataset','smart_merged','train','tags.json'), 'w') as f:
    json.dump(train_ds, f, indent=4)

for di in train_ds:
    path = os.path.join('dataset', 'bad', di["name"]) if 'bad' in di["name"] else os.path.join('dataset', 'healthy', di["name"])
    cv2.imwrite(os.path.join('dataset/smart_merged/train', di["name"]), cv2.imread(path))




os.makedirs('dataset/smart_merged/test', exist_ok=True)
with open(os.path.join('dataset','smart_merged','test','tags.json'), 'w') as f:
    json.dump(test_ds, f, indent=4)

for di in test_ds:
    path = os.path.join('dataset', 'bad', di["name"]) if 'bad' in di["name"] else os.path.join('dataset', 'healthy', di["name"])
    cv2.imwrite(os.path.join('dataset/smart_merged/test', di["name"]), cv2.imread(path))