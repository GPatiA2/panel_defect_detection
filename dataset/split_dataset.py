import os
import json
import cv2
import sys
from random import shuffle
from math import ceil

# Take the first argument as test fraction
test_fraction = float(sys.argv[1])

with open(os.path.join('bad', 'tags.json'), 'r') as f:
    bad_tags = json.load(f)

with open(os.path.join('healthy', 'tags.json'), 'r') as f2:
    healthy_tags = json.load(f2)

merged_ds = {}

for it in bad_tags:
    merged_ds[it["name"]] = it["label"]

for it in healthy_tags:
    merged_ds[it["name"]] = it["label"]

os.makedirs('merged', exist_ok=True)

merged_ds = list(merged_ds.items())

shuffle(merged_ds)

test_len = ceil(len(merged_ds) * test_fraction)

test_ds = merged_ds[:test_len]

train_ds = merged_ds[test_len:]

train_ds = [{ "name" : t[0], "label" : t[1]}  for t in train_ds ]
test_ds = [{ "name" : t[0], "label" : t[1]}  for t in test_ds ]


os.makedirs('merged/train', exist_ok=True)
with open(os.path.join('merged','train','tags.json'), 'w') as f:
    json.dump(train_ds, f, indent=4)

for f in train_ds:
    path = os.path.join('bad', f["name"]) if 'bad' in f["name"] else os.path.join('healthy', f["name"])
    cv2.imwrite(os.path.join('merged','train',f["name"]), cv2.imread(path))

os.makedirs('merged/test', exist_ok=True)
with open(os.path.join('merged','test','tags.json'), 'w') as f:
    json.dump(test_ds, f, indent=4)

for f in test_ds:
    path = os.path.join('bad', f["name"]) if 'bad' in f["name"] else os.path.join('healthy', f["name"])
    cv2.imwrite(os.path.join('merged','test',f["name"]), cv2.imread(path))