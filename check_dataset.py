import os
import json
import numpy as np
import matplotlib.pyplot as plt 
import sys

dir_name = sys.argv[1]

with open(os.path.join('dataset/' + dir_name, 'tags.json'), 'r') as f:
    pannels_data = json.load(f)

defects = [d["label"] for d in pannels_data]
defects = np.array(defects)

tags, count = np.unique(defects, return_counts=True)

for t in tags:
    print(f"{t}: {count[tags == t][0]}")

print("_________________________")

for i in range(len(tags)):
    print(f"{tags[i]}, = {i}")

plt.bar(list(range(len(tags))), count)
plt.show()

