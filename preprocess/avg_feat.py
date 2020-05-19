import os
import numpy as np
path = '/2t/ylc/image-paragraph-captioning/data/parabu_att/'
path2 = '/2t/ylc/image-paragraph-captioning/data/parabu_fc/'
files = os.listdir(path)
incorrect = 0
print(len(files))
for fname in files:
    filename = os.path.join(path, fname)
    filename2 = os.path.join(path2, fname.replace('.npz','.npy'))
    feat = np.load(filename)['feat']
    feat = feat.mean(0)
    np.save(filename2, feat)

