import os
import numpy as np
path = '/2t/ylc/image-paragraph-captioning/data/parabu_att/'
#path = '/2t/ylc/VG_feat/attn_feat/'
path2 = '/2t/ylc/image-paragraph-captioning/data/parabu_att_2/'
files = os.listdir(path)
files2 = os.listdir(path2)
incorrect = 0
print(len(files))
print(len(files2))
for fname in files2:
    filename = os.path.join(path, fname)
    filename2 = os.path.join(path2, fname)
    feat = np.load(filename)['feat']
    feat2 = np.load(filename2)['feat']
    cnt = feat.shape[0]
    cnt2 = feat2.shape[0]
    if cnt != cnt2:
        incorrect += 1
print(incorrect)
print(len(files))

