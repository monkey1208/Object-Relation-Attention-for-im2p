import json
from tqdm import tqdm

def high_freq_rel(relations, threshold=10):
    Pred2 = relations.copy()
    for pred, cnt in Pred2.copy().items():
        if cnt < threshold:
            Pred2.pop(pred, None)
    return Pred2

path = '/2t/ylc/VG_paragraph/VG/relationships.json'
fp = open(path, 'r')
datas = json.load(fp)
Pred = {}
for data in tqdm(datas):
    data = data['relationships']
    for _data in data:
        predicate = _data['predicate'].lower()
        if predicate not in Pred:
            Pred[predicate] = 1
        else:
            Pred[predicate] += 1
Pred2 = high_freq_rel(Pred, 10)
print(len(datas))
import ipdb
ipdb.set_trace()
