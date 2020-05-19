import json

f = open('../object_synsets.json', 'r')
f = open('../relationship_synsets.json', 'r')
data = json.load(f)
print(len(data))
data_d = set()
for obj, syn in data.items():
    data_d.add(syn)
import ipdb
ipdb.set_trace()
