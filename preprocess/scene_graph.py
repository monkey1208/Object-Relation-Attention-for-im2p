import json
import numpy as np
from tqdm import tqdm
import pickle
def high_freq_rel(relations, threshold=10):
    Pred2 = relations.copy()
    for pred, cnt in Pred2.copy().items():
        if cnt < threshold:
            Pred2.pop(pred, None)
    return Pred2
def save_object_id(image_list):
    #obj2id, id2obj = {}, {}
    #attr2id, id2attr = {}, {}
    all_obj, all_attr = {}, {}
    for image in tqdm(image_list):
        object_list = image['objects']
        for obj in object_list:
            names = obj['names']
            for name in names:
                name = name.replace('\"', '').strip()
                if name not in all_obj:
                    all_obj[name] = 0
                all_obj[name] += 1
            if 'attributes' not in obj:
                continue
            attributes = obj['attributes']
            for attr in attributes:
                attr = attr.strip()
                if attr not in all_attr:
                    all_attr[attr] = 0
                all_attr[attr] += 1
    # sort by word
    all_obj = sorted(all_obj.items(), key=lambda x:x[0])
    all_attr = sorted(all_attr.items(), key=lambda x:x[0])
    with open('data/all_obj.csv', 'w') as f:
        f.write('object,count\n')
        for (obj, cnt) in all_obj:
            f.write('{},{}\n'.format(obj,cnt))
    with open('data/all_attr.csv', 'w') as f:
        f.write('attribute,count\n')
        for (attr, cnt) in all_attr:
            f.write('{},{}\n'.format(attr,cnt))
    return
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', default='data/', type=str)
    args = parser.parse_args()
    path = args.data
    sg_path = os.path.join(path, 'scene_graphs.json')
    #obj_path = os.path.join(path, 'objects.json')
    f = open(os.path.join(path, '../paragraphs_v1.json'))
    paras = json.load(f)
    imids_p = set()
    for pg in paras:
        imids_p.add(pg['image_id'])
    f = open(sg_path)
    scene_graph = json.load(f)
    imids = set()
    rel_dict = {}
    SG = []
    for sg in scene_graph:
        imids.add(sg['image_id'])
        if sg['image_id'] in imids_p:
            SG.append(sg)
        rels = sg['relationships']
        for rel in rels:
            pred = rel['predicate'].lower()
            if pred not in rel_dict:
                rel_dict[pred] = 0
            rel_dict[pred] += 1
    notin = 0
    for idx in imids_p:
        if idx not in imids:
            notin += 1
    print(notin)
    print(len(imids_p))
    #images = os.listdir(os.path.join(path, 'VG_100k_all'))
    import ipdb
    ipdb.set_trace()
    #save_object_id(scene_graph)
