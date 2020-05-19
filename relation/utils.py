import numpy as np
from synset_vocab import obj2idx, attr2idx, rel2idx
from synset_vocab import idx2obj, idx2attr, idx2rel
import torch
import random
import ipdb

class obj_info():
    def __init__(self, x1, y1, x2, y2, cls):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls = int(cls)
        self.name = idx2obj.get(self.cls, '__background__')
        self.area = (x2-x1+1)*(y2-y1+1)
    def __repr__(self):
        return '(x1,y1,x2,y2) : ({} {} {} {})\narea : {}\nclass : {}({})\n'.format(
            self.x1, self.y1, self.x2, self.y2, self.area, self.cls, self.name)

def _calculate_iou(obj1, obj2):
    xA = max(obj1.x1, obj2.x1)
    yA = max(obj1.y1, obj2.y1)
    xB = min(obj1.x2, obj2.x2)
    yB = min(obj1.y2, obj2.y2)
    intersection = max(xB - xA + 1, 0)*max(yB - yA + 1, 0)
    iou = intersection / (obj1.area + obj2.area - intersection)
    return iou
def calculate_iou(objs, mask_same_cls=False):
    IOUs = np.identity(len(objs))
    for i in range(len(objs)):
        if objs[i].cls <= 0:
            continue
        for j in range(i+1, len(objs)):
            if (mask_same_cls and objs[i].cls == objs[j].cls) or objs[j].cls <= 0:
                continue
            IOUs[i,j] = IOUs[j,i] = _calculate_iou(objs[i], objs[j])
    return IOUs
def get_obj_list(boxes, clses):
    obj_list = []
    for box, cls in zip(boxes, clses):
        obj_list.append(obj_info(box[0],box[1],box[2],box[3],cls))
    return obj_list
def get_correlation(obj_list, clses, rel_list):
    # check the correlation between objects in obj_list
    triplets = []
    for sub_idx, (sub_cls, sub) in enumerate(zip(clses, obj_list)):
        for _sub in sub:
            for obj_idx, (obj_cls, obj) in enumerate(zip(clses, obj_list)):
                if sub_cls == obj_cls: continue
                for _obj in obj:
                    pred = rel_list.get((_sub, _obj), -1)
                    if pred != -1:
                        triplets.append([sub_idx, pred, obj_idx])
    return triplets
def match_obj(obj_list, gt_obj_list, iou_threshold=0.4):
    # Match the source boxes to target 
    # source : predict dict from faster-rcnn
    # target : GT
    
    # extract source key[image_id, att, fc, boxes, class, conf, w, h, ix]
    match_list = [-1]*len(obj_list)
    iou_list = [0]*len(obj_list)
    for idx, obj in enumerate(obj_list):
        if obj.cls <= 0:
            # not belong to any class
            continue
        for gt_idx, gt_obj in enumerate(gt_obj_list):
            if obj.cls == gt_obj.cls:
                iou = _calculate_iou(obj, gt_obj)
                if iou > iou_threshold and iou > iou_list[idx]:
                # if iou > iou_threshold:
                    # over threshold assign to the same object
                    match_list[idx] = gt_idx
                    iou_list[idx] = iou
    return match_list, iou_list
    #triplets = get_correlation(match_list, clses, gt_rel_list)

    return triplets
def random_sample(sample_range,sample_cnt):
    return sorted(random.sample(range(sample_range), sample_cnt))
if __name__ == '__main__':
    ipdb.set_trace()
