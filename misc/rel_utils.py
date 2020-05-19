import numpy as np
from preprocess.synset_vocab import obj2idx, attr2idx, rel2idx
from preprocess.synset_vocab import idx2obj, idx2attr, idx2rel
import torch
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
def get_obj_list(boxes, clses):
    obj_list = []
    for box, cls in zip(boxes, clses):
        obj_list.append(obj_info(box[0],box[1],box[2],box[3],cls))
    return obj_list
    
def get_triplet(data, relationloader):
    im_idx = data['infos']
    im_idx = [i['ix'] for i in im_idx]
    scene_graphs = [relationloader.ix2sg(i) for i in im_idx]
    triplets = []
    for i,idx in enumerate(im_idx):
        scene_graph = scene_graphs[i]
        boxes = data['boxes'][i]
        clses = data['class'][i]
        features = {'class': clses, 'boxes':boxes}
        triplet = match_boxes(features, scene_graph)
        triplets.append(triplet)
    max_len = len(max(triplets, key=len))
    tri = -torch.ones((len(triplets), max_len, 3))
    tri_len = torch.LongTensor([len(i) for i in triplets])
    for i, triplet in enumerate(triplets):
        for j, _triplet in enumerate(triplet):
            tri[i, j] = torch.Tensor(_triplet)
    triplets = tri.long().cuda()
    return triplets, tri_len

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
    
def gt_scene_graph(target):
    def extract_obj(obj):
        name = obj.get('name', None)
        if name is None:
            name = obj['names'][0]
        name = name.lower()
        h = obj['h']
        w = obj['w']
        x = obj['x']
        y = obj['y']
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return obj2idx.get(name, -1), x1, y1, x2, y2
    def extract_rel(rel):
        predicate = rel['predicate']
        predicate = rel2idx.get(predicate.lower(), -1)
        subject_id = rel['subject_id']
        object_id = rel['object_id']
        return subject_id, predicate, object_id
        
    relations = target['relationships']
    objects = target['objects']
    obj_set = set()
    obj_list = {}
    for _object in objects:
        obj = extract_obj(_object)
        obj_id = _object['object_id']
        if obj[0] < 0:
            continue
        #if obj in obj_set:
        #    ipdb.set_trace()
        obj_set.add(obj)
        cls, x1, y1, x2, y2 = obj
        obj_list[obj_id] = obj_info(x1,y1,x2,y2,cls)
    rel_set = set()
    rel_list = {}
    for _relation in relations:
        rel = extract_rel(_relation)
        if rel in rel_set or rel[1] < 0:
            continue
        rel_set.add(rel)
        sub_id, pred, obj_id = rel
        # generate triplet
        rel_list[sub_id, obj_id] = pred
    return obj_list, rel_list
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
def _match_boxes(obj1_list, obj2_list, iou_threshold):
    #match_list = [-1]*len(obj1_list)
    match_list = []
    iou_list = [0]*len(obj1_list)
    for idx, obj1 in enumerate(obj1_list):
        match_list.append([])
        if obj1.cls <= 0:
            # not belong to any class
            continue
        for obj_id, obj2 in obj2_list.items():
            if obj1.cls == obj2.cls:
                iou = _calculate_iou(obj1, obj2)
                #if iou > iou_threshold and iou > iou_list[idx]:
                if iou > iou_threshold:
                    # over threshold assign to the same object
                    match_list[idx].append(obj_id)
                    iou_list[idx] = max(iou, iou_list[idx])
    return match_list, iou_list

def match_boxes(source, target, iou_threshold=0.4):
    # Match the source boxes to target 
    # source : predict dict from faster-rcnn
    # target : json dict of gt
    
    # extract ground truth
    gt_obj_list, gt_rel_list = gt_scene_graph(target)
    # extract source key[image_id, att, fc, boxes, class, conf, w, h, ix]
    clses = source['class'].flatten()
    idx = clses.nonzero()[0]
    boxes = source['boxes']#[idx]
    #clses = clses[idx]
    obj_list = [obj_info(box[0], box[1], box[2], box[3], cls) 
        for box, cls in zip(boxes, clses)]
    match_list, iou_list = _match_boxes(obj_list, gt_obj_list, iou_threshold)
    triplets = get_correlation(match_list, clses, gt_rel_list)

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
                if iou > iou_list[idx]:
                # if iou > iou_threshold:
                    # over threshold assign to the same object
                    if iou > iou_threshold:
                        match_list[idx] = gt_idx
                    iou_list[idx] = iou
    return match_list, iou_list
if __name__ == '__main__':
    ipdb.set_trace()
