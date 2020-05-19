import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data
from synset_vocab import obj2idx, attr2idx, rel2idx
from synset_vocab import idx2obj, idx2attr, idx2rel
import utils
import h5py
import atexit
import ipdb

def closeh5(h5obj):
    h5obj.close()
    return
class SGDataSet(data.Dataset):
    def __init__(self, opt, sg, sg_dict, img_ids, ids):
        self.opt = opt
        if opt.w2v is not None:
            self.w2v = np.load(opt.w2v)
            self.w2v = np.insert(self.w2v, 0, 0, 0)
        else:
            self.w2v = None
        self.sg = sg
        self.sg_dict = sg_dict
        self.img_ids = img_ids
        self.ids = ids
        print('Att Feature file: ', opt.input_att_dir)
        self.input_att_dir = self.opt.input_att_dir
        self.__getitem__(0)
        
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        image_id = self.img_ids[index]
        sgid = self.ids[index]
        first_box = self.sg['img_to_first_box'][sgid]
        last_box = self.sg['img_to_last_box'][sgid]
        first_rel = self.sg['img_to_first_rel'][sgid]
        last_rel = self.sg['img_to_last_rel'][sgid]
        if first_box >= 0:
            gt_boxes = self.sg['boxes'][first_box:last_box+1]
            gt_label = self.sg['labels'][first_box:last_box+1]
        else:
            gt_boxes = np.array([])
            gt_label = np.array([])
        if first_rel >= 0:
            gt_rel = self.sg['relationships'][first_rel:last_rel+1]
            gt_pred = self.sg['predicates'][first_rel:last_rel+1]
            gt_rel = gt_rel - first_box
        else:

            gt_rel = np.array([])
            gt_pred = np.array([])
        feat_fname = os.path.join(self.input_att_dir, str(image_id)+'.npz')
        features = np.load(feat_fname)
        att_feat = features['feat']
        box = features['boxes']
        cls = features['predict_class']
        conf = features['confidence']
        
        word_emb = self.w2v[cls.flatten().astype(int)]
        # Reshape to K x C
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        return {'att_feat':att_feat, 
            'boxes':box, 
            'class':cls, 
            'conf':conf, 
            'gt_boxes':gt_boxes,
            'gt_class':gt_label,
            'gt_rel':gt_rel,
            'gt_pred':gt_pred,
            'w2v':word_emb,
            'image_id':image_id} 

    def __len__(self):
        return len(self.ids)
def collate_fn(batch):
    '''
    keys = batch[0].keys()
    output = {}
    for key in keys: output[key] = []
    for _batch in batch:
        for key in keys:
            output[key].append(_batch[key])
    '''
    output = batch
    return output
        
class TestLoader():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = getattr(opt, 'batch_size', 16)
        self.sg_path = opt.sg_dir
        print('Loading scene graph from {}'.format(self.sg_path))
        self.sg = h5py.File(os.path.join(self.sg_path, 'VG-SGG.h5'), 'r')
        self.sg_dict = json.load(open(os.path.join(self.sg_path,
            'VG-SGG-dicts.json'),'r'))
        print('Scene graph loaded')
        self.idx2img = self.sg['img_ids'].value
        self.split_dir = opt.split_dir
        self.train_idx = np.array(json.load(open(os.path.join(self.split_dir, 'train_split.json'))))
        self.val_idx = np.array(json.load(open(os.path.join(self.split_dir, 'val_split.json'))))
        self.test_idx = np.array(json.load(open(os.path.join(self.split_dir,'test_split.json'))))
        train_inds = []
        for idx in self.train_idx:
            train_inds.append(self.sg_dict['imgid_to_idx'][str(idx)])
        valid_inds = []
        for idx in self.val_idx:
            valid_inds.append(self.sg_dict['imgid_to_idx'][str(idx)])
        test_inds = []
        for idx in self.test_idx:
            test_inds.append(self.sg_dict['imgid_to_idx'][str(idx)])
        self.trainset = SGDataSet(opt, 
            self.sg, self.sg_dict, self.train_idx, train_inds)
        self.trainloader = data.DataLoader(dataset=self.trainset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)
        self.validset = SGDataSet(opt, 
            self.sg, self.sg_dict, self.val_idx, valid_inds)
        self.valloader = data.DataLoader(dataset=self.validset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)
        self.testset = SGDataSet(opt, 
            self.sg, self.sg_dict, self.test_idx, test_inds)
        self.testloader = data.DataLoader(dataset=self.testset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=False)
        atexit.register(closeh5, self.sg)
class DataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = getattr(opt, 'batch_size', 16)
        self.sg_path = opt.sg_dir
        print('Loading scene graph from {}'.format(self.sg_path))
        self.sg = h5py.File(os.path.join(self.sg_path, 'VG-SGG.h5'), 'r')
        self.sg_dict = json.load(open(os.path.join(self.sg_path,
            'VG-SGG-dicts.json'),'r'))

        print('Scene graph loaded')
        self.idx2img = self.sg['img_ids'].value
        if self.opt.only_para:
            self.split_dir = opt.split_dir
            self.train_idx = np.array(json.load(open(os.path.join(self.split_dir, 
                'train_split.json'))))
            self.val_idx = np.array(json.load(open(os.path.join(self.split_dir, 'val_split.json'))))
            self.test_idx = np.array(json.load(open(os.path.join(self.split_dir,'test_split.json'))))
        else:
            np.random.seed(1208)
            self.split_ratio = getattr(self.opt, 'split_ratio', 0.9)
            tmp = np.array(range(len(self.idx2img)))
            train_cnt = int(len(self.idx2img)*self.split_ratio)
            train_inds = np.random.choice(tmp,train_cnt,replace=False)
            train_inds.sort()
            self.train_idx = (self.idx2img[train_inds])
            self.val_idx = np.delete(self.idx2img, train_inds)
        print('Data Split, Train : {}, Val : {}'.format(len(self.train_idx), len(self.val_idx)))
        self.trainset = SGDataSet(opt, 
            self.sg, self.sg_dict, self.train_idx, train_inds)
        self.validset = SGDataSet(opt, 
            self.sg, self.sg_dict, self.val_idx, np.delete(tmp, train_inds))
        self.trainloader = data.DataLoader(dataset=self.trainset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0)
        self.valloader = data.DataLoader(dataset=self.validset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0)
        atexit.register(closeh5, self.sg)
        #print('Load paragraph meta file: {}'.format(opt.para_json))
        #self.para = json.load(open(self.opt.para_json))
def get_loader(opt):
    dataloader = DataLoader(opt)
    return dataloader
def get_test_loader(opt):
    dataloader = TestLoader(opt)
    return dataloader
