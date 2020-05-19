from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from sg.GCN import RelationNet,BGGCN
from dataloader import *
import test_utils as eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import ipdb

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def test(opt):

    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    # Load pretrained model, info file, histories file
    # Create model
    model = models.setup(opt).cuda()
    modelpath = opt.load_model
    gcnpath = opt.load_gcn
    relpath = opt.pretrain_rel
    model.load_state_dict(torch.load(modelpath))
    dp_model = torch.nn.DataParallel(model)
    rel_model = RelationNet(feature_dim=opt.att_feat_size, w2v_dim=300)
    gcn_model = BGGCN(gconv_layers=opt.gcn_layers ,
        hidden_dim=opt.gcn_hidden, feat_dim=opt.att_feat_size).cuda()
    print('GCN config\nlayers : {}, hidden_size : {}'.format(opt.gcn_layers, 
        opt.gcn_hidden))
    print('Do avg feat : {}'.format(opt.avg_feat))
    gcn_model = utils.load_model(gcn_model,opt.load_gcn)
    rel_model = utils.load_model(rel_model,opt.pretrain_rel)
    gcn_model = gcn_model.cuda()
    rel_model = rel_model.cuda()
    #gcn_model = torch.nn.DataParallel(gcn_model.cuda())
    #rel_model = torch.nn.DataParallel(rel_model.cuda())
    dp_model.eval()
    gcn_model.eval()
    rel_model.eval()


    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    # Evaluate model
    eval_kwargs = {'split': 'test', 'dataset': opt.input_json}
    eval_kwargs.update(vars(opt))
    val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, gcn_model, rel_model, opt.avg_feat, crit, loader, eval_kwargs)

opt = opts.parse_opt()
test(opt)
