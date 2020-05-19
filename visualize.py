
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
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import json
from json import encoder
import random
import string
import sys
from relation.iteration import rel_iteration_visualize, avg_features
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
    #dp_model = torch.nn.DataParallel(model)
    dp_model = model
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
    val_loss, predictions, lang_stats = eval_split(dp_model, gcn_model, rel_model, opt.avg_feat, crit, loader, eval_kwargs)


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/para_captions_{}.json'.format(split)
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('visual_results'):
        os.mkdir('visual_results')
    cache_path = os.path.join('visual_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    #preds_filt = [p for p in preds if p['image_id'] in valids]
    preds_filt = []
    id2bg = {}
    id2pairs = {}
    id2class = {}
    id2datalen = {}
    id2boxes = {}
    for p in preds:
        if p['image_id'] in valids:
            preds_filt.append({'image_id':p['image_id'], 'caption':p['caption']})
            id2bg[p['image_id']] = p['bg']
            id2pairs[p['image_id']] = p['obj_pairs']
            id2class[p['image_id']] = p['class']
            id2datalen[p['image_id']] = p['data_len']
            id2boxes[p['image_id']] = p['boxes']
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    vg_dict = json.load(open(os.path.join('/shared_home/ylc/vg_data/data_tools/1600-50/VG-SGG-dicts.json'),'r'))
    id2label = vg_dict['idx_to_label']
    
    vis_result = {}
    for i, p in enumerate(preds_filt):
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
        cider_score = imgToEval[image_id]['CIDEr']
        BLEU1 = imgToEval[image_id]['Bleu_1']
        scores = imgToEval[image_id]
        if cider_score > 2 or BLEU1 > 0.5:
        #if cider_score < 0.2 or BLEU1 < 0.2:
            bg = id2bg[image_id]
            _pairs = id2pairs[image_id]
            classes = id2class[image_id]
            data_len = id2datalen[image_id]
            boxes = id2boxes[image_id]
            pairs = []
            outputboxes = []
            for pair in _pairs:
                class1 = id2label[str(classes[pair[0]].astype(int)[0])]
                class2 = id2label[str(classes[pair[1]].astype(int)[0])]
                box1 = boxes[pair[0].astype(int)]
                box2 = boxes[pair[1].astype(int)]
                pairs.append([class1, class2])
                outputboxes.append([box1.tolist(),box2.tolist()])
            pairs = json.dumps(pairs)
            outputboxes = json.dumps(outputboxes)
            bg = bg.flatten().tolist()
            vis_result[image_id] = {'image_id':image_id,'cider':cider_score,'caption':caption,
                'pair':pairs,'bleu':BLEU1, 'datalen':data_len, 'scores':scores, 'boxes':outputboxes, 'bg':bg}
    json.dump(vis_result,open('vis_result.json','w'))
    
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, gcn_model, rel_model, avg_feat, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    bgs = []
    import time
    time0 = time.time()
    while True:
        batch_bgs = []
        batch_obj_pairs = []
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp
            
            # Graph Convolution
            ids = [_['id'] for _ in data['infos']]
            myid = 2385111
            if myid not in ids:
                continue
            tmp2 = (np.array(ids) == myid)
            for i in range(len(tmp)):
                tmp[i] = tmp[i][tmp2]
            with torch.no_grad():
                new_att_feats, bg_feat, _, _, _,_ = rel_iteration_visualize(data, att_feats, rel_model, gcn_model)
                att_feats = new_att_feats
                fc_feats = avg_features(fc_feats, att_feats, att_masks, avg_feat)
                fc_feats = fc_feats[tmp2]
                att_feats = att_feats[tmp2]
                labels = labels[tmp2]
                att_masks = att_masks[tmp2]
                ipdb.set_trace()
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        #
        # Graph Convolution
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            new_att_feats, bg_feat, obj_pairs, classes, boxes, data_len = rel_iteration_visualize(data, att_feats, rel_model, gcn_model)
            att_feats = new_att_feats
            fc_feats = avg_features(fc_feats, att_feats, att_masks, avg_feat)
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry['bg'] = bg_feat[k]
            entry['obj_pairs'] = obj_pairs[k]
            entry['class'] = classes[k]
            entry['boxes'] = boxes[k]
            entry['data_len'] = data_len[k]
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    
    time1 = time.time()
    print(time0)
    print(time1)
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    return loss_sum/loss_evals, predictions, lang_stats

opt = opts.parse_opt()
test(opt)
