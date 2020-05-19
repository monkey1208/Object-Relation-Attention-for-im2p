import misc.rel_utils as rutils
import math
import torch
import ipdb

def gcn_iteration(data, att_feats, gcn_model):
    new_att_feats = att_feats.clone()
    for b in range(len(data['gt_box'])):
        gt_box = data['gt_box'][b].copy()
        if gt_box.shape[0] == 0:
            continue
        gt_box[:,2] += gt_box[:,0]
        gt_box[:,3] += gt_box[:,1]
        gt_obj_list = rutils.get_obj_list(gt_box, data['gt_cls'][b])
        obj_list = rutils.get_obj_list(data['boxes'][b], data['class'][b])
        match_list, iou_list = rutils.match_obj(obj_list, gt_obj_list)
        obj2gt = {}
        gt2obj = {}
        for i in range(len(match_list)):
            matchobj = match_list[i]
            if matchobj >= 0:
                obj2gt[i] = matchobj
                gt2obj[matchobj] = i
        triplet, predicates = [], []
        for rel, pred in zip(data['gt_rel'][b], data['gt_pred'][b]):
            sub = gt2obj.get(rel[0])
            obj = gt2obj.get(rel[1])
            if sub is not None and obj is not None:
                #triplet.append([sub, pred[0], obj])
                triplet.append([sub, obj])
                predicates.append(pred[0])
        if len(triplet) == 0:
            continue
        triplet = torch.LongTensor(triplet)
        predicates = torch.LongTensor(predicates)
        obj_feat = gcn_model(att_feats[b], triplet, predicates)
        new_att_feats[b] = obj_feat
    return new_att_feats
def rel_iteration(data, att_feats, rel_model, gcn_model, rel_iou_threshold=0):
    new_att_feats = att_feats.clone()
    #new_w2v_feats = torch.zeros((att_feats.size(0),att_feats.size(1),300)).cuda()
    for b in range(len(data['gt_box'])):
        obj_list = rutils.get_obj_list(data['boxes'][b], data['class'][b])
        ious = rutils.calculate_iou(obj_list, mask_same_cls=True)
        obj_pairs = []
        iou_list = []
        for sub in range(ious.shape[0]):
            for obj in range(ious.shape[1]):
                if sub == obj:
                    continue
                if ious[sub, obj] > rel_iou_threshold:
                    obj_pairs.append([sub, obj])
                    iou_list.append(ious[sub,obj])
        if len(obj_pairs) == 0:
            continue
        batch_feats = att_feats[b]
        w2v = torch.Tensor(data['w2v'][b]).cuda()
        obj_pairs = torch.LongTensor(obj_pairs)
        iou_list = torch.Tensor(iou_list).cuda()
        bg_feat = rel_model(batch_feats, w2v, obj_pairs, iou_list)
        # shrink bg_feats to last50
        idx = bg_feat.argsort(0)[:50]
        # shrink bg_feats to top50
        #idx = bg_feat.argsort(0)[-50:]
        #idx = bg_feat.argsort(0)
        bg_feat = bg_feat[idx]
        obj_pairs = obj_pairs[idx.flatten()]
        if len(obj_pairs) == 0:
            continue
        new_att_feat = gcn_model(batch_feats, obj_pairs, bg_feat)
        #new_att_feat, new_w2v = gcn_model(batch_feats, obj_pairs, bg_feat, w2v)
        #new_w2v_feats[b] = new_w2v
        new_att_feats[b] = new_att_feat
    #new_att_feats = torch.cat((new_att_feats,new_w2v_feats),-1)
    return new_att_feats
def noattn_rel_iteration(data, att_feats, gcn_model, rel_iou_threshold=0):
    new_att_feats = att_feats.clone()
    #new_w2v_feats = torch.zeros((att_feats.size(0),att_feats.size(1),300)).cuda()
    for b in range(len(data['gt_box'])):
        obj_list = rutils.get_obj_list(data['boxes'][b], data['class'][b])
        ious = rutils.calculate_iou(obj_list, mask_same_cls=True)
        obj_pairs = []
        iou_list = []
        for sub in range(ious.shape[0]):
            for obj in range(ious.shape[1]):
                if sub == obj:
                    continue
                if ious[sub, obj] > rel_iou_threshold:
                    obj_pairs.append([sub, obj])
                    iou_list.append(ious[sub,obj])
        if len(obj_pairs) == 0:
            continue
        batch_feats = att_feats[b]
        w2v = torch.Tensor(data['w2v'][b]).cuda()
        obj_pairs = torch.LongTensor(obj_pairs)
        iou_list = torch.Tensor(iou_list).cuda()
        # shrink bg_feats to last50
        #idx = bg_feat.argsort(0)[:50]
        # shrink bg_feats to top50
        #idx = bg_feat.argsort(0)[-50:]
        #bg_feat = bg_feat[idx]
        #obj_pairs = obj_pairs[idx.flatten()]
        if len(obj_pairs) == 0:
            continue
        new_att_feat = gcn_model(batch_feats, obj_pairs, None)
        #new_att_feat, new_w2v = gcn_model(batch_feats, obj_pairs, bg_feat, w2v)
        #new_w2v_feats[b] = new_w2v
        new_att_feats[b] = new_att_feat
    #new_att_feats = torch.cat((new_att_feats,new_w2v_feats),-1)
    return new_att_feats
def emb_rel_iteration(data, att_feats, rel_model, gcn_model, rel_iou_threshold=0):
    new_att_feats = att_feats.clone()
    for b in range(len(data['gt_box'])):
        obj_list = rutils.get_obj_list(data['boxes'][b], data['class'][b])
        ious = rutils.calculate_iou(obj_list, mask_same_cls=True)
        obj_pairs = []
        iou_list = []
        for sub in range(ious.shape[0]):
            for obj in range(sub, ious.shape[1]):
                if sub == obj:
                    continue
                if ious[sub, obj] > rel_iou_threshold:
                    obj_pairs.append([sub, obj])
                    iou_list.append(ious[sub,obj])
        if len(obj_pairs) == 0:
            continue
        batch_feats = att_feats[b]
        w2v = torch.Tensor(data['w2v'][b]).cuda()
        obj_pairs = torch.LongTensor(obj_pairs)
        iou_list = torch.Tensor(iou_list)
        bg_feat = rel_model(batch_feats, w2v, obj_pairs, iou_list)
        # shrink bg_feats to last50
        idx = bg_feat.argsort(0)[:50]
        # shrink bg_feats to top50
        #idx = bg_feat.argsort(0)[-50:]
        #idx = bg_feat.argsort(0)
        bg_feat = bg_feat[idx]
        obj_pairs = obj_pairs[idx.flatten()]
        if len(obj_pairs) == 0:
            continue
        new_att_feat = gcn_model(batch_feats, obj_pairs, bg_feat, w2v)
        #new_att_feat = gcn_model(batch_feats, obj_pairs)
        new_att_feats[b] = new_att_feat
    return new_att_feats
def rel_iteration_visualize(data, att_feats, rel_model, gcn_model, rel_iou_threshold=0):
    new_att_feats = att_feats.clone()
    #data_len = [i.shape[0] for i in data['class']]
    data_len = []
    all_bg = []
    all_pairs = []
    for b in range(len(data['gt_box'])):
        obj_list = rutils.get_obj_list(data['boxes'][b], data['class'][b])
        ious = rutils.calculate_iou(obj_list, mask_same_cls=True)
        obj_pairs = []
        iou_list = []
        for sub in range(ious.shape[0]):
            for obj in range(ious.shape[1]):
                if sub == obj:
                    continue
                if ious[sub, obj] > rel_iou_threshold:
                    obj_pairs.append([sub, obj])
                    iou_list.append(ious[sub,obj])
        data_len.append(len(obj_pairs))
        if len(obj_pairs) == 0:
            all_bg.append(np.array([]))
            all_pairs.append(np.array([]))
            continue
        batch_feats = att_feats[b]
        w2v = torch.Tensor(data['w2v'][b]).cuda()
        obj_pairs = torch.LongTensor(obj_pairs).cuda()
        iou_list = torch.Tensor(iou_list).cuda()
        bg_feat = rel_model(batch_feats, w2v, obj_pairs, iou_list)
        idx = bg_feat.argsort(0)[-50:]
        #idx = bg_feat.argsort(0)[:50]
        bg_feat = bg_feat[idx]
        obj_pairs = obj_pairs[idx.flatten()]
        new_att_feat = gcn_model(batch_feats, obj_pairs, bg_feat)
        new_att_feats[b] = new_att_feat
        all_bg.append(bg_feat.detach().cpu().numpy())
        all_pairs.append(obj_pairs.detach().cpu().numpy())
    return new_att_feats, all_bg, all_pairs, data['class'], data['boxes'], data_len
def avg_features(fc_feats, att_feats, att_masks, use_avg):
    if not use_avg:
        return fc_feats
    fc_feats = fc_feats.clone()
    #att_feats = att_feats[:,:,:2048]
    
    mask_sum = att_masks.sum(1)
    for i in range(mask_sum.shape[0]):
        assert att_masks[i,:mask_sum[i].int().item()].sum() == mask_sum[i]
        #fc_feats[i] = att_feats[i,:mask_sum[i].int().item()].mean(0)
        fc_feats[i] = att_feats[i,:mask_sum[i].int().item()].mean(0).detach()
    return fc_feats
