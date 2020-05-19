import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from dataloader import get_test_loader
import ipdb
from tqdm import tqdm
import utils
from model import RelationNet
def main(args):
    dataloader = get_test_loader(args)
    device_id = getattr(args, 'device', 0)
    with torch.cuda.device(device_id):
        model = RelationNet(output_class=args.n_rel).cuda()
        model.load_state_dict(torch.load(args.model_path))
        evaluate(args, dataloader, model)
    
def runEpoch(model, loader, optimizer, criterion, rel_iou_threshold=0):
    losses = []
    bg_acc = []
    bg_acc2 = []
    bg_losses = []
    rel_losses = []
    for batch in loader:
        for data in batch:
            # from faster r-cnn
            att_feats = data['att_feat']
            boxes = data['boxes']
            clses = data['class'].flatten()
            confs = data['conf']
            w2v = data['w2v']
            # GT
            _gt_boxes = data['gt_boxes']
            if _gt_boxes.shape[0] == 0:
                continue
            gt_boxes = _gt_boxes.copy()
            # transform (x,y,w,h) to (x1,y1,x2,y2)
            gt_boxes[:,2] += gt_boxes[:,0]
            gt_boxes[:,3] += gt_boxes[:,1]
            gt_clses = data['gt_class'].flatten()
            gt_rel = data['gt_rel']
            gt_pred = data['gt_pred']
            obj_list = utils.get_obj_list(boxes, clses)
            gt_obj_list = utils.get_obj_list(gt_boxes, gt_clses)
            match_list, iou_list = utils.match_obj(obj_list, gt_obj_list, iou_threshold=0.4)
            match_pair = {}
            for i, obj in enumerate(match_list):
                if obj >= 0:
                    match_pair[i] = obj
            ious = utils.calculate_iou(obj_list, mask_same_cls=True)
            train_rel_pos = {}
            train_rel_neg = []
            train_pred_neg = []
            train_iou_neg = []
            for sub in range(ious.shape[0]):
                for obj in range(ious.shape[1]):
                    if sub == obj:
                        continue
                    if ious[sub,obj] > rel_iou_threshold:
                        norel_flag = True
                        # if sub, obj not in match pair
                        # means doesn't match to object in gt
                        subidx = match_pair.get(sub, -1)
                        objidx = match_pair.get(obj, -1)
                        if subidx >= 0 and objidx >= 0:
                            for (_sub, _obj), pred in zip(gt_rel, gt_pred):
                                if _sub == subidx and _obj == objidx:
                                    if (sub,obj) not in train_rel_pos:
                                        train_rel_pos[sub,obj] = []
                                    train_rel_pos[sub,obj].append(pred[0])
                                    #train_pred_pos.append(pred[0])
                                    #train_rel_pos.append([sub,obj])
                                    #train_iou_pos.append(ious[sub,obj])
                                    norel_flag = False
                        if norel_flag:
                            train_pred_neg.append(0)
                            train_rel_neg.append([sub,obj])
                            train_iou_neg.append(ious[sub,obj])
            # Generate training sample
            if len(train_rel_pos) > 0:
                sample_size = len(train_rel_pos)
                # to One-hot
                labels = torch.zeros((sample_size, args.n_rel))
                rel_pos = []
                iou_pos = []
                for i, ((sub,obj), pred) in enumerate(train_rel_pos.items()):
                    rel_pos.append([sub,obj])
                    for _pred in pred:
                        assert _pred >= 1
                        labels[i, _pred - 1] = 1
                    iou_pos.append(ious[sub,obj])
                # Sample some neg samples
                rel_pos = torch.Tensor(rel_pos)
                iou_pos = torch.Tensor(iou_pos)
                rel_neg = torch.Tensor(train_rel_neg)
                iou_neg = torch.Tensor(train_iou_neg)
                if len(train_rel_neg) > len(train_rel_pos):
                    # Sample to same size
                    sp_idx = utils.random_sample(len(train_pred_neg), sample_size)
                    rel_neg = rel_neg[sp_idx]
                    iou_neg = iou_neg[sp_idx]
                bg_label_0 = torch.zeros((rel_pos.shape[0],1))
                bg_label_1 = torch.ones((rel_neg.shape[0],1))
                bg_label = torch.cat((bg_label_0,bg_label_1))
                rel_sample = torch.cat((rel_pos, rel_neg)).cuda()
                iou_sample = torch.cat((iou_pos, iou_neg)).unsqueeze(-1).cuda()
                att_feats = torch.Tensor(att_feats).cuda()
                w2v = torch.Tensor(w2v).cuda()
                bg_out, rel_out = model(att_feats, w2v, rel_sample, iou_sample)
                #bg_out = model(att_feats, rel_sample, iou_sample)
                rel_out = rel_out[:sample_size]
                loss_bg = criterion(bg_out, bg_label.cuda())
                loss_rel = criterion(rel_out, labels.cuda())
                loss = loss_bg + loss_rel
                loss = loss_bg
                hit = (bg_out[:sample_size]<.5).sum()+(bg_out[sample_size:]>.5).sum()
                acc2 = float((bg_out[:sample_size]<.5).sum().item())/sample_size
                acc = float(hit.item())/bg_out.shape[0]
                bg_acc.append(acc)
                bg_acc2.append(acc2)
                losses.append(loss.item())
                bg_losses.append(loss_bg.item())
                rel_losses.append(loss_rel.item())
                loader.set_description('bg={:.4f},r={:.4f},acc={:.1f},{:.1f}'.format(np.average(bg_losses),np.average(rel_losses), np.average(bg_acc)*100, np.average(bg_acc2)*100))
                #loader.set_description('Ep{}:bg={:.4f},acc={:.3f}'.format(epoch, np.average(bg_losses),np.average(bg_acc)))
    return bg_losses, rel_losses, bg_acc, bg_acc2

def evaluate(args, dataloader, model):
    rel_iou_threshold = getattr(args, 'rel_iou_threshold', 0)
    
    criterion = nn.BCELoss()
    with torch.no_grad():
        model.eval()
        print('train set')
        tq = tqdm(dataloader.trainloader)
        bg_losses, rel_losses, bg_acc, bg_acc2 = runEpoch(model, tq, None, criterion,
            rel_iou_threshold=rel_iou_threshold)
        print('val set')
        tq = tqdm(dataloader.valloader)
        bg_losses, rel_losses, bg_acc, bg_acc2 = runEpoch(model, tq, None, criterion,
            rel_iou_threshold=rel_iou_threshold)
        print('test set')
        tq = tqdm(dataloader.testloader)
        bg_losses, rel_losses, bg_acc, bg_acc2 = runEpoch(model, tq, None, criterion,
            rel_iou_threshold=rel_iou_threshold)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_json', type=str, default='image_data.json',
                    help='meta data of vg')
    parser.add_argument('--para_json', type=str, default=None,
                    help='path to the json file of paragraphs')
    parser.add_argument('--only_para', action='store_true',
                    help='train with only paragraph data')
    parser.add_argument('--split_dir', type=str, default=None,
                    help='dir of split idx json')
    parser.add_argument('--sg_dir', type=str, default=None,
                    help='dir of VG-SGG.h5 and VG-SGG-dicts.json')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the att feats')
    parser.add_argument('--w2v', type=str, default=None,
                    help='path to w2v.npy')
    parser.add_argument('--model_path', type=str, default='model/model.pt')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_rel', type=int, default=50)

    args = parser.parse_args()
    main(args)
