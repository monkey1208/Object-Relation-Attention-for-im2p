import torch
import torch.nn as nn
import torch.nn.functional as F
from __init__ import build_layers
from preprocess.synset_vocab import OBJ_NUM, REL_NUM
import ipdb

def _init_weight(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
class GConv(nn.Module):
    # single layer
    def __init__(self, input_dim, hidden_dim, gconv_pooling='avg'):
        super(GConv, self).__init__()
        self.gconv_pooling = gconv_pooling
        # triplet input, encode to a hidden layer and output to subject, relation, object
        # TODO: Maybe Try these architecture
        # 1. object & subject feature forward to a symmetric hidden layer
        #conv_layers = [3*input_dim, gconv_hidden_dim, gconv_dim+gconv_hidden_dim*2]
        
        # 2. No hidden layer used
        #conv_layers = [3*embedding_dim, gconv_dim+gconv_hidden_dim*2]
        
        # 3. AE-liked architecture
        conv_layers = [sum(input_dim), hidden_dim, sum(input_dim)]
        
        self.input_dim = input_dim
        if len(input_dim) == 3:
            # use relation
            #conv_layers = [sum(input_dim), hidden_dim, input_dim[1]+hidden_dim*2]
            self.use_relation = True
        elif len(input_dim) == 2:
            #conv_layers = [sum(input_dim), hidden_dim, hidden_dim*2]
            self.use_relation = False
        else:
            raise ValueError
        
        
        self.conv = build_layers(conv_layers)
        self.conv.apply(_init_weight)
    def forward(self, object_feats, relation_feats, triplet, triplet_len):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        if triplet_len is None:
            triplet_len = len(triplet)
        obj_cnt = object_feats.size(0)
        obj_count = torch.zeros(obj_cnt).cuda()
        calculated_obj = set()
        triplet = triplet[:triplet_len]
        subs = triplet[:,0].contiguous()
        objs = triplet[:,1].contiguous()
        #rels = triplet[:,1]
        #objs = triplet[:,2]
        for sub, obj in zip(subs, objs):
            obj_count[sub] += 1
            obj_count[obj] += 1
            calculated_obj.add(sub.item())
            calculated_obj.add(obj.item())
        sub_feats = object_feats[subs]
        obj_feats = object_feats[objs]
        rel_feats = relation_feats
        if self.use_relation:
            gcn_feat = torch.cat((sub_feats, rel_feats, obj_feats), 1)
        else:
            gcn_feat = torch.cat((sub_feats, obj_feats), 1)
        output = self.conv(gcn_feat)
        if self.use_relation:
            sub_out = output[:,:self.input_dim[0]]
            rel_out = output[:,self.input_dim[0]:-self.input_dim[-1]]
            obj_out = output[:,-self.input_dim[-1]:]
        else:
            sub_out = output[:,:self.input_dim[0]]
            obj_out = output[:,-self.input_dim[-1]:]
        # Do avg pooling
        output_feat = torch.zeros(object_feats.shape).cuda()
        for feat1, feat2, rel in zip(sub_out, obj_out, triplet):
            output_feat[rel[0]] += feat1
            output_feat[rel[1]] += feat2
        obj_count = obj_count.clamp(1).expand(output_feat.size(1),output_feat.size(0)).transpose(1,0)
        output_feat = output_feat/obj_count
        # use resnet arch
        output_feat += object_feats
        # not using resnet
        '''
        for cobj in calculated_obj:
            object_feats[cobj] = output_feat[cobj]
        '''
        return output_feat, rel_out, triplet, triplet_len
        
class GCN(nn.Module):
    def __init__(self, embed_dim=1024, feat_dim=2048, hidden_dim=2048,
        gconv_pooling='avg', gconv_layers=1, use_relation=True, obj_embedding=0,
        rel_num=20, obj_num=1600):
        super(GCN, self).__init__()
        self.obj_cnt, self.rel_cnt = obj_num, rel_num
        self.use_obj_emb = False
        if obj_embedding > 0:
            self.obj_embedding = nn.Embedding(self.obj_cnt, obj_embedding)
            self.use_obj_emb = True
            
        self.rel_embedding = nn.Embedding(self.rel_cnt, embed_dim)
        if gconv_layers == 0: # 0 means no convolution, only map obj_embedding to output
            #self.gconv = nn.Linear(embed_dim, gconv_dim)
            self.gconv = Identity()
        elif gconv_layers > 0:
            if use_relation:
                self.gconv = GConv([feat_dim, embed_dim, feat_dim], hidden_dim)
            else:
                self.gconv = GConv([feat_dim]*2, hidden_dim)
        self.use_relation = use_relation
            
        
    def forward(self, features, triplets, triplet_lens):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        # for avg pooling
        # obj_count = torch.zeros(features.shape[:2])
        # Try to do GCN with single data a time(not a batch)
        obj_cnt, rel_cnt = features.size(1), triplets.size(1)
        output = features.clone()
        for batch, (triplet, triplet_len) in enumerate(zip(triplets, triplet_lens)):
            if triplet_len == 0:
                continue
            
            rels = triplet[:triplet_len,1]
            rel_feats = self.rel_embedding(rels)
            obj_feats, rel_feats, _, _ = self.gconv(features[batch], 
                rel_feats, triplet, triplet_len)
            output[batch] = obj_feats
        #obj_emb = self.obj_embedding(objects)
        return output
class GCN_unit(nn.Module):
    def __init__(self, embed_dim=1024, feat_dim=2048, hidden_dim=2048,
        gconv_pooling='avg', gconv_layers=1, use_relation=True, obj_embedding=0,
        rel_num=20, obj_num=1600):
        super(GCN_unit, self).__init__()
        self.obj_cnt, self.rel_cnt = obj_num, rel_num
        self.use_obj_emb = False
        if obj_embedding > 0:
            self.obj_embedding = nn.Embedding(self.obj_cnt, obj_embedding)
            self.use_obj_emb = True
            
        self.rel_embedding = nn.Embedding(self.rel_cnt+1, embed_dim)
        if gconv_layers == 0: # 0 means no convolution, only map obj_embedding to output
            #self.gconv = nn.Linear(embed_dim, gconv_dim)
            self.gconv = Identity()
        elif gconv_layers > 0:
            if use_relation:
                self.gconv = GConv([feat_dim, embed_dim, feat_dim], hidden_dim)
            else:
                self.gconv = GConv([feat_dim]*2, hidden_dim)
        self.use_relation = use_relation
            
        
    def forward(self, obj_feats, pairs, relationship):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        # for avg pooling
        # obj_count = torch.zeros(features.shape[:2])
        # Try to do GCN with single data a time(not a batch)
        obj_cnt, rel_cnt = len(obj_feats), len(pairs)
        rel_feats = self.rel_embedding(relationship)
        obj_feats, rel_feats, _, _ = self.gconv(obj_feats, 
            rel_feats, pairs, None)
        return obj_feats
# Non-symmetric
class BGConv_unit(nn.Module):
    # single layer
    def __init__(self, input_dim, output_dim, hidden_dim, gconv_pooling='avg'):
        super(BGConv_unit, self).__init__()
        self.gconv_pooling = gconv_pooling
        conv_layers = [sum(input_dim), hidden_dim, sum(output_dim)]
        # exp 2-layer
        #conv_layers = [sum(input_dim), sum(output_dim)]
        # exp 3-layer no bottleneck
        #conv_layers = [sum(input_dim), sum(input_dim), sum(output_dim)]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = build_layers(conv_layers)
        self.conv.apply(_init_weight)

        #self.baseline = nn.Parameter(2*torch.ones(1))
        #self.baseline = 2*torch.ones(1)
        self.constant = 10*torch.ones(1)
        #self.object_weight = nn.Parameter(torch.ones(1))
        print('constant = %f'%self.constant.item())
    def forward(self, object_feats, pairs, confidence):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        #tmp = torch.cat((object_feats, w2v),-1)
        #tmp = object_feats
        features = object_feats[pairs.long()]
        #features = tmp[pairs.long()]
        features = torch.cat((features[:,0],features[:,1]),1)
        output = self.conv(features)
        feat_dict = {}
        conf_dict = {}
        new_feats = object_feats.clone()
        for i in range(pairs.max().item()+1):
            feat_dict[i] = [object_feats[i]]
            conf_dict[i] = [self.constant.cuda()]
        
        for feat, pair, conf in zip(output, pairs, confidence):
            
            sub_feat, obj_feat = feat[:self.output_dim[0]],feat[-self.output_dim[1]:]
            sub, obj = pair[0].item(), pair[1].item()
            feat_dict[sub].append(sub_feat)
            conf_dict[sub].append(conf.view(-1))
            feat_dict[obj].append(obj_feat)
            conf_dict[obj].append(conf.view(-1))
        
            # exp asym->sym
        '''
        for feat, pair, conf in zip(output, pairs, confidence):
            sub, obj = pair[0].item(), pair[1].item()
            feat_dict[sub].append(feat)
            conf_dict[sub].append(conf.view(-1))
            feat_dict[obj].append(feat)
            conf_dict[obj].append(conf.view(-1))
        '''
        '''
        for feat, pair in zip(output, pairs):
            sub_feat, obj_feat = feat[:self.output_dim[0]],feat[-self.output_dim[1]:]
            sub, obj = pair[0].item(), pair[1].item()
            feat_dict[sub].append(sub_feat)
            conf_dict[sub].append(self.object_weight)
            feat_dict[obj].append(obj_feat)
            conf_dict[obj].append(self.object_weight)
        '''
        for key in feat_dict:
            if len(feat_dict[key]) == 0:
                continue
            confs = torch.cat(conf_dict[key])
            feats = torch.stack(feat_dict[key])
            '''
            if confs.shape[0] > 1:
                weights = nn.functional.softmax(confs[1:])
                new_feat = torch.matmul(weights, feats[1:])
                new_feats[key] *= .4
                new_feat *= .6#.6
                new_feats[key] += new_feat
                #new_feats[key] += new_feat
                #new_feats[key] *= .5
            '''
            weights = nn.functional.softmax(confs)
            new_feat = torch.matmul(weights, feats)
            new_feats[key] = new_feat
        return new_feats, pairs, confidence
class BGGCN(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=2048,
        gconv_pooling='avg', gconv_layers=1, w2v_dim=300):
        super(BGGCN, self).__init__()
        print('gconv_layers = %i'%gconv_layers)
        self.gconv_layers = gconv_layers
        self.w2v_dim = w2v_dim
        if gconv_layers == 0: # 0 means no convolution, only map obj_embedding to output
            #self.gconv = nn.Linear(embed_dim, gconv_dim)
            self.gconv = Identity()
        elif gconv_layers > 0:
            self.gconv = BGConv_unit([feat_dim]*2, [feat_dim]*2, hidden_dim)
            # w2v conv
            #self.gconv = BGConv_unit([feat_dim+w2v_dim]*2, [feat_dim]*2, hidden_dim)
            #self.w2v_gconv = BGConv_unit([w2v_dim]*2, [w2v_dim]*2, hidden_dim)
            # asym->sym
            #self.gconv = BGConv_unit([feat_dim]*2, [feat_dim], hidden_dim)
        
    def forward(self, obj_feats, pairs, confidence):#, w2v):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        # Try to do GCN with single data a time(not a batch)
        #pairs2 = pairs.clone()
        #confidence2 = confidence.clone()
        #tmp = torch.zeros((obj_feats.size(0),w2v.size(1))).cuda()
        #tmp[:w2v.size(0)] = w2v
        #w2v = tmp
        #obj_feats = torch.cat((obj_feats, w2v),-1)
        for i in range(self.gconv_layers):
            obj_feats, pairs, confidence = self.gconv(obj_feats, pairs, confidence)
            #obj_feats, pairs, confidence = self.gconv(obj_feats, pairs, confidence, w2v)
            #obj_feats, pairs, confidence = self.gconv(obj_feats, pairs, confidence, None)
            #w2v, pairs2, confidence2 = self.w2v_gconv(w2v, pairs2, confidence2, None)
        return obj_feats
        #return obj_feats, w2v
# Symmetric
class EmbBGConv_unit(nn.Module):
    # single layer
    def __init__(self, input_dim, output_dim, hidden_dim, gconv_pooling='avg'):
        super(EmbBGConv_unit, self).__init__()
        self.gconv_pooliing = gconv_pooling
        conv_layers = [input_dim, output_dim]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb = build_layers(conv_layers)
        self.emb.apply(_init_weight)

        #self.baseline = nn.Parameter(2*torch.ones(1))
        #self.baseline = 2*torch.ones(1)
        self.baseline = 10*torch.ones(1)
        print('constant = %f'%self.baseline.item())
    def forward(self, object_feats, pairs, confidence):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        max_idx = pairs.max()
        input_feats = object_feats[:max_idx+1]
        input_feats = self.emb(input_feats)
        output = input_feats[pairs.long()]
        feat_dict = {}
        conf_dict = {}
        new_feats = object_feats.clone()
        for i in range(pairs.max().item()+1):
            feat_dict[i] = [object_feats[i]]
            conf_dict[i] = [self.baseline.cuda()]
        for feat, pair, conf in zip(output, pairs, confidence):
            sub_feat, obj_feat = feat[0],feat[1]
            new_feat = sub_feat*obj_feat
            sub, obj = pair[0].item(), pair[1].item()
            # sym->sym
            feat_dict[sub].append(new_feat)
            conf_dict[sub].append(conf.view(-1))
            feat_dict[obj].append(new_feat)
            conf_dict[obj].append(conf.view(-1))
            
            # experiment sym->asym
            #sub_feat, obj_feat = feat[:self.output_dim[0]],feat[-self.output_dim[1]:]
        for key in feat_dict:
            if len(feat_dict[key]) == 0:
                continue
            confs = torch.cat(conf_dict[key])
            feats = torch.stack(feat_dict[key])
            weights = nn.functional.softmax(confs)
            new_feat = torch.matmul(weights, feats)
            new_feats[key] = new_feat
        return new_feats, pairs, confidence
# Symmetric
class EmbBGGCN(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=2048,
        gconv_pooling='avg', gconv_layers=1):
        super(EmbBGGCN, self).__init__()
        print('gconv_layers = %i'%gconv_layers)
        self.gconv_layers = gconv_layers
        if gconv_layers == 0: # 0 means no convolution, only map obj_embedding to output
            #self.gconv = nn.Linear(embed_dim, gconv_dim)
            self.gconv = Identity()
        elif gconv_layers > 0:
            self.gconv = EmbBGConv_unit(feat_dim, hidden_dim, hidden_dim)
            #self.gconv = EmbBGConv_unit(feat_dim, hidden_dim, [hidden_dim]*2)
        
    def forward(self, obj_feats, pairs, confidence):
        '''
        objects : Tensor of all objects id, (O, )
        triplets : Tensor of all relations [subject, relation, object], (R, 3)
        '''
        # Try to do GCN with single data a time(not a batch)
        for i in range(self.gconv_layers):
            obj_feats, pairs, confidence = self.gconv(obj_feats, 
                pairs, confidence)
        return obj_feats
class RelationNet(nn.Module):
    # single layer
    def __init__(self, feature_dim=2048,w2v_dim=300,output_class=2, hidden_dim=1024):
        super(RelationNet, self).__init__()
        self.output_class = output_class
        self.vpairnet = PairNet(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.wpairnet = PairNet(feature_dim=w2v_dim, hidden_dim=hidden_dim)
        self.bg_classifier = nn.Sequential(nn.Linear(hidden_dim, 1),
        #self.bg_classifier = nn.Sequential(nn.Linear(hidden_dim, 2),
            #nn.BatchNorm1d(1),
            #nn.Sigmoid()
        )
        
        self.bg_classifier.apply(_init_weight)
        #self.rel_classifier.apply(_init_weight)
    def forward(self, feats, w2v, pairs, ious):
        '''
        feats : Tensor of all objects id, (O, )
        pairs : Tensor of all relations [subject, object], (F, )
        '''
        vfeat = self.vpairnet(feats, pairs, ious)
        wfeat = self.wpairnet(w2v, pairs, ious)
        feat = vfeat + wfeat
        bg_out = self.bg_classifier(feat)
        
        return bg_out
# Non-symmetric
class PairNet(nn.Module):
    # single layer
    def __init__(self, feature_dim=2048, hidden_dim=1024):
        super(PairNet, self).__init__()
        self.feature_dim = feature_dim
        self.pairnet = nn.Sequential(nn.Linear(self.feature_dim*2+1, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.pairnet.apply(_init_weight)
    def forward(self, feats, pairs, ious):
        '''
        feats : Tensor of all objects id, (O, )
        pairs : Tensor of all relations [subject, object], (F, )
        '''
        input_feats = feats[pairs.long()]
        ious = ious.view(ious.shape[0],1)
        input_feats = torch.cat((input_feats[:,0],input_feats[:,1],ious),-1)
        if input_feats.shape[0] == 0 or input_feats.shape[1] != (self.feature_dim*2+1):
            ipdb.set_trace()
        hidden = self.pairnet(input_feats)
        
        return hidden
# Symmetric
class EmbRelationNet(nn.Module):
    # single layer
    def __init__(self, feature_dim=2048,w2v_dim=300,output_class=2, hidden_dim=1024):
        super(EmbRelationNet, self).__init__()
        self.output_class = output_class
        self.vpairnet = EmbPairNet(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.wpairnet = EmbPairNet(feature_dim=w2v_dim, hidden_dim=hidden_dim)
        self.bg_classifier = nn.Sequential(nn.Linear(hidden_dim, 1),
        )
        
        self.bg_classifier.apply(_init_weight)
        #self.rel_classifier.apply(_init_weight)
    def forward(self, feats, w2v, pairs, ious):
        '''
        feats : Tensor of all objects id, (O, )
        pairs : Tensor of all relations [subject, object], (F, )
        '''
        vfeat = self.vpairnet(feats, pairs, ious)
        wfeat = self.wpairnet(w2v, pairs, ious)
        feat = vfeat + wfeat
        bg_out = self.bg_classifier(feat)
        #bg_out = feat.sum(1)
        
        return bg_out
# Symmetric
class EmbPairNet(nn.Module):
    # single layer
    def __init__(self, feature_dim=2048, hidden_dim=1024):
        super(EmbPairNet, self).__init__()
        self.feature_dim = feature_dim
        self.emb = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
        )
        
        self.emb.apply(_init_weight)
    def forward(self, feats, pairs, ious):
        '''
        feats : Tensor of all objects id, (O, )
        pairs : Tensor of all relations [subject, object], (F, )
        '''
        #input_feats = feats[pairs.long()]
        #ious = ious.view(ious.shape[0],1)
        max_idx = pairs.max()
        input_feats = feats[:max_idx+1]
        new_feats = self.emb(input_feats)
        new_feats2 = new_feats[pairs.long()]
        hidden = new_feats2[:,0]*new_feats2[:,1]
        #hidden = new_feats2[:,0]+new_feats2[:,1]
        
        return hidden
if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
