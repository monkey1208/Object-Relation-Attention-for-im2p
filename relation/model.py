
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RelationNet(nn.Module):
    # single layer
    def __init__(self, feature_dim=2048,w2v_dim=300,output_class=2, hidden_dim=1024):
        super(RelationNet, self).__init__()
        self.output_class = output_class
        self.vpairnet = PairNet(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.wpairnet = PairNet(feature_dim=w2v_dim, hidden_dim=hidden_dim)
        self.bg_classifier = nn.Sequential(nn.Linear(hidden_dim, 1),
            #nn.Sigmoid()
        )
        #self.rel_classifier = nn.Sequential(nn.Linear(hidden_dim, output_class),
        #    nn.Sigmoid()
        #)
        
        self.bg_classifier.apply(_init_weight)
        # self.rel_classifier.apply(_init_weight)
    def forward(self, feats, w2v, pairs, ious):
        '''
        feats : Tensor of all objects id, (O, )
        pairs : Tensor of all relations [subject, object], (F, )
        '''
        vfeat = self.vpairnet(feats, pairs, ious)
        wfeat = self.wpairnet(w2v, pairs, ious)
        feat = vfeat + wfeat
        bg_out = self.bg_classifier(feat)
        #rel_out = self.rel_classifier(feat)
        
        #return bg_out, rel_out
        return bg_out
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
        input_feats = torch.cat((input_feats[:,0],input_feats[:,1],ious),-1)
        hidden = self.pairnet(input_feats)
        
        return hidden
        
if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
