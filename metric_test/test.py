import pickle as pk
import sys
sys.path.append('/shared_home/ylc/im2p/coco-caption/pycocoevalcap/')
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
gts = pk.load(open('gts.pk','r'))
res = pk.load(open('res.pk','r'))
meteor = Meteor()
cider = Cider()
bleu = Bleu(4)
import ipdb
ipdb.set_trace()
