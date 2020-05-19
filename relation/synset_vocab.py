import os
def get_vocab2idx(fname, zero_start=True):
    tmp = {}
    tmp2 = {}
    with open(fname, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            vocab = line.strip().split(',')
            for voc in vocab:
                if zero_start:
                    tmp[voc] = idx
                    tmp2[idx] = vocab
                else:
                    tmp[voc] = idx + 1 # starts from 1, 0 is bg
                    tmp2[idx + 1] = vocab
    return tmp, tmp2
            
def load_vocab_index(dirname):
    obj_fname = os.path.join(dirname, 'objects_vocab.txt')
    attr_fname = os.path.join(dirname, 'attributes_vocab.txt')
    rel_fname = os.path.join(dirname, 'relations_vocab.txt')
    obj2idx, idx2obj = get_vocab2idx(obj_fname, zero_start=False) # 1600
    attr2idx, idx2attr = get_vocab2idx(attr_fname) # 400
    rel2idx, idx2rel = get_vocab2idx(rel_fname) # 20
    return obj2idx, attr2idx, rel2idx, idx2obj, idx2attr, idx2rel
dir_name = '/shared_home/ylc/vg_data/1600-400-20'
print('load vocab from {}'.format(dir_name))
obj2idx, attr2idx, rel2idx, idx2obj, idx2attr, idx2rel = load_vocab_index(dir_name)
OBJ_NUM = len(idx2obj)
ATTR_NUM = len(idx2rel)
REL_NUM = len(idx2rel)
