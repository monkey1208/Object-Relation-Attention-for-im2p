import json
import ipdb
def print_cnt(pairs):
    pair_cnt = {}
    for pair in pairs:
        pair = (pair[0], pair[1])
        if pair not in pair_cnt.keys():
            pair_cnt[pair] = 1
        else:
            pair_cnt[pair] += 1
    import operator
    pair_cnt = sorted(pair_cnt.items(), key=operator.itemgetter(1))
    print(pair_cnt)
#datas = json.load(open('vis_result.json','r'))
datas = json.load(open('fail_result.json','r'))
for image_id, data in datas.items():
    caption = data['caption']
    cider = data['cider']
    bleu = data['bleu']
    boxes = json.loads(data['boxes'])
    pairs = json.loads(data['pair'])
    #image_id = data['image_id']
    score = (cider, bleu)
    print(image_id)
    print(score)
    print(caption)
    datalen = data['datalen']
    pairs = json.loads(data['pair'])
    ipdb.set_trace()
