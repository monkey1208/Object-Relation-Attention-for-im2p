python eval.py  --image_json /2t/ylc/VG/image_data.json \
    --para_json /2t/ylc/VG/paragraphs_v1.json \
    --sg_dir ~/vg_data/data_tools/1600-50/ \
    --split_dir ~/vg_data/para_split/ \
    --input_att_dir /2t/ylc/VG/obj_feat/ \
    --device 1 \
    --model_path model/ep9-acc86.26.pt \
    --w2v /2t/ylc/word/w2v.npy
