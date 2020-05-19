python train.py \
    --batch_size 10 \
    --input_att_dir '/2t/ylc/VG/parabu_att' \
    --input_fc_dir '/2t/ylc/VG/parabu_fc' \
    --sg_dir '/shared_home/ylc/vg_data/data_tools/1600-50' \
    --w2v '/2t/ylc/word/w2v.npy' \
    --para_json '/2t/ylc/VG/paragraphs_v1.json' \
    --input_json '/2t/ylc/image-paragraph-captioning/data/paratalk.json' \
    --input_label_h5 '/2t/ylc/image-paragraph-captioning/data/paratalk_label.h5' \
    --language_eval 1 \
    --learning_rate 5e-4 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --max_epochs 100 \
    --rnn_type 'lstm' \
    --val_images_use 5000 \
    --save_checkpoint_every 3000 \
    --checkpoint_path '/2t/ylc/im2p/log_xe_noatt_norp(resume)/' \
    --id 'xe' \
    --print_freq 200 \
    --fc_feat_size 2048 \
    --att_feat_size 2048 \
    --gcn_layers 1 \
    --no-avg-feat \
    --start_from '/2t/ylc/im2p/log_xe_noatt_norp/' \
    --load_gcn '/2t/ylc/im2p/log_xe_noatt_norp/gcn_model.pth' \
    --load_model '/2t/ylc/im2p/log_xe_noatt_norp/model.pth' \
    --pretrain_rel '/2t/ylc/im2p/log_xe_noatt_norp/rel_model.pth'
