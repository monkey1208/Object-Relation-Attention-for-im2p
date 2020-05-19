python visualize.py \
    --input_att_dir '/2t/ylc/VG/parabu_att' \
    --input_fc_dir '/2t/ylc/VG/parabu_fc' \
    --sg_dir '/shared_home/ylc/vg_data/data_tools/1600-50' \
    --w2v '/2t/ylc/word/w2v.npy' \
    --para_json '/2t/ylc/VG_paragraph/paragraphs_v1.json' \
    --input_json '/2t/ylc/image-paragraph-captioning/data/paratalk.json' \
    --input_label_h5 '/2t/ylc/image-paragraph-captioning/data/paratalk_label.h5' \
    --batch_size 10 \
    --learning_rate 5e-5 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --save_checkpoint_every 4000 \
    --language_eval 1 \
    --val_images_use 5000 \
    --max_epochs 200 \
    --self_critical_after 0 \
    --cached_tokens para-train-idxs \
    --cider_reward_weight 1 \
    --block_trigrams 1 \
    --fc_feat_size 2048 \
    --att_feat_size 2048 \
    --print_freq 200 \
    --load_model '/2t/ylc/im2p/log_sc_rel_test_top50_from2/model-best-i114000-score0.2987.pth' \
    --load_gcn '/2t/ylc/im2p/log_sc_rel_test_top50_from2/gcn-best-i114000-score0.2987.pth' \
    --pretrain_rel '/2t/ylc/im2p/log_sc_rel_test_top50_from2/rel-best-i114000-score0.2987.pth' \
    --do-avg-feat
#--caption_model topdown \
    #--load_model '/2t/ylc/im2p/log_sc_rel_test_top50_2_meteor5bleu5/model-best-i120000-score0.3365.pth' \
    #--load_gcn '/2t/ylc/im2p/log_sc_rel_test_top50_2_meteor5bleu5/gcn-best-i120000-score0.3365.pth' \
    #--pretrain_rel '/2t/ylc/im2p/log_sc_rel_test_top50_2_meteor5bleu5/rel-best-i120000-score0.3365.pth' \
