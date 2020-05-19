from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from sg.GCN import RelationNet, BGGCN, EmbRelationNet, EmbBGGCN
from dataloader import *
from relloader import getloader
import eval_utils
import misc.utils as utils
import misc.rel_utils as rutils
from misc.rewards import init_scorer, get_self_critical_reward
import ipdb
from relation.iteration import rel_iteration, avg_features#, noattn_rel_iteration

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    #relationloader = getloader(opt)
    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    if not os.path.exists(opt.checkpoint_path):
        os.mkdir(opt.checkpoint_path)
    print('Check point path : {}'.format(opt.checkpoint_path))
    flog = open(os.path.join(opt.checkpoint_path, 'log.csv'), 'w')
    flog.write('epoch_iteration,CIDEr,METEOR,ROUGE_L,BLEU-1,BLEU-2,BLEU-3,BLEU-4\n')

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Create model
    torch.manual_seed(211)
    #torch.manual_seed(1208)
    #opt.att_feat_size += 300
    #if opt.avg_feat:
    #    opt.fc_feat_size += 300
    model = models.setup(opt).cuda()
    #opt.att_feat_size -= 300
    #if opt.avg_feat:
    #    opt.fc_feat_size -= 300
    dp_model = torch.nn.DataParallel(model)
    dp_model.train()
    torch.manual_seed(1208)

    rel_model = RelationNet(feature_dim=opt.att_feat_size, w2v_dim=300)
    #rel_model = EmbRelationNet(feature_dim=opt.att_feat_size, w2v_dim=300)
    
    #torch.manual_seed(128)
    torch.manual_seed(211)
    gcn_model = BGGCN(gconv_layers=opt.gcn_layers ,
        hidden_dim=opt.gcn_hidden, feat_dim=opt.att_feat_size, w2v_dim=300).cuda()
    #gcn_model = EmbBGGCN(gconv_layers=opt.gcn_layers ,
    #    hidden_dim=opt.gcn_hidden, feat_dim=opt.att_feat_size).cuda()
    print('GCN config\nlayers : {}, hidden_size : {}'.format(opt.gcn_layers, 
        opt.gcn_hidden))
    print('Do avg feat : {}'.format(opt.avg_feat))
    if opt.start_from is not None:
        # load gcn model
        # need to remove module. in the state dict keys
        gcn_model = utils.load_model(gcn_model,os.path.join(opt.start_from, opt.load_gcn))
    if opt.pretrain_rel is not None:
        rel_model = utils.load_model(rel_model,opt.pretrain_rel)
    #gcn_model = torch.nn.DataParallel(gcn_model.cuda())
    gcn_model = gcn_model.cuda()
    gcn_model.train()
    rel_model = rel_model.cuda()
    rel_model.train()

    # Loss function
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    # Optimizer and learning rate adjustment flag
    optimizer = utils.build_optimizer(list(model.parameters())+list(gcn_model.parameters())+list(rel_model.parameters()), opt)
    #optimizer = utils.build_optimizer(list(model.parameters())+list(gcn_model.parameters()), opt)
    update_lr_flag = True

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # Training loop
    while True:

        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            print('current lr = {}'.format(opt.current_lr))

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')
        data_time = time.time() - start
        start = time.time()

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        # get image idx to index scene graph
        #relationloader = getloader(opt)
        #triplets2, tri_len2 = rutils.get_triplet(data, relationloader)
        
        new_att_feats = rel_iteration(data, att_feats, rel_model, gcn_model)
        #new_att_feats = noattn_rel_iteration(data, att_feats, gcn_model)
        #new_att_feats = emb_rel_iteration(data, att_feats, rel_model, gcn_model)
        # new_att_feats = gcn_iteration(data, att_feats, gcn_model)
        # Forward pass and loss
        optimizer.zero_grad()
        # Graph Convolution
        #at = gcn_model(att_feats, triplets2, tri_len2)
        #att_feats = gcn_model(att_feats, triplets, tri_len)
        att_feats = new_att_feats
        # Avg fc feats
        fc_feats = avg_features(fc_feats, att_feats, att_masks, opt.avg_feat)
        '''
        mask_sum = att_masks.sum(1)
        for i in range(mask_sum.shape[0]):
            assert att_masks[i,:mask_sum[i].int().item()].sum() == mask_sum[i]
            #fc_feats[i] = att_feats[i,:mask_sum[i].int().item()].mean(0)
           fc_feats[i] = att_feats[i,:mask_sum[i].int().item(),:opt.fc_feat_size].mean(0)
        '''

        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        # Backward pass
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()

        # Print 
        total_time = time.time() - start
        if iteration % opt.print_freq == 1:
            print('Read data:', time.time() - start)
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, data_time, total_time))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, np.mean(reward[:,0]), data_time, total_time))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # Validate and save model 
        if (iteration % opt.save_checkpoint_every == 0):

            # Evaluate model
            #eval_kwargs = {'split': 'val',
            #                'dataset': opt.input_json}
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, gcn_model, rel_model, opt.avg_feat, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Our metric is CIDEr if available, otherwise validation loss
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
            print('Val loss {}'.format(val_loss))
            flog.write('{}-{},{},{},{},{},{},{},{}\n'.format(epoch, iteration,
                lang_stats['CIDEr'], lang_stats['METEOR'], lang_stats['ROUGE_L'],
                lang_stats['Bleu_1'], lang_stats['Bleu_2'],
                lang_stats['Bleu_3'], lang_stats['Bleu_4']))
            flog.flush()
            # Save model in checkpoint path 
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            gcn_path = os.path.join(opt.checkpoint_path, 'gcn_model.pth')
            torch.save(gcn_model.state_dict(), gcn_path)
            rel_path = os.path.join(opt.checkpoint_path, 'rel_model.pth')
            torch.save(rel_model.state_dict(), rel_path)
            print("model saved to {}".format(checkpoint_path))
            print("gcn model saved to {}".format(gcn_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            # Save model to unique file if new best model
            if best_flag or (lang_stats['Bleu_1']>0.436 and lang_stats['Bleu_2']>0.275 and lang_stats['Bleu_3']>0.174 and lang_stats['Bleu_3']>0.106 and lang_stats['CIDEr']>0.307) or (lang_stats['METEOR']>0.179):
                model_fname = 'model-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                infos_fname = 'model-best-i{:05d}-infos.pkl'.format(iteration)
                gcn_fname = 'gcn-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                rel_fname = 'rel-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                checkpoint_path = os.path.join(opt.checkpoint_path, model_fname)
                torch.save(model.state_dict(), checkpoint_path)
                gcn_path = os.path.join(opt.checkpoint_path, gcn_fname)
                torch.save(gcn_model.state_dict(), gcn_path)
                rel_path = os.path.join(opt.checkpoint_path, rel_fname)
                torch.save(rel_model.state_dict(), rel_path)
                print("model saved to {}".format(checkpoint_path))
                print("gcn_model saved to {}".format(gcn_path))
                with open(os.path.join(opt.checkpoint_path, infos_fname), 'wb') as f:
                    cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
