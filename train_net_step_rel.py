import os
import sys
import cv2
import pdb
import yaml
import random
import pickle
import logging
import resource
import traceback
import numpy as np
import os.path as osp
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable


cv2.setNumThreads(0)
sys.path.insert(0, 'lib')
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from datasets.roidb_rel import combined_roidb_for_training
from roi_data.loader_rel import RoiDataLoader, collate_minibatch
from modeling.model_builder_rel import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats_rel import TrainingStats
from utils.val_stats_rel import ValStats
from utils.test_stats_rel import TestStats
from test_net_step_rel import run_eval
from arguments import parse_args, set_configs

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(10)

# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))



def save_ckpt(output_dir, args, step, batch_size, model, optimizer, is_best, best_total_loss):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if is_best:
        save_name = os.path.join(ckpt_dir, 'best_model.pth')
    else:
        save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))

    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'batch_size': batch_size,
        'model': model.state_dict(),
        'best_total_loss': best_total_loss,
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)



def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    cfg = set_configs(args)
    timers = defaultdict(Timer)


    ### --------------------------------------------------------------------------------
    ### Dataset Training ###
    ### --------------------------------------------------------------------------------
    timers['roidb_training'].tic()
    roidb_training, ratio_list_training, ratio_index_training, category_to_id_map, prd_category_to_id_map = combined_roidb_for_training(cfg.TRAIN.DATASETS)
    timers['roidb_training'].toc()
    roidb_size_training = len(roidb_training)
    logger.info('{:d} training roidb entries'.format(roidb_size_training))
    logger.info('Takes %.2f sec(s) to construct training roidb', timers['roidb_training'].average_time)


    batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH

    dataset_training = RoiDataLoader(
        roidb_training,
        cfg.MODEL.NUM_CLASSES,
        training=True,
        dataset=cfg.TRAIN.DATASETS)
    dataloader_training = torch.utils.data.DataLoader(
        dataset_training,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch,
        shuffle=True,
        drop_last=True)
    dataiterator_training = iter(dataloader_training)

    ### --------------------------------------------------------------------------------
    ### Dataset Validation ###
    ### --------------------------------------------------------------------------------
    timers['roidb_val'].tic()
    roidb_val, ratio_list_val, ratio_index_val, _, _ = combined_roidb_for_training(cfg.VAL.DATASETS)
    timers['roidb_val'].toc()
    roidb_size_val = len(roidb_val)
    logger.info('{:d} val roidb entries'.format(roidb_size_val))
    logger.info('Takes %.2f sec(s) to construct val roidb', timers['roidb_val'].average_time)

    dataset_val = RoiDataLoader(
        roidb_val,
        cfg.MODEL.NUM_CLASSES,
        training=False,
        dataset=cfg.VAL.DATASETS)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch,
        drop_last=True)

    ### --------------------------------------------------------------------------------
    ### Dataset Test ###
    ### --------------------------------------------------------------------------------
    timers['roidb_test'].tic()
    roidb_test, ratio_list_test, ratio_index_test, _, _ = combined_roidb_for_training(cfg.TEST.DATASETS)
    timers['roidb_test'].toc()
    roidb_size_test = len(roidb_test)
    logger.info('{:d} test roidb entries'.format(roidb_size_test))
    logger.info('Takes %.2f sec(s) to construct test roidb', timers['roidb_test'].average_time)

    dataset_test = RoiDataLoader(
        roidb_test,
        cfg.MODEL.NUM_CLASSES,
        training=False,
        dataset=cfg.TEST.DATASETS)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch,
        drop_last=True)



    ### --------------------------------------------------------------------------------
    ### Dataset Unseen ###
    ### --------------------------------------------------------------------------------
    if args.dataset=='vhico':
        timers['roidb_unseen'].tic()
        roidb_unseen, ratio_list_unseen, ratio_index_unseen, _, _ = combined_roidb_for_training(cfg.UNSEEN.DATASETS)
        timers['roidb_unseen'].toc()
        roidb_size_unseen = len(roidb_unseen)
        logger.info('{:d} test unseen roidb entries'.format(roidb_size_unseen))
        logger.info('Takes %.2f sec(s) to construct test roidb', timers['roidb_unseen'].average_time)

        dataset_unseen = RoiDataLoader(
            roidb_unseen,
            cfg.MODEL.NUM_CLASSES,
            training=False,
            dataset=cfg.UNSEEN.DATASETS)
        dataloader_unseen = torch.utils.data.DataLoader(
            dataset_unseen,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            collate_fn=collate_minibatch,
            drop_last=True)


    ### --------------------------------------------------------------------------------
    ### Model ###
    ### --------------------------------------------------------------------------------
    maskRCNN = Generalized_RCNN(category_to_id_map=category_to_id_map, prd_category_to_id_map=prd_category_to_id_map, args=args)
    if cfg.CUDA:
        maskRCNN.cuda()

    
    ### --------------------------------------------------------------------------------
    ### Optimizer ###
    # record backbone params, i.e., conv_body and box_head params
    ### --------------------------------------------------------------------------------
    gn_params = []
    backbone_bias_params = []
    backbone_bias_param_names = []
    prd_branch_bias_params = []
    prd_branch_bias_param_names = []
    backbone_nonbias_params = []
    backbone_nonbias_param_names = []
    prd_branch_nonbias_params = []
    prd_branch_nonbias_param_names = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'gn' in key:
                gn_params.append(value)
            elif 'Conv_Body' in key or 'Box_Head' in key or 'Box_Outs' in key or 'RPN' in key:
                if 'bias' in key:
                    backbone_bias_params.append(value)
                    backbone_bias_param_names.append(key)
                else:
                    backbone_nonbias_params.append(value)
                    backbone_nonbias_param_names.append(key)
            else:
                if 'bias' in key:
                    prd_branch_bias_params.append(value)
                    prd_branch_bias_param_names.append(key)
                else:
                    prd_branch_nonbias_params.append(value)
                    prd_branch_nonbias_param_names.append(key)
    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': backbone_nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': backbone_bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': prd_branch_nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': prd_branch_bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)


    ### --------------------------------------------------------------------------------
    ### Load checkpoint
    ### --------------------------------------------------------------------------------
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        print('--------------------------------------------------------------------------------')
        print('loading checkpoint %s' % load_name)
        print('--------------------------------------------------------------------------------')

        if args.resume:
            print('resume')
            args.start_step = checkpoint['step'] + 1
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print('args.load_ckpt', args.load_ckpt)

    
    lr = optimizer.param_groups[2]['lr']  # lr of non-backbone parameters, for commmand line outputs.
    backbone_lr = optimizer.param_groups[0]['lr']  # lr of backbone parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True)


    ### --------------------------------------------------------------------------------
    ### Training Setups ###
    ### --------------------------------------------------------------------------------
    args.run_name = args.out_dir
    output_dir = misc_utils.get_output_dir(args, args.out_dir)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            tblogger = SummaryWriter(output_dir)



    ### --------------------------------------------------------------------------------
    ### Training Loop ###
    ### --------------------------------------------------------------------------------
    maskRCNN.train()

    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None,
        True)

    val_stats = ValStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None,
        False)

    test_stats = TestStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None,
        False)
    
    best_total_loss = np.inf
    best_eval_result = 0




    ### --------------------------------------------------------------------------------
    ### EVAL ###
    ### --------------------------------------------------------------------------------
    if cfg.EVAL_SUBSET == 'unseen':
        print('testing unseen ...')
        is_best, best_eval_result = run_eval(args, cfg, maskRCNN, dataloader_unseen, step=0, output_dir=output_dir, test_stats=test_stats, best_eval_result=best_eval_result, eval_subset=cfg.EVAL_SUBSET)
        return
    elif cfg.EVAL_SUBSET == 'test':
        print('testing ...')
        is_best, best_eval_result = run_eval(args, cfg, maskRCNN, dataloader_test, step=0, output_dir=output_dir, test_stats=test_stats, best_eval_result=best_eval_result, eval_subset=cfg.EVAL_SUBSET)
        return


    ### --------------------------------------------------------------------------------
    ### TRAIN ###
    ### --------------------------------------------------------------------------------
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):
            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate_rel(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            #########################################################################################################################
            ## train
            #########################################################################################################################
            training_stats.IterTic()
            optimizer.zero_grad()

            for inner_iter in range(args.iter_size):
                try:
                    input_data = next(dataiterator_training)
                except StopIteration:
                    print('recurrence data loader')
                    dataiterator_training = iter(dataloader_training)
                    input_data = next(dataiterator_training)

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))


                net_outputs = maskRCNN(**input_data)
                
                training_stats.UpdateIterStats(net_outputs['gt_label'], inner_iter)
                loss = net_outputs['gt_label']['total_loss']
                loss.backward()

            optimizer.step()
            training_stats.IterToc()
            training_stats.LogIterStats(step, lr, backbone_lr)


            if (step+1) % cfg.SAVE_MODEL_ITER == 0:
                save_ckpt(output_dir, args, step, batch_size, maskRCNN, optimizer, False, best_total_loss)

        # ---- Training ends ----
        save_ckpt(output_dir, args, step, batch_size, maskRCNN, optimizer, False, best_total_loss)


    except (RuntimeError, KeyboardInterrupt):
        del dataiterator_training
        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir, args, step, batch_size, maskRCNN, optimizer, False, best_total_loss)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


if __name__ == '__main__':
    main()
