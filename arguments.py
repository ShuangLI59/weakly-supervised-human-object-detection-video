import numpy as np
import os
import cv2
import re
import pdb
import argparse
import torch
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Weakly Supervised Human-Object Interaction Detection in Video via Contrastive Spatiotemporal Regions')

    parser.add_argument('--dataset', dest='dataset', required=True,help='Dataset to use')
    parser.add_argument('--cfg', dest='cfg_file', required=True,help='Config file for training (and optionally testing)')
    parser.add_argument('--set', dest='set_cfgs',help='Set config keys. Key value sequence seperate by whitespace.''e.g. [key] [value] [key] [value]',default=[], nargs='+')

    parser.add_argument('--num_gpus', default=8, type=int)
    parser.add_argument('--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')
    parser.add_argument('--use_tfboard', help='Use tensorflow tensorboard to log training info',action='store_true')

    # Training
    parser.add_argument('--iter_size',help='Update once every iter_size steps, as in Caffe.',default=1, type=int)
    parser.add_argument('--o', dest='optimizer', help='Training optimizer.',default=None)
    parser.add_argument('--lr', help='Base learning rate.',default=None, type=float)
    parser.add_argument('--lr_decay_gamma',help='Learning rate decay rate.',default=None, type=float)
    parser.add_argument('--backbone_lr_scalar',default=0.01, type=float)
    
    # Data
    parser.add_argument('--video_frame',default=12, type=int)
    parser.add_argument('--nw', dest='num_workers',help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',type=int)


    # Model
    parser.add_argument('--roi_box_head',default='fast_rcnn_heads.roi_2mlp_head', type=str)
    parser.add_argument('--dropout',default=0.0, type=float)


    parser.add_argument('--obj_loss',default='contrastive', type=str, choices=['npair_objloss', 'contrastive_objloss'])

    parser.add_argument('--com_weight',default='cat', type=str, choices=['add', 'cat', 'cat_video_soft_attention'])
    parser.add_argument('--human_obj_spatial',help='binary loss',action='store_true')

    parser.add_argument('--weight_reg',default='L2', type=str)
    parser.add_argument('--l2_weight',default=1.0, type=float)

    parser.add_argument('--video_loss',default='all', type=str, choices=['contrastive_nearest_neighbor_plus', 'npair_nearest_neighbor_plus', 'npair_nearest_neighbor', 'npair_max', 'npair_max_plus', 'contrastive_nearest_neighbor', 'contrastive_max', 'contrastive_max_plus', 'none'])
    parser.add_argument('--video_weight',default=1.0, type=float)
    
    parser.add_argument('--binary_loss',help='binary loss',action='store_true')


	## Eval
    parser.add_argument('--eval_subset',default='', type=str)
    parser.add_argument('--eval_map', help='default',action='store_true')

    # Epoch
    parser.add_argument('--start_step',help='Starting step count for training epoch. 0-indexed.',default=0, type=int)
    parser.add_argument('--disp_interval',help='Display training info every N iterations',default=20, type=int)
    
    parser.add_argument('--no_save', help='do not save anything', action='store_true')
    parser.add_argument('--out_dir', help='output name',default='debug',type=str)
    
    # Others
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--resume',help='resume to training on a checkpoint',action='store_true')
    parser.add_argument('--load_ckpt', help='checkpoint path to load')
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')
    parser.add_argument('--vis', help='visulize boxes', action='store_true')
    parser.add_argument('--vis_video', help='visulize video boxes', action='store_true')
    
    parser.add_argument('--iou',default=0.3, type=float)

    return parser.parse_args()


def set_configs(args):
	if not torch.cuda.is_available():
		sys.exit("Need a CUDA device to run the code.")

	if args.cuda or cfg.NUM_GPUS > 0:
	    cfg.CUDA = True
	else:
	    raise ValueError("Need Cuda device to run !")

	if args.dataset == "vhico":
	    cfg.TRAIN.DATASETS = ('vhico_train',)
	    cfg.VAL.DATASETS = ('vhico_val',)
	    cfg.TEST.DATASETS = ('vhico_test',)
	    cfg.UNSEEN.DATASETS = ('vhico_unseen',)
	else:
	    raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

	cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
	    cfg_from_list(args.set_cfgs)


	if args.debug:
	    cfg.DEBUG = True
	
	cfg.NUM_GPUS = args.num_gpus
	cfg.VIDEO_FRAME = args.video_frame
	
	cfg.SOLVER.BACKBONE_LR_SCALAR = args.backbone_lr_scalar

	cfg.FAST_RCNN.ROI_BOX_HEAD = args.roi_box_head
	cfg.DROPOUT = args.dropout

	cfg.OBJ_LOSS = args.obj_loss
	cfg.COM_WEIGHT = args.com_weight
	cfg.HUMAN_OBJ_SPATIAL = args.human_obj_spatial
	cfg.WEIGHT_REG = args.weight_reg
	cfg.L2_WEIGHT = args.l2_weight
	cfg.VIDEO_LOSS = args.video_loss
	cfg.VIDEO_WEIGHT = args.video_weight
	cfg.BINARY_LOSS = args.binary_loss
	
	cfg.EVAL_SUBSET = args.eval_subset
	cfg.EVAL_MAP = args.eval_map
	
	cfg.SAVE_MODEL_ITER = 100
	cfg.IOU = args.iou
	cfg.vis = args.vis
	cfg.vis_video = args.vis_video
	cfg.load_ckpt = args.load_ckpt

	### Overwrite some solver settings from command line arguments    
	if args.optimizer is not None:
	    cfg.SOLVER.TYPE = args.optimizer
	if args.lr is not None:
	    cfg.SOLVER.BASE_LR = args.lr
	if args.lr_decay_gamma is not None:
	    cfg.SOLVER.GAMMA = args.lr_decay_gamma
	assert_and_infer_cfg()

	return cfg

