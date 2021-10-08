import os
import pdb
import cv2
import h5py
import math
import copy
import gensim
import json
import pickle
import random
import logging
import importlib
import numpy as np
from copy import deepcopy
from functools import wraps
from numpy import linalg as la

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.relpn_heads as relpn_heads
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.net as net_utils
import utils.fpn as fpn_utils

import utils.resnet_weights_helper as resnet_utils
from utils.detectron_weight_helper import load_detectron_weight

import modeling.reldn_heads as reldn_heads
from datasets.dataset_catalog_rel import WORD2VEC_GOOGLE, WORD2VEC_TEST, WORD2VEC_UNSEEN, TRIPLET_TEST, TRIPLET_UNSEEN, TRIPLET_TRAIN
from datasets.dataset_catalog_rel import NUM_OBJ, NUM_PRD, NUM_OBJ_UNSEEN, NUM_PRD_UNSEEN

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def get_obj_prd_vecs(dataset_name, category_to_id_map, prd_category_to_id_map):
    if 'vhico' in dataset_name:
        if cfg.EVAL_SUBSET == 'unseen':
            obj_prd_w2v_dir = WORD2VEC_UNSEEN
        else:
            obj_prd_w2v_dir = WORD2VEC_TEST
    print('load word2vec %s' % obj_prd_w2v_dir)

    obj_cats = list(category_to_id_map[0].keys())
    prd_cats = list(prd_category_to_id_map[0].keys())

    hf = h5py.File(obj_prd_w2v_dir, 'r')
    all_obj_vecs = hf.get('all_obj_vecs')
    all_prd_vecs = hf.get('all_prd_vecs')

    all_obj_vecs = np.asarray(all_obj_vecs)
    all_prd_vecs = np.asarray(all_prd_vecs)

    return all_obj_vecs, all_prd_vecs




def collect_output(cfg, dataset_name, im_info, roidb, obj_rois, gt_obj_label, gt_prd_label, output, device_id):
    obj_loss, prd_loss, weight_loss, weight_human_loss, video_loss, video_binary_loss, roi_weights_unpacked, roi_weights_human_unpacked, densepose_roi, roi_weights_ori, roi_weights_human_ori, cls_prediction = output

    return_dict = {}
    
    if (dataset_name == cfg.TEST.DATASETS[0]) or (dataset_name == cfg.UNSEEN.DATASETS[0]):
        cls_prediction['obj_label'] = gt_obj_label.view(-1, cfg.VIDEO_FRAME, 1).detach().cpu().numpy()
        cls_prediction['prd_label'] = gt_prd_label.view(-1, cfg.VIDEO_FRAME, 1).detach().cpu().numpy()

        im_info_unpacked = im_info.view(-1, cfg.VIDEO_FRAME, 3).detach().cpu().numpy()
        obj_rois_unpacked = torch.tensor(obj_rois).view(-1, cfg.VIDEO_FRAME, cfg.TRAIN.BATCH_SIZE_PER_IM, 5)
        obj_rois_unpacked = obj_rois_unpacked.detach().cpu().numpy()

        roi_weights_unpacked = roi_weights_unpacked.view(-1, cfg.VIDEO_FRAME, cfg.TRAIN.BATCH_SIZE_PER_IM)
        roi_weights_unpacked = roi_weights_unpacked.detach().cpu().numpy()
        roi_weights_human_unpacked = roi_weights_human_unpacked.view(-1, cfg.VIDEO_FRAME, cfg.MAX_NUM_HUMAN)
        roi_weights_human_unpacked = roi_weights_human_unpacked.detach().cpu().numpy()

        roi_weights_ori = roi_weights_ori.view(-1, cfg.VIDEO_FRAME, cfg.TRAIN.BATCH_SIZE_PER_IM)
        roi_weights_ori = roi_weights_ori.detach().cpu().numpy()            
        roi_weights_human_ori = roi_weights_human_ori.view(-1, cfg.VIDEO_FRAME, cfg.MAX_NUM_HUMAN)
        roi_weights_human_ori = roi_weights_human_ori.detach().cpu().numpy()                
        densepose_obj_rois_unpacked = torch.tensor(densepose_roi).view(-1, cfg.VIDEO_FRAME, cfg.MAX_NUM_HUMAN, 5)
        densepose_obj_rois_unpacked = densepose_obj_rois_unpacked.detach().cpu().numpy()


        predictions = {}
        predictions['files'] = []
        predictions['box'] = []
        predictions['score'] = []
        predictions['score_ori'] = []
        predictions['obj_gt_cls_name'] = []
        predictions['prd_gt_cls_name'] = []
        predictions['obj_gt_cls'] = []
        predictions['prd_gt_cls'] = []
        predictions['binary_pred'] = []
        
        human_predictions = {}
        human_predictions['files'] = []
        human_predictions['box'] = []
        human_predictions['score'] = []
        human_predictions['score_ori'] = []
        human_predictions['obj_gt_cls_name'] = []
        human_predictions['prd_gt_cls_name'] = []
        human_predictions['obj_gt_cls'] = []
        human_predictions['prd_gt_cls'] = []

        for test_i, obj_rois_unpacked_i in enumerate(obj_rois_unpacked):
            for test_j, obj_rois_unpacked_i_j in enumerate(obj_rois_unpacked_i):
                file_name = '/'.join(roidb[test_j]['file_name'].split('/')[-3:])
                this_obj_rois = obj_rois_unpacked_i_j / im_info_unpacked[test_i][test_j][2]
                prd_gt_cls_name = roidb[test_j]['prd_gt_cls_name']
                obj_gt_cls_name = roidb[test_j]['obj_gt_cls_name']

                prd_gt_cls = roidb[test_j]['prd_gt_cls']
                obj_gt_cls = roidb[test_j]['obj_gt_cls']

                # obj
                predictions['files'].append(file_name)
                predictions['box'].append(this_obj_rois)
                predictions['score'].append(roi_weights_unpacked[test_i][test_j])
                predictions['score_ori'].append(roi_weights_ori[test_i][test_j])
                predictions['obj_gt_cls_name'].append(obj_gt_cls_name)
                predictions['prd_gt_cls_name'].append(prd_gt_cls_name)
                predictions['obj_gt_cls'].append(obj_gt_cls)
                predictions['prd_gt_cls'].append(prd_gt_cls)
                if cfg.BINARY_LOSS:
                    predictions['binary_pred'].append(cls_prediction['binary_pred'][test_i])

                # human
                this_obj_rois = densepose_obj_rois_unpacked[test_i][test_j]  / im_info_unpacked[test_i][test_j][2]
                human_predictions['files'].append(file_name)
                human_predictions['box'].append(this_obj_rois)
                human_predictions['score'].append(roi_weights_human_unpacked[test_i][test_j])
                human_predictions['score_ori'].append(roi_weights_human_ori[test_i][test_j])
                human_predictions['obj_gt_cls_name'].append(obj_gt_cls_name)
                human_predictions['prd_gt_cls_name'].append(prd_gt_cls_name)
                human_predictions['obj_gt_cls'].append(obj_gt_cls)
                human_predictions['prd_gt_cls'].append(prd_gt_cls)
        # obj
        predictions['files'] = blob_utils.serialize(predictions['files'])
        predictions['obj_gt_cls_name'] = blob_utils.serialize(predictions['obj_gt_cls_name'])
        predictions['prd_gt_cls_name'] = blob_utils.serialize(predictions['prd_gt_cls_name'])
        predictions['obj_gt_cls'] = blob_utils.serialize(predictions['obj_gt_cls'])
        predictions['prd_gt_cls'] = blob_utils.serialize(predictions['prd_gt_cls'])
        return_dict['predictions'] = predictions

        # human
        human_predictions['files'] = blob_utils.serialize(human_predictions['files'])
        human_predictions['obj_gt_cls_name'] = blob_utils.serialize(human_predictions['obj_gt_cls_name'])
        human_predictions['prd_gt_cls_name'] = blob_utils.serialize(human_predictions['prd_gt_cls_name'])
        human_predictions['obj_gt_cls'] = blob_utils.serialize(predictions['obj_gt_cls'])
        human_predictions['prd_gt_cls'] = blob_utils.serialize(predictions['prd_gt_cls'])
        return_dict['human_predictions'] = human_predictions

    return_dict['losses'] = {}
    return_dict['losses']['obj_loss'] = obj_loss
    return_dict['losses']['prd_loss'] = prd_loss
    return_dict['losses']['weight_loss'] = weight_loss
    return_dict['losses']['weight_human_loss'] = weight_human_loss
    return_dict['losses']['video_loss'] = video_loss

    if cfg.BINARY_LOSS:
        return_dict['losses']['video_binary_loss'] = video_binary_loss
        
    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
        return_dict['losses'][k] = v.unsqueeze(0)
    
    return return_dict










class Generalized_RCNN(nn.Module):
    def __init__(self, category_to_id_map, prd_category_to_id_map, args=None):
        super().__init__()
        
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.category_to_id_map = category_to_id_map
        self.prd_category_to_id_map = prd_category_to_id_map
        self.args = args

        # -------------------------------------------------------------------------------------------------------------------------------
        # initialize word vectors
        # -------------------------------------------------------------------------------------------------------------------------------
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]
        self.obj_vecs, self.prd_vecs = get_obj_prd_vecs(ds_name, self.category_to_id_map, self.prd_category_to_id_map)

        # -------------------------------------------------------------------------------------------------------------------------------
        # Backbone for feature extraction
        # -------------------------------------------------------------------------------------------------------------------------------
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # -------------------------------------------------------------------------------------------------------------------------------
        # Region Proposal Network
        # -------------------------------------------------------------------------------------------------------------------------------
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]


        # -------------------------------------------------------------------------------------------------------------------------------
        # BBOX Branch
        # -------------------------------------------------------------------------------------------------------------------------------
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)

        # -------------------------------------------------------------------------------------------------------------------------------
        # RelPN
        # -------------------------------------------------------------------------------------------------------------------------------
        self.RelPN = relpn_heads.generic_relpn_outputs()

        # -------------------------------------------------------------------------------------------------------------------------------
        # RelDN
        # -------------------------------------------------------------------------------------------------------------------------------
        self.RelDN = reldn_heads.reldn_head(self.Box_Head.dim_out, self.obj_vecs, self.prd_vecs)
        self.reldn_heads = reldn_heads


        # -------------------------------------------------------------------------------------------------------------------------------
        # triplets
        # -------------------------------------------------------------------------------------------------------------------------------
        if cfg.BINARY_LOSS or cfg.EVAL_MAP:
            if 'vhico' in self.args.dataset:
                if cfg.EVAL_SUBSET=='test':
                    self.video_name_triplet_dict = pickle.load(open(TRIPLET_TEST, 'rb'))
                    # self.video_name_triplet_dict = pickle.load(open(TRIPLET_TRAIN, 'rb'))
                    print('there are %d triplets in %s' % (len(self.video_name_triplet_dict['triplet_id_frame']), TRIPLET_TEST) )
                elif cfg.EVAL_SUBSET=='unseen':
                    self.video_name_triplet_dict = pickle.load(open(TRIPLET_UNSEEN, 'rb'))
                    print('there are %d triplets in %s' % (len(self.video_name_triplet_dict['triplet_id_frame']), TRIPLET_UNSEEN) )
                else:
                    self.video_name_triplet_dict = pickle.load(open(TRIPLET_TRAIN, 'rb'))
                    print('there are %d triplets in %s' % (len(self.video_name_triplet_dict['triplet_id_frame']), TRIPLET_TRAIN) )

        # -------------------------------------------------------------------------------------------------------------------------------
        # initialize model
        # -------------------------------------------------------------------------------------------------------------------------------
        self._init_modules()
        

    def _init_modules(self):
        
        # VGG16 imagenet pretrained model is initialized in VGG16.py
        if cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS != '':
            logger.info("Loading pretrained weights from %s", cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
            resnet_utils.load_pretrained_imagenet_weights(self)

        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VRD_PRETRAINED_WEIGHTS)

        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VG_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VG_PRETRAINED_WEIGHTS)

        if cfg.RESNETS.VHICO_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VHICO_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VHICO_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VHICO_PRETRAINED_WEIGHTS)


        if cfg.RESNETS.COCO_PRETRAINED_WEIGHTS != '':
            load_detectron_weight(self, cfg.RESNETS.COCO_PRETRAINED_WEIGHTS, ('cls_score', 'bbox_pred'))


        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

        if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '' or \
            cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '' or \
            cfg.RESNETS.VHICO_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VHICO_PRD_PRETRAINED_WEIGHTS != '':

            if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)

            if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)

            if cfg.RESNETS.VHICO_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VHICO_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VHICO_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VHICO_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VHICO_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VHICO_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)


            # not using the last softmax layers
            del checkpoint['model']['Box_Outs.cls_score.weight']
            del checkpoint['model']['Box_Outs.cls_score.bias']
            del checkpoint['model']['Box_Outs.bbox_pred.weight']
            del checkpoint['model']['Box_Outs.bbox_pred.bias']


    def load_detector_weights(self, weight_name):
        logger.info("loading pretrained weights from %s", weight_name)
        checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(self, checkpoint['model'])
        # freeze everything above the rel module
        for p in self.Conv_Body.parameters():
            p.requires_grad = False
        for p in self.RPN.parameters():
            p.requires_grad = False
        if not cfg.MODEL.UNFREEZE_DET:
            for p in self.Box_Head.parameters():
                p.requires_grad = False
            for p in self.Box_Outs.parameters():
                p.requires_grad = False

    def forward(self, data, human_mask, human_box, im_info, roidb=None, dataset_name=None, use_gt_labels=False, full_batch=False, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, human_mask, human_box, im_info, roidb, dataset_name, use_gt_labels, full_batch=full_batch, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, human_mask, human_box, im_info, roidb, dataset_name, use_gt_labels, full_batch=full_batch, **rpn_kwargs)

    def _forward(self, data, human_mask, human_box, im_info, roidb=None, dataset_name=None, use_gt_labels=False, full_batch=False, **rpn_kwargs):
        # -------------------------------------------------------------------------------------------------------------------------------
        # preapare data
        # -------------------------------------------------------------------------------------------------------------------------------
        im_data = data
        B, _, _, _ = im_data.shape
        roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()
        assert roidb[0]['dataset']==roidb[-1]['dataset']
        dataset_name = roidb[0]['dataset']

        # -------------------------------------------------------------------------------------------------------------------------------
        # run model
        # -------------------------------------------------------------------------------------------------------------------------------
        blob_conv = self.Conv_Body(im_data)    
        frame_feat = F.avg_pool2d(blob_conv[0], kernel_size=(blob_conv[0].shape[2], blob_conv[0].shape[3]))
        frame_feat = frame_feat.view(B, -1)

        ### retrun rpn_ret and put add 2000 rois in roidb
        rpn_ret = self.RPN(blob_conv, im_info, roidb)
        
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

            
        # ---------------------------------------------------------------------------------------------------------
        # now go through the predicate branch
        # ---------------------------------------------------------------------------------------------------------
        use_relu = False if cfg.MODEL.NO_FC7_RELU else True
        det_rois = rpn_ret['rois']
        ## assign generated rois to object and predicate bounding box
        rel_ret = self.RelPN(det_rois, im_info, dataset_name, roidb)
        obj_rois = rel_ret['obj_rois']
        obj_feat = self.Box_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu) # [651, 1024]

        if cfg.EVAL_MAP:
            gt_obj_label =  (torch.IntTensor([db['obj_gt_cls'] for db in roidb]).cuda(device_id))
            gt_prd_label =  (torch.IntTensor([db['prd_gt_cls'] for db in roidb]).cuda(device_id))
                
            return_dicts = {}
            for triplet_name, _ in self.video_name_triplet_dict['triplet_id_frame'].items():
                prd_cls_query = int(triplet_name.split('___')[0])
                obj_cls_query = int(triplet_name.split('___')[1])
                
                if (obj_cls_query!=int(gt_obj_label[0]) or prd_cls_query!=int(gt_prd_label[0])):
                    binary_label = torch.tensor([0]).cuda(device_id)
                else:
                    binary_label = torch.tensor([1]).cuda(device_id)
    
                for db in roidb: db['obj_gt_cls'] = obj_cls_query
                for db in roidb: db['prd_gt_cls'] = prd_cls_query
                
                output = self.RelDN(frame_feat, obj_feat, human_mask, human_box, roidb=roidb, roi=obj_rois, batch=rel_ret['obj_gt_cls'].shape[0], no_dropout=(dataset_name == cfg.TEST.DATASETS[0]), full_batch=full_batch, binary_label=binary_label)
                return_dict = collect_output(cfg, dataset_name, im_info, roidb, obj_rois, gt_obj_label, gt_prd_label, output, device_id)
                return_dicts[triplet_name] = deepcopy(return_dict)

        else:
            gt_obj_label =  torch.IntTensor([db['obj_gt_cls'] for db in roidb]).cuda(device_id)
            gt_prd_label =  torch.IntTensor([db['prd_gt_cls'] for db in roidb]).cuda(device_id)
                
            if cfg.BINARY_LOSS:
                if self.training:
                    if random.uniform(0,1)>0.5:
                        count = 0
                        while 1:
                            count += 1
                            triplet_name = random.choice(list(self.video_name_triplet_dict['triplet_id_frame'].keys()))
                            prd_cls_query = int(triplet_name.split('___')[0])
                            obj_cls_query = int(triplet_name.split('___')[1])
                            
                            if (obj_cls_query!=int(gt_obj_label[0]) or prd_cls_query!=int(gt_prd_label[0])):
                                for db in roidb: db['obj_gt_cls'] = obj_cls_query
                                for db in roidb: db['prd_gt_cls'] = prd_cls_query
                                binary_label = torch.tensor([0]).cuda(device_id)
                                break

                            if count>100:
                                print('count>100 for binary loss')
                                binary_label = torch.tensor([1]).cuda(device_id)
                                break

                        gt_obj_label =  torch.IntTensor([db['obj_gt_cls'] for db in roidb]).cuda(device_id)
                        gt_prd_label =  torch.IntTensor([db['prd_gt_cls'] for db in roidb]).cuda(device_id)
                    else:
                        binary_label = torch.tensor([1]).cuda(device_id)
                else:
                    binary_label = torch.tensor([1]).cuda(device_id)
            else:
                binary_label = torch.tensor([1]).cuda(device_id)
            
            output = self.RelDN(frame_feat, obj_feat, human_mask, human_box, roidb=roidb, roi=obj_rois, batch=rel_ret['obj_gt_cls'].shape[0], no_dropout=(dataset_name == cfg.TEST.DATASETS[0]), full_batch=full_batch, binary_label=binary_label)
            return_dict = collect_output(cfg, dataset_name, im_info, roidb, obj_rois, gt_obj_label, gt_prd_label, output, device_id)
            return_dicts = {}
            return_dicts['gt_label'] = return_dict

        return return_dicts
    


    def get_prediction(self):
        return self.prediction

    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds
    
    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)


        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order [4, 256, 56, 56]
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id) # [1981, 5]
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                if type(spatial_scale) == list:
                    spatial_scale = spatial_scale[-1]
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out


    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    if name in ['Conv_Body', 'RPN', 'Box_Head']:
                        child_map, child_orphan = m_child.detectron_weight_mapping()
                        d_orphan.extend(child_orphan)
                        for key, value in child_map.items():
                            new_key = name + '.' + key
                            d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

