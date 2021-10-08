import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import nn as mynn

from core.config import cfg
import utils.net as net_utils
import math
import pdb
import utils.blob as blob_utils
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction


logger = logging.getLogger(__name__)



class reldn_head(nn.Module):
    def __init__(self, dim_in, all_obj_vecs=None, all_prd_vecs=None):
        super().__init__()

        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs

        assert not math.isnan(all_obj_vecs.max())
        assert not math.isnan(all_prd_vecs.max())


        self.obj_text_feats = nn.Sequential(nn.Linear(300, 128))
        self.prd_text_feats = nn.Sequential(nn.Linear(300, 128))


        self.obj_feats = nn.Sequential(
                nn.Linear(dim_in, 256))

        self.obj_feats_2 = nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Linear(256, 128))
   
        self.human_mask_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            )

        self.human_mask_feats = nn.Sequential(
            nn.Linear(64*49, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128)
            )

        ## ------------------------------------------------------------------------------------------------
        # COM_WEIGHT
        ## ------------------------------------------------------------------------------------------------
        if cfg.COM_WEIGHT=='cat_video_soft_attention':
            self.frame_fc = nn.Sequential(
                                nn.Linear(256, 256),
                                nn.LeakyReLU(0.1),
                                nn.Linear(256, 128))
        
            self.soft_attention_fc1 = nn.Linear(128, 128)
            self.soft_attention_fc2 = nn.Linear(128, 128)
            self.soft_attention_fc3 = nn.Linear(128, 128)
            self.soft_attention_fc4 = nn.Linear(128, 128)
            self.soft_attention_fc5 = nn.Linear(128, 128)
            self.soft_attention_fc6 = nn.Linear(128, 128)
            
            self.roi_weights_net_obj = nn.Sequential(
                    nn.Linear(4 * 128, 128),
                    nn.LeakyReLU(0.1),
                    nn.Linear(128, 1)
                    )
            self.roi_weights_net_human = nn.Sequential(
                    nn.Linear(4 * 128, 128),
                    nn.LeakyReLU(0.1),
                    nn.Linear(128, 1)
                    )
        else:
            error('please select COM_WEIGHT')
        ## ------------------------------------------------------------------------------------------------

        if cfg.BINARY_LOSS:
            self.frame_fc_binary = nn.Sequential(
                                nn.Linear(256, 256),
                                nn.LeakyReLU(0.1),
                                nn.Linear(256, 128))
            
            self.frame_fc_cat_binary = nn.Linear(128*3, 128)

            self.binary_classifier = nn.Sequential(
                                nn.Linear(128, 128),
                                nn.LeakyReLU(0.1),
                                nn.Linear(128, 2))


        self.dropout = torch.nn.Dropout(p=cfg.DROPOUT)
        self.word_embed_contrast_obj = nn.Sequential(nn.Linear(300, 128))
        self.word_embed_contrast_prd = nn.Sequential(nn.Linear(300, 128))

        if cfg.HUMAN_OBJ_SPATIAL:
            self.fuse_human_obj_feat = nn.Sequential(
                                nn.Linear(256, 256),
                                nn.LeakyReLU(0.1),
                                nn.Linear(256, 128))

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, frame_feat=None, obj_feat=None, human_mask=None, human_box=None, roidb=None, roi=None, batch=1, no_dropout=False, full_batch=False, binary_label=None):
        device_id = obj_feat.get_device()

        B = frame_feat.shape[0]

        obj_label =  torch.IntTensor([db['obj_gt_cls'] for db in roidb]).cuda(device_id)
        prd_label =  torch.IntTensor([db['prd_gt_cls'] for db in roidb]).cuda(device_id)

        # -------------------------------------------------------------------------------------------------------------------
        # obj visual and human visual
        # -------------------------------------------------------------------------------------------------------------------
        obj_inter = self.obj_feats(obj_feat)
        obj_hidden = self.obj_feats_2(obj_inter)
        obj_hidden_norm = F.normalize(obj_hidden, p=2, dim=1)

        densepose_mask = human_mask
        densepose_mask = densepose_mask.view(-1,1,densepose_mask.shape[1],densepose_mask.shape[2])
        densepose_mask_conv = self.human_mask_conv(densepose_mask)
        
        roi_feature = RoIAlignFunction(7, 7, 1. / 8, 0.0)(densepose_mask_conv, Variable(torch.from_numpy(roi)).cuda(device_id))
        roi_feature = roi_feature.view(-1, 64*49)
        densepose_mask_hidden = self.human_mask_feats(roi_feature)
        densepose_mask_hidden_norm = F.normalize(densepose_mask_hidden, p=2, dim=1)


        ## ----------------------------------------------------------------------------------------------------------------------
        ## extract the features of densepose bounding boxes
        ## ----------------------------------------------------------------------------------------------------------------------
        human_boxs = []
        for batch_idx, tem_human_box in enumerate(human_box):
            repeated_batch_idx = batch_idx * blob_utils.ones((tem_human_box.shape[0], 1))
            tem_human_box = np.hstack((repeated_batch_idx, tem_human_box[:,1:]))
            for tem in tem_human_box:
                human_boxs.append(tem) 
        densepose_roi = np.array(human_boxs)

        densepose_roi_feature = RoIAlignFunction(7, 7, 1. / 8, 0.0)(densepose_mask_conv, Variable(torch.from_numpy(densepose_roi)).cuda(device_id))
        densepose_roi_feature = densepose_roi_feature.view(-1, 64*49)
        densepose_box_feature_hidden = self.human_mask_feats(densepose_roi_feature)
        densepose_box_feature_hidden_norm = F.normalize(densepose_box_feature_hidden, p=2, dim=1)
        

        # -------------------------------------------------------------------------------------------------------------------
        # obj text and prd text
        # -------------------------------------------------------------------------------------------------------------------
        ## obj text
        obj_text_vecs = self.obj_vecs[obj_label]
        obj_text_vecs = Variable(torch.from_numpy(obj_text_vecs.astype('float32'))).cuda(device_id)
        if obj_text_vecs.dim()==1:
            obj_text_vecs = obj_text_vecs.view(1, -1)
        
        obj_text_hidden = self.obj_text_feats(obj_text_vecs)
        obj_text_hidden_norm = F.normalize(obj_text_hidden, p=2, dim=1)  # (#prd, 1024)

        ## prd text
        prd_text_vecs = self.prd_vecs[prd_label]
        prd_text_vecs = Variable(torch.from_numpy(prd_text_vecs.astype('float32'))).cuda(device_id)
        if prd_text_vecs.dim()==1:
            prd_text_vecs = prd_text_vecs.view(1,-1)

        prd_text_hidden = self.prd_text_feats(prd_text_vecs)
        prd_text_hidden_norm = F.normalize(prd_text_hidden, p=2, dim=1)



        # -------------------------------------------------------------------------------------------------------------------
        # video binary loss, text match video
        # -------------------------------------------------------------------------------------------------------------------
        if cfg.BINARY_LOSS:
            frame_feat_binary = self.frame_fc_binary(frame_feat)
            frame_feat_binary = torch.cat([frame_feat_binary, obj_text_hidden, prd_text_hidden], dim=-1)
            frame_feat_binary = self.frame_fc_cat_binary(frame_feat_binary)
            frame_feat_binary = frame_feat_binary.mean(0)
            frame_feat_binary_pred = self.binary_classifier(frame_feat_binary)
            frame_feat_binary_pred = frame_feat_binary_pred.view(1, -1)
            video_binary_loss = F.cross_entropy(frame_feat_binary_pred, binary_label)
            video_binary_loss = 10*video_binary_loss
        else:
            video_binary_loss = torch.tensor([0]).cuda(device_id)

        # -------------------------------------------------------------------------------------------------------------------
        # concate visual obj + obj text + prd text --> weight
        # -------------------------------------------------------------------------------------------------------------------
        if roi is None:
            obj_text_hidden_norm_expn = obj_text_hidden_norm.expand(obj_hidden_norm.shape[0], obj_text_hidden_norm.shape[1])
            prd_text_hidden_norm_expn = prd_text_hidden_norm.expand(obj_hidden_norm.shape[0], prd_text_hidden_norm.shape[1])
    
            densepose_mask_hidden_norm_expn = densepose_mask_hidden_norm.expand(obj_hidden_norm.shape[0], densepose_mask_hidden_norm.shape[1])
            densepose_box_feature_hidden_norm_expn = densepose_box_feature_hidden_norm.expand(densepose_box_feature_hidden_norm.shape[0], densepose_box_feature_hidden_norm.shape[1])
        else:
            gather_obj_index = torch.Tensor(roi[:, 0:1]).long().repeat(1, obj_text_hidden_norm.shape[1]).cuda(device_id)
            gather_prd_index = torch.Tensor(roi[:, 0:1]).long().repeat(1, prd_text_hidden_norm.shape[1]).cuda(device_id)
            obj_text_hidden_norm_expn = torch.gather(obj_text_hidden_norm, 0, gather_obj_index)
            prd_text_hidden_norm_expn = torch.gather(prd_text_hidden_norm, 0, gather_prd_index)


            gather_obj_index = torch.Tensor(densepose_roi[:, 0:1]).long().repeat(1, obj_text_hidden_norm.shape[1]).cuda(device_id)
            gather_prd_index = torch.Tensor(densepose_roi[:, 0:1]).long().repeat(1, prd_text_hidden_norm.shape[1]).cuda(device_id)
            densepose_obj_text_hidden_norm_expn = torch.gather(obj_text_hidden_norm, 0, gather_obj_index)
            densepose_prd_text_hidden_norm_expn = torch.gather(prd_text_hidden_norm, 0, gather_prd_index)


        if cfg.HUMAN_OBJ_SPATIAL:
            obj_hidden_norm = torch.max(obj_hidden_norm, densepose_mask_hidden_norm)
            obj_hidden = torch.max(obj_hidden, densepose_mask_hidden)


        # -------------------------------------------------------------------------------------------------------------------
        # COM_WEIGHT
        # -------------------------------------------------------------------------------------------------------------------
        if cfg.COM_WEIGHT=='cat_video_soft_attention':
            frame_feat = self.frame_fc(frame_feat)
            
            query = self.soft_attention_fc2(F.relu(self.soft_attention_fc1(frame_feat)))
            key = self.soft_attention_fc4(F.relu(self.soft_attention_fc3(frame_feat)))
            sim = query[:, None, : ] * key[None, :, : ]
            sim = F.softmax(sim.sum(dim=-1), dim=-1)
            value = self.soft_attention_fc6(F.relu(self.soft_attention_fc5(frame_feat)))
            frame_feat = (sim[:, :, None] * value[:, None, : ].repeat(1, sim.size(1), 1)).sum(dim=1)
            frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

            gather_index = torch.Tensor(roi[:, 0:1]).long().repeat(1, frame_feat_norm.shape[1]).cuda(device_id)
            frame_feat_norm_obj = torch.gather(frame_feat_norm, 0, gather_index)
            gather_index = torch.Tensor(densepose_roi[:, 0:1]).long().repeat(1, frame_feat_norm.shape[1]).cuda(device_id)
            frame_feat_norm_human = torch.gather(frame_feat_norm, 0, gather_index)
            
            concated_vecs = torch.cat((obj_hidden_norm, obj_text_hidden_norm_expn, prd_text_hidden_norm_expn, frame_feat_norm_obj), dim=1)
            densepose_concated_vecs = torch.cat((densepose_box_feature_hidden_norm, densepose_obj_text_hidden_norm_expn, densepose_prd_text_hidden_norm_expn, frame_feat_norm_human), dim=1) 
            
            roi_weights = self.roi_weights_net_obj(concated_vecs)
            roi_weights_human = self.roi_weights_net_human(densepose_concated_vecs)

        
        # -------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------------
        if roi is None:
            roi_weights = F.softmax(roi_weights, dim=0)
            roi_weights = roi_weights.view(1,-1)
            obj_hidden_weighted = torch.mm(roi_weights, obj_hidden)

            roi_weights_human = F.softmax(roi_weights_human, dim=0)
            roi_weights_human = roi_weights_human.view(1,-1)
            densepose_box_feature_hidden_weighted = torch.mm(roi_weights_human, densepose_box_feature_hidden)
        else:
            roi_weights_unpacked = roi_weights.view(-1, cfg.TRAIN.BATCH_SIZE_PER_IM, 1)
            roi_weights_human_unpacked = roi_weights_human.view(-1, cfg.MAX_NUM_HUMAN, 1)

            roi_weights_ori = roi_weights.view(-1, cfg.TRAIN.BATCH_SIZE_PER_IM)
            roi_weights_human_ori = roi_weights_human.view(-1, cfg.MAX_NUM_HUMAN)

            
            if not no_dropout:
                roi_weights_unpacked = self.dropout(roi_weights_unpacked)
                roi_weights_human_unpacked = self.dropout(roi_weights_human_unpacked)

            roi_weights_unpacked = F.softmax(roi_weights_unpacked, dim=1)
            roi_weights_human_unpacked = F.softmax(roi_weights_human_unpacked, dim=1)


            ## ---------------------------------------------------------------------------------------------------------
            ## feature
            ## ---------------------------------------------------------------------------------------------------------
            obj_hidden_unpacked = obj_hidden.view(-1, cfg.TRAIN.BATCH_SIZE_PER_IM, obj_hidden.size(1))
            densepose_box_feature_hidden_unpacked = densepose_box_feature_hidden.view(-1, cfg.MAX_NUM_HUMAN, densepose_box_feature_hidden.size(1))
            
            obj_hidden_weighted = torch.sum(roi_weights_unpacked * obj_hidden_unpacked, dim=1)
            obj_hidden_human_weighted = torch.sum(roi_weights_human_unpacked * densepose_box_feature_hidden_unpacked, dim=1)

            if cfg.VIDEO_LOSS=='contrastive_max_plus':
                obj_feat = obj_hidden
                obj_hidden_video_unpacked = obj_feat.view(-1, cfg.VIDEO_FRAME, cfg.TRAIN.BATCH_SIZE_PER_IM, obj_feat.shape[1])
                # obj_hidden_video_unpacked = F.normalize(obj_hidden_video_unpacked, p=2, dim=3) * 4
                roi_weights_unpacked_batch = roi_weights_unpacked.view(-1, cfg.VIDEO_FRAME, cfg.TRAIN.BATCH_SIZE_PER_IM, 1)
                # idx = torch.max(roi_weights_unpacked_batch, dim=2)[1][:, :, None, :]
                sort_idx = torch.sort(roi_weights_unpacked_batch, dim=2)[1]
                idx_select = sort_idx[:, :, -1:].repeat(1, 1, 1, obj_feat.shape[1])
                anchor_embed = torch.gather(obj_hidden_video_unpacked, 2, idx_select)

                # Randomly sample a positive pair of frames for positive samples
                permute = torch.randperm(cfg.VIDEO_FRAME).cuda(device_id)
                pos_embed = anchor_embed[:, permute]

                permute = torch.randperm(cfg.TRAIN.BATCH_SIZE_PER_IM).cuda(device_id)
                obj_hidden_video_permute = obj_hidden_video_unpacked[:, :, permute]
                neg_sample = 15

                neg_embed = obj_hidden_video_unpacked[:, :, :neg_sample]

                pos_dot = (pos_embed * anchor_embed).sum(dim=3)
                neg_dot = (neg_embed * anchor_embed).sum(dim=3)

                
                neg_dot = -torch.cat([pos_dot, neg_dot], dim=-1)
                pos_dot = -pos_dot
                video_loss = pos_dot.view(-1, cfg.VIDEO_FRAME) + torch.logsumexp(-neg_dot, dim=-1)

                select_frames = max(int(cfg.VIDEO_FRAME * 0.7), 1)
                video_loss, _ = torch.sort(video_loss, dim=1)
                video_loss =  cfg.VIDEO_WEIGHT * video_loss[:, :select_frames].mean()
            else:
                video_loss = torch.zeros(1)[0].cuda(device_id)


        if cfg.OBJ_LOSS == 'contrastive_objloss':
            hidden_weighted_obj = obj_hidden_weighted
            hidden_weighted_prd = obj_hidden_human_weighted

            nsample = 15
            word_embed_contrast_obj = self.word_embed_contrast_obj(obj_text_vecs)
            word_embed_contrast_prd = self.word_embed_contrast_prd(prd_text_vecs)

            ## neg obj
            n_obj = self.obj_vecs.shape[0]
            neg_sample = np.random.choice(np.arange(n_obj, dtype=np.int32), size=(obj_text_vecs.shape[0]*nsample,))
            neg_embed_obj = self.obj_vecs[neg_sample]
            neg_embed_obj = neg_embed_obj.reshape((int(obj_text_vecs.shape[0]), nsample, 300))
            neg_embed_obj = Variable(torch.from_numpy(neg_embed_obj.astype('float32'))).cuda(device_id)

            ## neg prd
            n_prd = self.prd_vecs.shape[0]
            neg_sample = np.random.choice(np.arange(n_prd, dtype=np.int32), size=(prd_text_vecs.shape[0]*nsample,))
            neg_embed_prd = self.prd_vecs[neg_sample]
            neg_embed_prd = neg_embed_prd.reshape((int(prd_text_vecs.shape[0]), nsample, 300))
            neg_embed_prd = Variable(torch.from_numpy(neg_embed_prd.astype('float32'))).cuda(device_id)

            ## embed neg obj and prd
            neg_embed_contrast_obj =  self.word_embed_contrast_obj(neg_embed_obj)
            neg_embed_contrast_prd =  self.word_embed_contrast_prd(neg_embed_prd)

            pos_dot = (hidden_weighted_obj[:, None, :] * word_embed_contrast_obj[:, None, :]).sum(dim=2)
            neg_dot = (hidden_weighted_obj[:, None, :] * neg_embed_contrast_obj).sum(dim=2)
            neg_dot = -torch.cat([pos_dot, neg_dot], dim=-1)
            pos_dot = -pos_dot
            obj_loss = pos_dot.view(-1, cfg.VIDEO_FRAME) + torch.logsumexp(-neg_dot, dim=-1).view(-1, cfg.VIDEO_FRAME)


            pos_dot = (hidden_weighted_prd[:, None, :] * word_embed_contrast_prd[:, None, :]).sum(dim=2)
            neg_dot = (hidden_weighted_prd[:, None, :] * neg_embed_contrast_prd).sum(dim=2)
            neg_dot = -torch.cat([pos_dot, neg_dot], dim=-1)
            pos_dot = -pos_dot
            prd_loss = pos_dot.view(-1, cfg.VIDEO_FRAME) + torch.logsumexp(-neg_dot, dim=-1).view(-1, cfg.VIDEO_FRAME)

            # select_frames = max(int(cfg.VIDEO_FRAME * 0.5), 1)
            select_frames = cfg.VIDEO_FRAME

            obj_loss, _ = torch.sort(obj_loss, dim=-1)
            obj_loss = obj_loss[:, :select_frames]
            # obj_loss = torch.clamp(obj_loss, 0, 1e5)
            obj_loss = obj_loss.mean()
            obj_scores = None

            prd_loss, _ = torch.sort(prd_loss, dim=-1)
            prd_loss = prd_loss[:, :select_frames]
            prd_loss = prd_loss.mean()
            

        if cfg.WEIGHT_REG == 'L2':
            weight_loss = torch.norm(roi_weights_unpacked,2.0,1)
            weight_loss = -cfg.L2_WEIGHT * torch.log(weight_loss.mean())

            weight_human_loss = torch.norm(roi_weights_human_unpacked,2.0,1)
            weight_human_loss = -cfg.L2_WEIGHT * torch.log(weight_human_loss.mean())
        

        cls_prediction = {}
        if not self.training and cfg.BINARY_LOSS:
            cls_prediction['binary_pred'] = frame_feat_binary_pred

        
        if cfg.BINARY_LOSS:
            loss_scale = F.softmax(frame_feat_binary_pred)[0][1]
            obj_loss = obj_loss * loss_scale
            prd_loss = prd_loss * loss_scale
            video_loss = video_loss
            weight_loss = weight_loss
            weight_human_loss = weight_human_loss

        return obj_loss, prd_loss, weight_loss, weight_human_loss, video_loss, video_binary_loss, roi_weights_unpacked, roi_weights_human_unpacked, densepose_roi, roi_weights_ori, roi_weights_human_ori, cls_prediction



def reldn_losses(obj_cls_scores, obj_labels_int32, fg_only=False, roi=None):
    device_id = obj_cls_scores.get_device()
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)

    if cfg.OBJ_LOSS == "cls_trn":
        obj_labels = obj_labels.view(-1, cfg.VIDEO_FRAME)
        obj_labels = obj_labels[:,0]

    obj_loss = F.cross_entropy(obj_cls_scores, obj_labels)
    # class accuracy
    prd_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    obj_accuracy = prd_cls_preds.eq(obj_labels).float().mean(dim=0)

    return obj_loss, obj_accuracy


def reldn_so_losses(sbj_cls_scores, obj_cls_scores, sbj_labels_int32, obj_labels_int32):
    device_id = sbj_cls_scores.get_device()

    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_sbj = F.cross_entropy(sbj_cls_scores, sbj_labels)
    sbj_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_sbj = sbj_cls_preds.eq(sbj_labels).float().mean(dim=0)
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = F.cross_entropy(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_sbj, loss_cls_obj, accuracy_cls_sbj, accuracy_cls_obj
