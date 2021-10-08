import numpy as np
from numpy import linalg as la
import json
import logging

from torch import nn
from torch.nn import init
import torch.nn.functional as F

from core.config import cfg
from modeling.generate_rel_proposal_labels import GenerateRelProposalLabelsOp
import modeling.FPN as FPN
import utils.net as net_utils
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import roi_data.fast_rcnn_rel
import pdb

logger = logging.getLogger(__name__)


def generic_relpn_outputs():
    return single_scale_relpn_outputs()


class single_scale_relpn_outputs(nn.Module):
    """Add RelPN outputs to a single scale model (i.e., no FPN)."""
    def __init__(self):
        super().__init__()
        
        print('single_scale_relpn_outputs')

    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def remove_self_pairs(self, det_size, sbj_inds, obj_inds):
        mask = np.ones(sbj_inds.shape[0], dtype=bool)
        for i in range(det_size):
            mask[i + det_size * i] = False
        keeps = np.where(mask)[0]
        sbj_inds = sbj_inds[keeps]
        obj_inds = obj_inds[keeps]
        return sbj_inds, obj_inds

    def forward(self, det_rois, im_info, dataset_name, roidb=None):
        obj_rois = det_rois

        # ------------------------------------------------------------------------------------------------------------------------
        ## add obj_boxes and human_gt_boxes in roidb
        # ------------------------------------------------------------------------------------------------------------------------
        im_scales = im_info.data.numpy()[:, 2]
        human_box_list = []
        obj_box_list = []
        for i, entry in enumerate(roidb):
            inv_im_scale = 1. / im_scales[i]
            idx = np.where(obj_rois[:, 0] == i)[0]
            
            obj_box_list.append(obj_rois[idx, 1:] * inv_im_scale)
            entry['obj_boxes'] = obj_box_list[i]

        # ------------------------------------------------------------------------------------------------------------------------
        ## add obj_rois and human_gt_boxes in roidb
        # ------------------------------------------------------------------------------------------------------------------------
        output_blob_names = ['obj_rois', 'rel_rois', 'prd_gt_cls', 'obj_gt_cls']
        blobs = {k: [] for k in output_blob_names}
        
        roi_data.fast_rcnn_rel.add_rel_blobs(blobs, im_scales, roidb)

        return_dict = {}
        return_dict.update(blobs)

        return return_dict














