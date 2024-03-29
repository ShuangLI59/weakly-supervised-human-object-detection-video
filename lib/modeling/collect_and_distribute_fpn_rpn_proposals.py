import numpy as np
from torch import nn

from core.config import cfg
from datasets import json_dataset
import roi_data.fast_rcnn
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import pdb

class CollectAndDistributeFpnRpnProposalsOp(nn.Module):
    """Merge RPN proposals generated at multiple FPN levels and then
    distribute those proposals to their appropriate FPN levels. An anchor
    at one FPN level may predict an RoI that will map to another level,
    hence the need to redistribute the proposals.

    This function assumes standard blob names for input and output blobs.

    Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                  rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
        - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
        documentation from GenerateProposals.
        - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
        level i; see rpn_roi_probs documentation from GenerateProposals.

    If used during training, then the input blobs will also include:
        [roidb, im_info] (see GenerateProposalLabels).

    Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                   rois_idx_restore]
        - rois_fpn<i> are the RPN proposals for FPN level i
        - rois_idx_restore is a permutation on the concatenation of all
        rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
        restored to their original order in the input blobs.

    If used during training, then the output blobs will also include:
        [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, roidb, im_info):
        """
        Args:
            inputs: a list of [rpn_rois_fpn2, ..., rpn_rois_fpn6,
                               rpn_roi_probs_fpn2, ..., rpn_roi_probs_fpn6] # (2905, 5)
            im_info: [[im_height, im_width, im_scale], ...]
        """
        num_img = im_info.shape[0]
        rois = collect(inputs, self.training, num_img)

        # During training we reuse the data loader code. We populate roidb
        # entries on the fly using the rois generated by RPN.
        im_scales = im_info.data.numpy()[:, 2]
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).

        ### add 2000 rois in roidb
        json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)

        # Compute training labels for the RPN proposals; also handles
        # distributing the proposals over FPN levels
        # output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
        
        output_blob_names = ['rois']
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
            # Support for FPN multi-level rois without bbox reg isn't
            # implemented (... and may never be implemented)
            k_max = cfg.FPN.ROI_MAX_LEVEL
            k_min = cfg.FPN.ROI_MIN_LEVEL
            # Same format as rois blob, but one per FPN level
            for lvl in range(k_min, k_max + 1):
                output_blob_names += ['rois_fpn' + str(lvl)]
            output_blob_names += ['rois_idx_restore_int32']


        blobs = {k: [] for k in output_blob_names}
        roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)

        return blobs


def collect(inputs, is_training, num_img):
    # cfg_key = 'TRAIN' if is_training else 'TEST'
    post_nms_topN = int(cfg['TRAIN'].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5) # 2000
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]

    # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
    # Combine predictions across all levels and retain the top scoring
    rois = np.concatenate(roi_inputs) # (4387, 5)
    scores = np.concatenate(score_inputs).squeeze() #(4387,)

    rois_each_gpu = []
    for img_i in range(num_img):
        roi_img_i = rois[rois[:,0] == img_i]
        score_img_i = scores[rois[:,0] == img_i]
        assert len(roi_img_i) == len(score_img_i)
        inds = np.argsort(-score_img_i)[:post_nms_topN] # (2000, 5)
        rois_each_gpu.append(roi_img_i[inds, :])
    
    rois_each_gpu = np.concatenate(rois_each_gpu)
    return rois_each_gpu


