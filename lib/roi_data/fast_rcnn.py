# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Construct minibatches for Fast R-CNN training. Handles the minibatch blobs
that are specific to Fast R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.random as npr

from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import pdb


def get_fast_rcnn_blob_names(is_training=True):
    """Fast R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        
    return blob_names


# def get_fast_rcnn_blob_names(is_training=True):
#     """Fast R-CNN blob names."""
#     # rois blob: holds R regions of interest, each is a 5-tuple
#     # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
#     # rectangle (x1, y1, x2, y2)
#     blob_names = ['rois']
#     if is_training:
#         # labels_int32 blob: R categorical labels in [0, ..., K] for K
#         # foreground classes plus background
#         blob_names += ['labels_int32']
#         # if cfg.MODEL.USE_SE_LOSS:
#         #     blob_names += ['bce_labels']
#     if is_training:
#         # bbox_targets blob: R bounding-box regression targets with 4
#         # targets per class
#         blob_names += ['bbox_targets']
#         # bbox_inside_weights blob: At most 4 targets per roi are active
#         # this binary vector sepcifies the subset of active targets
#         blob_names += ['bbox_inside_weights']
#         blob_names += ['bbox_outside_weights']
#     if is_training and cfg.MODEL.MASK_ON:
#         # 'mask_rois': RoIs sampled for training the mask prediction branch.
#         # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
#         blob_names += ['mask_rois']
#         # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
#         # indicating if each RoI has a mask or not. Note that in some cases
#         # a *bg* RoI will have an all -1 (ignore) mask associated with it in
#         # the case that no fg RoIs can be sampled. Shape is (batchsize).
#         blob_names += ['roi_has_mask_int32']
#         # 'masks_int32' holds binary masks for the RoIs specified in
#         # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
#         # mask size.
#         # if cfg.MRCNN.CLS_SPECIFIC_MASK: Shape is (#masks, #classes * M ** 2)
#         blob_names += ['masks_int32']
#     if is_training and cfg.MODEL.KEYPOINTS_ON:
#         # 'keypoint_rois': RoIs sampled for training the keypoint prediction
#         # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
#         # y2).
#         blob_names += ['keypoint_rois']
#         # 'keypoint_locations_int32': index of keypoint in
#         # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances * #keypoints). Used in
#         # SoftmaxWithLoss.
#         blob_names += ['keypoint_locations_int32']
#         # 'keypoint_weights': weight assigned to each target in
#         # 'keypoint_locations_int32'. Shape is (#instances * #keypoints). Used in
#         # SoftmaxWithLoss.
#         blob_names += ['keypoint_weights']
#         # 'keypoint_loss_normalizer': optional normalization factor to use if
#         # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
#         blob_names += ['keypoint_loss_normalizer']
#     if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
#         # Support for FPN multi-level rois without bbox reg isn't
#         # implemented (... and may never be implemented)
#         k_max = cfg.FPN.ROI_MAX_LEVEL
#         k_min = cfg.FPN.ROI_MIN_LEVEL
#         # Same format as rois blob, but one per FPN level
#         for lvl in range(k_min, k_max + 1):
#             blob_names += ['rois_fpn' + str(lvl)]
#         blob_names += ['rois_idx_restore_int32']
#         if is_training:
#             if cfg.MODEL.MASK_ON:
#                 for lvl in range(k_min, k_max + 1):
#                     blob_names += ['mask_rois_fpn' + str(lvl)]
#                 blob_names += ['mask_rois_idx_restore_int32']
#             if cfg.MODEL.KEYPOINTS_ON:
#                 for lvl in range(k_min, k_max + 1):
#                     blob_names += ['keypoint_rois_fpn' + str(lvl)]
#                 blob_names += ['keypoint_rois_idx_restore_int32']
#     return blob_names


def add_fast_rcnn_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
 
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i) 
        ## ----------------- ls -----------------------
        ## select positive and negative samples
        # ['labels_int32', 'rois', 'bbox_targets', 'bbox_inside_weights', 'bbox_outside_weights']
        ## ----------------- ls -----------------------
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    
    return valid


def _sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM) # 512
    
    rois_per_this_image = np.minimum(rois_per_image, len(roidb['boxes']))
    keep_inds = npr.choice(range(len(roidb['boxes'])), size=rois_per_this_image, replace=False)
    keep_inds = keep_inds.astype(np.int32)
    sampled_boxes = roidb['boxes'][keep_inds] # (512, 4)
    
    # sampled_boxes = roidb['boxes'][:512]

    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    blob_dict = dict(rois=sampled_rois)

    return blob_dict


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.MODEL.BBOX_REG_WEIGHTS)
    # Use class "1" for all fg boxes if using class_agnostic_bbox_reg
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        labels.clip(max=1, out=labels)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn_utils.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn_utils.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('rois')
    
