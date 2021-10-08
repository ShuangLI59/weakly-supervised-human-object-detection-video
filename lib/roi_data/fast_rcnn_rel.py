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
import logging

from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import pdb

logger = logging.getLogger(__name__)


def add_rel_blobs(blobs, im_scales, roidb):
    
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_pairs(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
        
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_rel_multilevel_rois(blobs)

    return True


def _sample_pairs(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    
    sampled_obj_boxes = roidb['obj_boxes']
    sampled_obj_rois = sampled_obj_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_obj_boxes.shape[0], 1))
    sampled_obj_rois = np.hstack((repeated_batch_idx, sampled_obj_rois))

    blob_dict = {}
    blob_dict['obj_rois'] = sampled_obj_rois
    
    sampled_rel_rois = sampled_obj_rois
    
    blob_dict['rel_rois'] = sampled_rel_rois
    

    prd_gt_cls = np.zeros(1, dtype=np.int32)
    prd_gt_cls[0] = roidb['prd_gt_cls']
    blob_dict['prd_gt_cls'] = prd_gt_cls


    obj_gt_cls = np.zeros(1, dtype=np.int32)
    obj_gt_cls[0] = roidb['obj_gt_cls']
    blob_dict['obj_gt_cls'] = obj_gt_cls
    

    return blob_dict


def _add_rel_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_names):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        lowest_target_lvls = None
        for rois_blob_name in rois_blob_names:
            target_lvls = fpn_utils.map_rois_to_fpn_levels(
                blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max)
            if lowest_target_lvls is None:
                lowest_target_lvls = target_lvls
            else:
                lowest_target_lvls = np.minimum(lowest_target_lvls, target_lvls)
        for rois_blob_name in rois_blob_names:
            # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
            fpn_utils.add_multilevel_roi_blobs(
                blobs, rois_blob_name, blobs[rois_blob_name], lowest_target_lvls, lvl_min,
                lvl_max)

    _distribute_rois_over_fpn_levels(['obj_rois'])
    _distribute_rois_over_fpn_levels(['rel_rois'])
    

