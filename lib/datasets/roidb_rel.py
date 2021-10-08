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

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import six
import logging
import numpy as np

import utils.boxes as box_utils
import utils.blob as blob_utils
from core.config import cfg
from .json_dataset_rel import JsonDataset
import pdb

logger = logging.getLogger(__name__)


def combined_roidb_for_training(dataset_names):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    
    def get_roidb(dataset_name):
        ds = JsonDataset(dataset_name)
        roidb = ds.get_roidb(
            gt=True,
            crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH
        )
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        
        return [roidb, ds.category_to_id_map, ds.prd_category_to_id_map]

    if isinstance(dataset_names, six.string_types):
        dataset_names = (dataset_names, )
    
    results = [get_roidb(*args) for args in zip(dataset_names)]
    
    roidbs = [tem[0] for tem in results]
    category_to_id_map = [tem[1] for tem in results]
    prd_category_to_id_map = [tem[2] for tem in results]

    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    
    if cfg.TRAIN.ASPECT_GROUPING or cfg.TRAIN.ASPECT_CROPPING:
        logger.info('Computing image aspect ratios and ordering the ratios...')
        ratio_list, ratio_index = rank_for_training(roidb)
        logger.info('done')
    else:
        ratio_list, ratio_index = None, None

    return roidb, ratio_list, ratio_index, category_to_id_map, prd_category_to_id_map



def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """

    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    
    for video_name, entries in roidb.items():
        width = entries['video_info']['width']
        height = entries['video_info']['height']
        ratio = width / float(height)
        assert ratio<=RATIO_HI or  ratio>=RATIO_HI
        

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entries['video_info']['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entries['video_info']['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entries['video_info']['need_crop'] = False
        else:
            entries['video_info']['need_crop'] = False


        ratio_list.append(ratio)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',need_crop_cnt, RATIO_LO, RATIO_HI)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

