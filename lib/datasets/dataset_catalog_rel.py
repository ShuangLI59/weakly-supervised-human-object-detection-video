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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
ANN_FN2 = 'annotation_file2'
ANN_FN3 = 'predicate_file'
# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'


# -------------------------------------------------------
## VHICO
# -------------------------------------------------------
ROOT = ''
GT_DATA_ROOT = os.path.join(ROOT, 'data')
DATA_ROOT = os.path.join(ROOT, 'data/video_256_30fps/16frames')
IMG_DIR_TEST = os.path.join(DATA_ROOT, 'test_16frames')
IMG_DIR_UNSEEN = os.path.join(DATA_ROOT, 'unseen_16frames')

WORD2VEC_GOOGLE = 'data/word2vec/GoogleNews-vectors-negative300.bin'
WORD2VEC_TEST = 'data/word2vec/obj_prd_w2v.h5'
WORD2VEC_UNSEEN = 'data/word2vec/obj_prd_w2v_unseen.h5'

TRIPLET_TRAIN = 'data/triplet/triplet_video_name_dict_train.p'
TRIPLET_VAL = 'data/triplet/triplet_video_name_dict_val.p'
TRIPLET_TEST = 'data/triplet/triplet_video_name_dict_test.p'
TRIPLET_UNSEEN = 'data/triplet/triplet_video_name_dict_unseen.p'


NUM_OBJ = 193
NUM_PRD = 94

NUM_OBJ_UNSEEN = 51
NUM_PRD_UNSEEN = 32


GT_PATH_TEST = os.path.join(ROOT, 'data/gt_annotations/gt_bbox_test.json')
GT_PATH_UNSEEN = os.path.join(ROOT, 'data/gt_annotations/gt_bbox_unseen.json')


# Available datasets
DATASETS = {
    'vhico_train': {
        IM_DIR: DATA_ROOT,
    },
    'vhico_val': {
        IM_DIR: DATA_ROOT,
    },
    'vhico_test': {
        IM_DIR: DATA_ROOT,
    },
    'vhico_unseen': {
        IM_DIR: DATA_ROOT,
    }
}
