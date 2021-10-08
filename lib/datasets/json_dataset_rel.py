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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
import cv2
import random
import torch
import pdb
from scipy.spatial import distance

import utils.env as envu
envu.set_up_matplotlib()
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog_rel import DATASETS, IM_DIR, DATA_ROOT, TRIPLET_TRAIN, TRIPLET_VAL, TRIPLET_TEST, TRIPLET_UNSEEN
from .VHICO_dataloader import VHICO

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):
        
        assert name in DATASETS.keys(), 'Unknown dataset name: {}'.format(name)
        logger.debug('Creating: {}'.format(name))

        self.name = name
        self.image_directory = [os.path.join(DATA_ROOT, 'training_16frames'), os.path.join(DATA_ROOT, 'validation_16frames'), os.path.join(DATA_ROOT, 'test_16frames'), os.path.join(DATA_ROOT, 'unseen_16frames')]


        if 'vhico' in name:
            self.VHICO = VHICO(name)

            if 'unseen' in self.name:
                category_ids = self.VHICO.getCatIds_unseen()
                categories = [c['name'] for c in self.VHICO.loadCats_unseen()]
                self.category_to_id_map = dict(zip(categories, category_ids))

                prd_category_ids = self.VHICO.getPreCatIds_unseen()
                prd_categories = [c['name'] for c in self.VHICO.loadPreCats_unseen()]
                self.prd_category_to_id_map = dict(zip(prd_categories, prd_category_ids))

                if os.path.exists(TRIPLET_UNSEEN):
                    self.video_name_triplet_dict = pickle.load(open(TRIPLET_UNSEEN, 'rb'))
                    for k,v in self.video_name_triplet_dict['triplet_id_name'].items():
                        prd_id = int(k.split('___')[0])
                        obj_id = int(k.split('___')[1])

                        prd_name = v.split('___')[0]
                        obj_name = v.split('___')[1]

                        assert prd_id==self.prd_category_to_id_map[prd_name]
                        assert obj_id==self.category_to_id_map[obj_name]

            else:
                category_ids = self.VHICO.getCatIds() # 0-7836
                categories = [c['name'] for c in self.VHICO.loadCats()]
                self.category_to_id_map = dict(zip(categories, category_ids))
                
                prd_category_ids = self.VHICO.getPreCatIds()
                prd_categories = [c['name'] for c in self.VHICO.loadPreCats()]
                self.prd_category_to_id_map = dict(zip(prd_categories, prd_category_ids))

                if os.path.exists(TRIPLET_TRAIN):
                    self.video_name_triplet_dict = pickle.load(open(TRIPLET_TRAIN, 'rb'))
                    for k,v in self.video_name_triplet_dict['triplet_id_name'].items():
                        prd_id = int(k.split('___')[0])
                        obj_id = int(k.split('___')[1])

                        prd_name = v.split('___')[0]
                        obj_name = v.split('___')[1]

                        assert prd_id==self.prd_category_to_id_map[prd_name]
                        assert obj_id==self.category_to_id_map[obj_name]

            self.densepose = self.VHICO.densepose
            self.densepose_score_th = 0.9
        
        

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0,
            split='training'
        ):
        
        
        if 'vhico' in self.name:
            if cfg.DEBUG:
                roidb = {tem:self.VHICO.data[tem] for tem in list(self.VHICO.data.keys())[:100]}
            else:
                roidb = self.VHICO.data

            if ('train' in self.name):
                TRIPLET_DICT = TRIPLET_TRAIN
            elif ('val' in self.name):
                TRIPLET_DICT = TRIPLET_VAL
            if ('test' in self.name):
                TRIPLET_DICT = TRIPLET_TEST
            if ('unseen' in self.name):
                TRIPLET_DICT = TRIPLET_UNSEEN

        
        missing_densepose = []
        total_frame = 0
        new_roidb = {}
        if not os.path.exists(TRIPLET_DICT):
            triplet_frame_dict = {}
            triplet_frame_dict['triplet_id_frame'] = {}
            triplet_frame_dict['triplet_id_name'] = {}

        for video_name, entries in roidb.items():
            if ('vhico' in self.name) and (len(entries)<12):
                continue
            
            assert entries[0]['width']==entries[-1]['width']
            assert entries[0]['height']==entries[-1]['height']
            
            for entry in entries:
                if 'vhico' in self.name:
                    entry['file_name'] = os.path.join(DATASETS[self.name][IM_DIR], '/'.join(entry['file_name'].split('/')[-4:]))
                    file_name = '/'.join(entry['file_name'].split('/')[-3:])
                
                if ('train' in self.name) or ('val' in self.name) or ('test' in self.name) or ('unseen' in self.name):
                    entry['obj_gt_cls'] = self.category_to_id_map[entry['obj_name']]
                    entry['prd_gt_cls'] = self.prd_category_to_id_map[entry['action_name']]
                    entry['dataset'] = self.name

                    entry['obj_gt_cls_name'] = entry['obj_name']
                    entry['prd_gt_cls_name'] = entry['action_name']

                else:
                    error('please select from train/val/test/unseen')
            
                
                if file_name in self.densepose:
                    densepose_boxes = self.densepose[file_name]['boxes']
                    densepose_mask = self.densepose[file_name]['mask']
                    
                    entry['densepose_boxes'] = densepose_boxes
                    entry['densepose_mask'] = densepose_mask


                else:
                    entry['densepose_boxes'] = np.zeros([1,5])
                    entry['densepose_mask'] = np.zeros([256, 256])

                    missing_densepose.append(file_name)
                total_frame += 1
                    

                if not os.path.exists(TRIPLET_DICT):
                    triplet_id = '%d___%d' % (entry['prd_gt_cls'], entry['obj_gt_cls'])
                    triplet_name = '%s___%s' % (entry['action_name'], entry['obj_name'])
                    
                    if triplet_id not in triplet_frame_dict['triplet_id_frame']:
                        triplet_frame_dict['triplet_id_frame'][triplet_id] = []
                    triplet_frame_dict['triplet_id_frame'][triplet_id].append(entry['file_name'])
                    
                    if triplet_id in triplet_frame_dict['triplet_id_name']:
                        assert triplet_frame_dict['triplet_id_name'][triplet_id] == triplet_name
                    else:
                        triplet_frame_dict['triplet_id_name'][triplet_id] = triplet_name


            index = 0
            for i in range(len(entries)):
                entries_new = {}
                entries_new['frames_info'] = [entries[i]]
                entries_new['video_info'] = {'width': entries[index]['width'], 'height': entries[index]['height'], 'data_root': DATASETS[self.name][IM_DIR]}
                new_roidb[video_name+"|{}".format(i)] = entries_new


        print('%s missing densepose data %d/%d' % (self.name, len(missing_densepose), total_frame) )

        if not os.path.exists(TRIPLET_DICT):
            print('writing triplet dict %s' % TRIPLET_DICT)
            pickle.dump(triplet_frame_dict, open(TRIPLET_DICT, 'wb'))

        
        roidb = new_roidb

        return roidb
