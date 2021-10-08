import json
import numpy as np
import pdb
import os
import cv2
import re
import random
import collections
import operator
import pickle

import utils.boxes as box_utils
from core.config import cfg
from .dataset_catalog_rel import DATA_ROOT, GT_DATA_ROOT

class VHICO:
    def __init__(self, split):
        training_data_path = os.path.join(GT_DATA_ROOT, 'gt_annotations/gt_lang_training.json')

        if 'train' in split:
            data_path = os.path.join(GT_DATA_ROOT, 'gt_annotations/gt_lang_training.json')
            densepose_path = os.path.join(GT_DATA_ROOT, 'densepose/training-densepose-multimask-boxth0-maskth0.7.pkl')
        elif 'val' in split:
            data_path = os.path.join(GT_DATA_ROOT, 'gt_annotations/gt_lang_validation.json')
            densepose_path = os.path.join(GT_DATA_ROOT, 'densepose/validation-densepose-multimask-boxth0-maskth0.7.pkl')
        elif 'test' in split:
            data_path = os.path.join(GT_DATA_ROOT, 'gt_annotations/gt_lang_test.json')
            densepose_path = os.path.join(GT_DATA_ROOT, 'densepose/test-densepose-multimask-boxth0-maskth0.7.pkl')
        elif 'unseen' in split:
            data_path = os.path.join(GT_DATA_ROOT, 'gt_annotations/gt_lang_unseen.json')
            densepose_path = os.path.join(GT_DATA_ROOT, 'densepose/unseen-densepose-multimask-boxth0-maskth0.7.pkl')
        else:
            error('please select from train/val/test/unseen')
            
        self.img_dir = [os.path.join(DATA_ROOT, 'training_16frames'), os.path.join(DATA_ROOT, 'validation_16frames'), os.path.join(DATA_ROOT, 'test_16frames'), os.path.join(DATA_ROOT, 'unseen_16frames')]
        data, training_object_classes, training_action_classes = load_training_validation_data(split, data_path, training_data_path)

        self.training_object_classes = training_object_classes
        self.training_action_classes = training_action_classes

        if 'unseen' in split:
            self.unseen_object_classes = data['objects']
            self.unseen_action_classes = data['actions']
            
        self.data = data['data']
        self.objects = data['objects']
        self.actions = data['actions']
        self.num_frame = data['num_frame']
        self.num_video = data['num_video']
    
        self.densepose = pickle.load(open(densepose_path, "rb"))
        self.densepose = {('/'.join(k.split('/')[-3:])[:-3]+'jpg'):v for k,v in self.densepose.items()}
        
        
    def getCatIds(self):
        ids = list(range(len(self.training_object_classes)))
        return ids

    def getCatIds_unseen(self):
        ids = list(range(len(self.unseen_object_classes)))
        return ids

    def loadCats(self):
    	idNames = []
    	for i,k in enumerate(self.training_object_classes):
            idName = {'id':i, 'name':k}
            idNames.append(idName)
    	return idNames

    def loadCats_unseen(self):
        idNames = []
        for i,k in enumerate(self.unseen_object_classes):
            idName = {'id':i, 'name':k}
            idNames.append(idName)
        return idNames

    def getPreCatIds(self):
        ids = list(range(len(self.training_action_classes)))
        return ids

    def getPreCatIds_unseen(self):
        ids = list(range(len(self.unseen_action_classes)))
        return ids

    def loadPreCats(self):
        idNames = []
        for i,k in enumerate(self.training_action_classes):
            idName = {'id':i, 'name':k}
            idNames.append(idName)
        return idNames

    def loadPreCats_unseen(self):
        idNames = []
        for i,k in enumerate(self.unseen_action_classes):
            idName = {'id':i, 'name':k}
            idNames.append(idName)
        return idNames




def load_training_validation_data(split, data_path, training_data_path):
    print('------------------------------------------------------------------------------------------------------------------')
    print('loading %s data: ...' % split)

    if os.path.exists(data_path):
        output = pickle.load(open(data_path, 'rb') )
    else:
        print('cannot find %s' % data_path)

    print('%d videos' % output['num_video'])
    print('%d frames' % output['num_frame'])
    print('%d objects' % len(output['objects']))
    print('%d actions' % len(output['actions']))

    ## load training object classes and action classes
    train_data = pickle.load(open(training_data_path, 'rb') )
    
    object_classes = train_data['objects']
    action_classes = train_data['actions']
    
    return output, object_classes, action_classes

    



