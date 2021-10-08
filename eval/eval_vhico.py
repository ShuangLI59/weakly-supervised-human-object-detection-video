import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import pickle
import numpy as np
import pdb
import cv2
import torch
from tqdm import tqdm
import torch.nn.functional as F
import argparse

from eval.eval_recall import recall
from eval.eval_mAP import mAP
from eval.eval_utils import *

def convert_format_mAP(eval_input, annotations, eval_subset, SCORE_TYPE):
    ground_truth = convert_ground_truth_format_mAP(annotations, eval_subset)
    predictions = convert_prediction_format_mAP(eval_input, eval_subset, TOPK_BOX=10, SCORE_TYPE=SCORE_TYPE)
    print('mAP: convert prediction/ground truth format done!!!')
    return predictions, ground_truth


def convert_format_recall(eval_input, annotations, eval_subset, SCORE_TYPE):
    ground_truth = convert_ground_truth_format_recall(annotations, eval_subset)
    predictions = convert_prediction_format_recall(eval_input, eval_subset, RECALL_TOPK=1, SCORE_TYPE=SCORE_TYPE)
    print('Recall: convert prediction/ground truth format done!!!')
    return predictions, ground_truth


def vhico_eval(cfg, eval_subset='test', eval_input=None, GT_PATH_TEST=None, GT_PATH_UNSEEN=None):
    if eval_subset=='test':
        annotations = pickle.load( open(GT_PATH_TEST, 'rb') )
    elif eval_subset=='unseen':
        annotations = pickle.load( open(GT_PATH_UNSEEN, 'rb') )
    else:
        error('please select from test/unseen')

    
    SCORE_TYPE = 'original'
    ## ---------------------------------------------------------------------------
    ## convert format
    ## ---------------------------------------------------------------------------
    save_dir = cfg.results_dir
    if cfg.EVAL_MAP:
        if os.path.exists('%s/%s_prediction_convert_map_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower())):
            print('loading prediction and ground truth')
            predictions = pickle.load( open('%s/%s_prediction_convert_map_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'rb') )
            ground_truth = pickle.load( open('%s/%s_ground_truth_convert_map_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'rb') )
        else:
            print('converting prediction/ground truth format')
            predictions, ground_truth = convert_format_mAP(eval_input, annotations, eval_subset=eval_subset, SCORE_TYPE=SCORE_TYPE)
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump( predictions, open('%s/%s_prediction_convert_map_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'wb') )
            pickle.dump( ground_truth, open('%s/%s_ground_truth_convert_map_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'wb') )
    
        for eval_criteria in ['phrase_ko', 'phrase_def', 'relation_ko', 'relation_def']:
            mAP_result = mAP(predictions, ground_truth, cfg.IOU, eval_criteria)
    
    else:
        if os.path.exists('%s/%s_prediction_convert_recall_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower())):
            print('loading prediction and ground truth')
            predictions = pickle.load( open('%s/%s_prediction_convert_recall_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'rb') )
            ground_truth = pickle.load( open('%s/%s_ground_truth_convert_recall_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'rb') )
        else:
            print('converting prediction/ground truth format')
            predictions, ground_truth = convert_format_recall(eval_input, annotations, eval_subset=eval_subset, SCORE_TYPE=SCORE_TYPE)
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump( predictions, open('%s/%s_prediction_convert_recall_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'wb') )
            pickle.dump( ground_truth, open('%s/%s_ground_truth_convert_recall_%s.p' % (save_dir, eval_subset, SCORE_TYPE.lower()), 'wb') )
    
        frame_recall, video_one_recall, video_all_recall = {}, {}, {}
        for eval_criteria in ['phrase_ko', 'relation_ko']:
            frame_recall[eval_criteria], video_one_recall[eval_criteria], video_all_recall[eval_criteria] = recall(predictions, ground_truth, cfg.IOU, eval_criteria, RECALL_TOPK=1)    
        
        return frame_recall['phrase_ko']



def main():
    parser = argparse.ArgumentParser(description='Weakly Supervised Human-Object Interaction Detection in Video via Contrastive Spatiotemporal Regions')
    parser.add_argument('--IOU', default=0.3, type=float)
    parser.add_argument('--EVAL_MAP', default=1, type=int)
    parser.add_argument('--eval_subset', default='test', type=str)
    parser.add_argument('--results_dir', default='results/cat+Spa+Hum+Tem+Con', type=str)
    cfg = parser.parse_args()

    GT_PATH_TEST = 'data/gt_annotations/gt_bbox_test.json'
    GT_PATH_UNSEEN = 'data/gt_annotations/gt_bbox_unseen.json'
    
    eval_result = vhico_eval(cfg, cfg.eval_subset, GT_PATH_TEST=GT_PATH_TEST, GT_PATH_UNSEEN=GT_PATH_UNSEEN)
    

if __name__ == "__main__":
    main()













