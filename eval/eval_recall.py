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
from eval.eval_utils import *


def recall(predictions, ground_truth, IOU, eval_criteria, RECALL_TOPK=1):
    assert len([tem for tem in ground_truth if tem not in predictions]) == 0
    frame_names = list(ground_truth.keys())

    frame_result = {}
    for index_frame in tqdm(range(len(frame_names))):
        frame_name = frame_names[index_frame]

        prediction_boxes = predictions[frame_name]
        gt_boxes = ground_truth[frame_name]

        count_matching_frame = 0
        for i, gt_box in enumerate(gt_boxes):
            gt_object_box = gt_box[:4]
            gt_human_box = gt_box[4:]

            for idx in range(RECALL_TOPK):
                prediction_box = prediction_boxes[idx]

                prediction_object_box = prediction_box[:4]
                prediction_human_box = prediction_box[4:]

                ov = compute_overlap(prediction_object_box, prediction_human_box, gt_object_box, gt_human_box, eval_criteria)
                if ov > IOU:
                    count_matching_frame += 1
                    break

        if count_matching_frame == len(gt_boxes):
            frame_result[frame_name] = True
        else:
            frame_result[frame_name] = False

    
    video_result = {}
    for frame_name, _ in frame_result.items():
        video_name = '/'.join(frame_name.split('/')[:-1])
        if video_name not in video_result:
            video_result[video_name] = []
        video_result[video_name].append(frame_result[frame_name])

    video_result = {k: np.sum(v)/len(v) for k,v in video_result.items()}
    video_one_result = {k:v for k,v in video_result.items() if v>0}
    video_all_result = {k:v for k,v in video_result.items() if v==1}
    

    frame_recall = np.sum(list(frame_result.values()))/len(frame_result)*100
    video_one_recall = len(video_one_result)/len(video_result)*100
    video_all_recall = len(video_all_result)/len(video_result)*100

    print('--------------------------------------------------------------------------------')
    print('IOU:', IOU, 'RECALL_TOPK:', RECALL_TOPK, 'eval_criteria:', eval_criteria)
    print('frame recall@1 = %.4f' % frame_recall)
    print('video one recall@1 = %.4f' % video_one_recall)
    print('video all recall@1 = %.4f' % video_all_recall)
    print('--------------------------------------------------------------------------------')


    return frame_recall, video_one_recall, video_all_recall
    




