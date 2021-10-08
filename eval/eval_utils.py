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
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


unseen_prediction_map_name_dict = {'slicing fresh cherry tomatoe': 'slicing fresh cherry tomatoes',
                                    'picking raw coffee bean': 'picking raw coffee beans',
                                    'folding bike fold bike': 'folding bikes',
                                    'putting golden coin': 'putting golden coins'}

unseen_ground_truth_map_name_dict = {'folding bikes folding bikes': 'folding bikes'}


test_prediction_map_name_dict = {'filling eyebrow': 'filling eyebrows',
                                'throwing be': 'throwing axe',
                                'snapping finger': 'snapping fingers',
                                'inflating tire': 'inflating tires',
                                'knocking door': 'knocking doors',
                                'slicing onion': 'slicing onions', 
                                'sweeping leave': 'sweeping leaves', 
                                'washing hand': 'washing hands', 
                                'dipping doughnut': 'dipping doughnuts', 
                                'filling car': 'filling cars', 
                                'shaking hand': 'shaking hands', 
                                'chopping onion': 'chopping onions', 
                                'juggling snowball': 'juggling snowballs', 
                                'eating cookie': 'eating cookies', 
                                'picking tomatoe': 'picking tomatoes', 
                                'picking card': 'picking cards', 
                                'picking mushroom': 'picking mushrooms', 
                                'picking flower': 'picking flowers', 
                                'repairing car': 'repairing cars', 
                                'stretching arm': 'stretching arms', 
                                'injecting orange': 'injecting oranges',
                                'spreading arm': 'spreading arms', 
                                'smashing car': 'smashing cars',
                                'pouring glass': 'pouring glasses', 
                                'removing bread': 'removing breads', 
                                'placing stamp': 'placing stamps', 
                                'placing flower': 'placing flowers', 
                                'tying shoelace': 'tying shoelaces', 
                                'tying shoe': 'tying shoes',
                                'tearing petal': 'tearing petals', 
                                'unpacking grocery': 'unpacking groceries', 
                                'unpacking book': 'unpacking books', 
                                'unpacking suitcase': 'unpacking suitcases', 
                                'stacking cup': 'stacking cups', 
                                'rubbing foot': 'rubbing feet', 
                                'shaving leg': 'shaving legs', 
                                'pressing button': 'pressing buttons', 
                                'pulling carrot': 'pulling carrots', 
                                'planting seed': 'planting seeds', 
                                'drying hand': 'drying hands', 
                                'brushing tooth': 'brushing teeth', 
                                'peeling vegetable': 'peeling vegetables', 
                                'peeling potatoe': 'peeling potatoes', 
                                'lifting weight': 'lifting weights',
                                'inflating balloon': 'inflating balloons', 
                                'peeling apple': 'peeling apples', 
                                'throwing ball': 'throwing balls', 
                                'hugging tree': 'hugging trees', 
                                'juggling ball': 'juggling balls', 
                                'sweeping floor': 'sweeping floors', 
                                'buttoning button': 'buttoning buttons', 
                                'flipping pancake': 'flipping pancakes', 
                                'stretching leg': 'stretching legs', 
                                'shaking head': 'shaking heads', 
                                'mopping floor': 'mopping floors', 
                                'taping foot': 'taping feet', 
                                'massaging foot': 'massaging feet', 
                                'combing hair': 'combing hairs', 
                                'cracking egg': 'cracking eggs'}



test_ground_truth_map_name_dict = {'inflating balloon': 'inflating balloons', 
                                    'peeling apple': 'peeling apples', 
                                    'throwing ball': 'throwing balls', 
                                    'hugging tree': 'hugging trees', 
                                    'juggling ball': 'juggling balls', 
                                    'sweeping floor': 'sweeping floors', 
                                    'buttoning button': 'buttoning buttons', 
                                    'flipping pancake': 'flipping pancakes', 
                                    'stretching leg': 'stretching legs', 
                                    'shaking head': 'shaking heads', 
                                    'mopping floor': 'mopping floors', 
                                    'taping foot': 'taping feet', 
                                    'massaging foot': 'massaging feet', 
                                    'combing hair': 'combing hairs', 
                                    'cracking egg': 'cracking eggs'}





def get_boxes_sort(object_prediction, human_prediction, TOPK_BOX, SCORE_TYPE):
    ## binary score
    pred_binary_preds = object_prediction['binary_preds'].cpu()
    pred_binary_preds = F.softmax(pred_binary_preds)
    assert pred_binary_preds.shape[0]==2
    
    pred_object_scores = object_prediction['scores_ori']
    pred_human_scores = human_prediction['scores_ori']
    
    ## multiple binary weight
    pred_object_scores = pred_object_scores*pred_binary_preds[1]
    pred_human_scores = pred_human_scores*pred_binary_preds[1]
    scores = pred_object_scores[:, np.newaxis] + pred_human_scores

    index = np.array(np.zeros(len(pred_object_scores)))[:, np.newaxis] + np.array(range(len(pred_human_scores)))
    index2 = np.array(range(len(pred_object_scores)))[:, np.newaxis] + np.array(np.zeros(len(pred_human_scores)))

    scores = scores.flatten()
    index = index.flatten()
    index2 = index2.flatten()
    top_index = scores.sort(descending=True)[1]

    if pred_binary_preds[1] > -np.inf:
        all_boxes = []
        for idx in range(TOPK_BOX):
            object_index = int(index2[top_index[idx]])
            human_index = int(index[top_index[idx]])
        
            object_bb = object_prediction['boxes'][object_index][1:]
            human_bb = human_prediction['boxes'][human_index][1:]

            oh_box = torch.cat([human_bb, object_bb])
            oh_box = torch.cat([oh_box, torch.tensor([scores[top_index[idx]]])])
            
            all_boxes.append(oh_box)
        all_boxes = np.stack(all_boxes)
    else:
        all_boxes = None

    return all_boxes



def convert_prediction_format_mAP(eval_input, eval_subset, TOPK_BOX, SCORE_TYPE):
    predictions_all_triplet = eval_input['predictions_object_bbox']
    human_predictions_all_triplet = eval_input['predictions_human_bbox']
    video_name_triplet_dict = eval_input['video_name_triplet_dict']

    predictions = {}
    for triplet_id, _ in predictions_all_triplet.items():
        triplet_name = video_name_triplet_dict['triplet_id_name'][triplet_id]
        triplet_name = ' '.join(triplet_name.split('___'))

        if eval_subset=='unseen':
            if triplet_name in unseen_prediction_map_name_dict:
                triplet_name = unseen_prediction_map_name_dict[triplet_name]
        elif eval_subset=='test':
            if triplet_name in test_prediction_map_name_dict:
                triplet_name = test_prediction_map_name_dict[triplet_name]

        print(triplet_name)
        predictions[triplet_name] = {}

        for frame_name, _ in predictions_all_triplet[triplet_id].items():
            object_prediction = predictions_all_triplet[triplet_id][frame_name]
            human_prediction = human_predictions_all_triplet[triplet_id][frame_name]

            all_boxes = get_boxes_sort(object_prediction, human_prediction, TOPK_BOX, SCORE_TYPE)
            predictions[triplet_name][frame_name] = all_boxes

    return predictions




def convert_ground_truth_format_mAP(annotations, eval_subset):
    ground_truth = {}
    for action, _ in annotations['annos'].items():
        videos = annotations['annos'][action]
        for video_name, annos in videos.items():
            for frame_anno in annos:
                frame_name = '/'.join(frame_anno['path'].split('/')[-3:])
                triplet_name = frame_anno['label']
                triplet_name = triplet_name.lower()

                if eval_subset=='unseen':
                    if triplet_name in unseen_ground_truth_map_name_dict:
                        triplet_name = unseen_ground_truth_map_name_dict[triplet_name]
                elif eval_subset=='test':
                    if triplet_name in test_ground_truth_map_name_dict:
                        triplet_name = test_ground_truth_map_name_dict[triplet_name]

                if triplet_name not in ground_truth:
                    ground_truth[triplet_name] = {}

                object_bboxes = frame_anno['object']
                human_bboxes = frame_anno['human']

                bboxes = []
                for human_bbox in human_bboxes:
                    for object_bbox in object_bboxes:
                        bboxes.append(np.concatenate((human_bbox, object_bbox), axis=0))
                bboxes = np.stack(bboxes)

                assert frame_name not in ground_truth[triplet_name]
                ground_truth[triplet_name][frame_name] = bboxes

    return ground_truth



def convert_prediction_format_recall(eval_input, eval_subset, RECALL_TOPK, SCORE_TYPE):
    predictions_all_triplet = eval_input['predictions_object_bbox']
    human_predictions_all_triplet = eval_input['predictions_human_bbox']
    video_name_triplet_dict = eval_input['video_name_triplet_dict']

    predictions_all_triplet = predictions_all_triplet['gt_label']
    human_predictions_all_triplet = human_predictions_all_triplet['gt_label']

    predictions = {}
    for frame_name, _ in predictions_all_triplet.items():
        object_prediction = predictions_all_triplet[frame_name]
        human_prediction = human_predictions_all_triplet[frame_name]

        all_boxes = get_boxes_sort(object_prediction, human_prediction, RECALL_TOPK, SCORE_TYPE)
        predictions[frame_name] = all_boxes

    return predictions



def convert_ground_truth_format_recall(annotations, eval_subset):
    ground_truth = {}
    for action, _ in annotations['annos'].items():
        videos = annotations['annos'][action]
        for video_name, annos in videos.items():
            for frame_anno in annos:
                frame_name = '/'.join(frame_anno['path'].split('/')[-3:])
                object_bboxes = frame_anno['object']
                human_bboxes = frame_anno['human']

                bboxes = []
                for human_bbox in human_bboxes:
                    for object_bbox in object_bboxes:
                        bboxes.append(np.concatenate((human_bbox, object_bbox), axis=0))
                bboxes = np.stack(bboxes)

                assert frame_name not in ground_truth
                ground_truth[frame_name] = bboxes

    return ground_truth


def compute_overlap(bb_1, bb_2, bbgt_1, bbgt_2, eval_criteria):
    if 'phrase' in eval_criteria:
        size = 500
        assert np.max(bb_1)<size
        assert np.max(bb_2)<size
        assert np.max(bbgt_1)<size
        assert np.max(bbgt_2)<size
        
        mask1 = np.zeros([size, size])
        mask1[int(bb_1[0]):int(bb_1[2]), int(bb_1[1]):int(bb_1[3])] = 1

        mask2 = np.zeros([size, size])
        mask2[int(bb_2[0]):int(bb_2[2]), int(bb_2[1]):int(bb_2[3])] = 1

        mask_union = mask1+mask2
        mask_union = mask_union>=1
        
        ## gt
        mask1 = np.zeros([size, size])
        mask1[int(bbgt_1[0]):int(bbgt_1[2]), int(bbgt_1[1]):int(bbgt_1[3])] = 1

        mask2 = np.zeros([size, size])
        mask2[int(bbgt_2[0]):int(bbgt_2[2]), int(bbgt_2[1]):int(bbgt_2[3])] = 1
        gt_mask_union = mask1+mask2
        gt_mask_union = gt_mask_union>=1
        
        ## prediction and ground truth
        gt_pred_mask = mask_union.astype(int) + gt_mask_union.astype(int)
        ov = len(np.where(gt_pred_mask==2)[0])/(len(np.where(gt_pred_mask==1)[0])+len(np.where(gt_pred_mask==2)[0]))
        
    elif 'relation' in eval_criteria:
        # % compare box 1
        bi_1 = [np.max([bb_1[0], bbgt_1[0]]), np.max([bb_1[1], bbgt_1[1]]), np.min([bb_1[2], bbgt_1[2]]), np.min([bb_1[3], bbgt_1[3]])]
        iw_1 = bi_1[2]-bi_1[0]+1
        ih_1 = bi_1[3]-bi_1[1]+1

        if iw_1 > 0 and ih_1 > 0:
            # % compute overlap as area of intersection / area of union
            ua_1 = (bb_1[2]-bb_1[0]+1)*(bb_1[3]-bb_1[1]+1) + (bbgt_1[2]-bbgt_1[0]+1)*(bbgt_1[3]-bbgt_1[1]+1) - iw_1*ih_1
            ov_1 = iw_1*ih_1/ua_1
        else:
            ov_1 = 0

        # % compare box 2
        bi_2 = [np.max([bb_2[0], bbgt_2[0]]), np.max([bb_2[1], bbgt_2[1]]), np.min([bb_2[2], bbgt_2[2]]), np.min([bb_2[3], bbgt_2[3]])]
        iw_2 = bi_2[2]-bi_2[0]+1
        ih_2 = bi_2[3]-bi_2[1]+1
        if iw_2 > 0 and ih_2 > 0:
            # % compute overlap as area of intersection / area of union
            ua_2 = (bb_2[2]-bb_2[0]+1)*(bb_2[3]-bb_2[1]+1) + (bbgt_2[2]-bbgt_2[0]+1)*(bbgt_2[3]-bbgt_2[1]+1) - iw_2*ih_2
            ov_2 = iw_2*ih_2/ua_2
        else:
            ov_2 = 0

        # % get minimum
        min_ov = np.min([ov_1, ov_2])
        ov = min_ov
    
    return ov


