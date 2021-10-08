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
from eval.VOCevaldet_bboxpair import VOCevaldet_bboxpair


def mAP(predictions, ground_truth, IOU, eval_criteria):
    images = []
    for k,v in ground_truth.items():
        images += list(v.keys())
    images = list(np.unique(images))
                
    overlap_triplet = list(set(predictions.keys()) & set(ground_truth.keys()))
    print('%d overlapped triplets (prediction and ground truth)' % len(overlap_triplet))
    predictions = {k:v for k,v in predictions.items() if k in overlap_triplet}
    ground_truth = {k:v for k,v in ground_truth.items() if k in overlap_triplet}


    triplets = list(ground_truth.keys())

    AP = np.zeros([len(triplets), 1])
    REC = np.zeros([len(triplets), 1])

    for i, triplet in enumerate(triplets):

        det_id = []
        det_bb = []
        det_conf = []

        for idx, image in enumerate(images):
            
            if image in predictions[triplet]:
                if predictions[triplet][image] is not None:
                    prediction = predictions[triplet][image]
                    num_det = len(prediction)

                    if 'ko' in eval_criteria:
                        if image not in ground_truth[triplet]:
                            continue

                    if len(det_id)==0:
                        det_id = idx * np.ones([num_det, 1])
                        det_bb = prediction[:, :8]
                        det_conf = prediction[:, 8]
                    else:
                        det_id = np.concatenate((det_id, idx * np.ones([num_det, 1])), axis=0)
                        det_bb = np.concatenate((det_bb, prediction[:, :8]), axis=0)
                        det_conf = np.concatenate((det_conf, prediction[:, 8]), axis=0)
            else:
                print('%s is not in the prediction' % image)

        
        gt = []
        for idx, image in enumerate(images):
            if image in ground_truth[triplet]:
                gt.append(ground_truth[triplet][image])
            else:
                gt.append([])

    


        rec, prec, ap = VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt, IOU, eval_criteria)

        AP[i] = ap
        if len(rec)>0:
            REC[i] = rec[-1]
        else:
            print('rec empty, ap: ', ap)
        # print('%03d: ap: %.4f  rec: %.4f  (%s)' % (i, ap, REC[i], triplet))

    
    print('%s: mAP / mRec (full): %.4f / %.4f' % (eval_criteria, np.mean(AP)*100, np.mean(REC)*100))

    



















