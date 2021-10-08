import os
import sys
import pdb
from copy import deepcopy
from torch.autograd import Variable
import utils.blob as blob_utils

from eval.eval_vhico import *
from lib.datasets.dataset_catalog_rel import GT_PATH_TEST, GT_PATH_UNSEEN

def run_eval(args, cfg, maskRCNN, dataloader_test, step, output_dir, test_stats, best_eval_result, eval_subset='test'):

    is_best = False
    maskRCNN.eval()

    print('------------------------------------------------------------------------------------------------------------')
    print('eval %s: %s' % (eval_subset, 'mAP' if cfg.EVAL_MAP else 'recall') )
    print('------------------------------------------------------------------------------------------------------------')
    
    ### -------------------------------------------------------------------------------------------------------------------
    # get results
    ### -------------------------------------------------------------------------------------------------------------------
    file_names = {}
    boxes = {}
    scores = {}
    scores_ori = {}

    human_file_names = {}
    human_boxes = {}
    human_scores = {}
    human_scores_ori = {}

    obj_gt_cls_names = {}
    prd_gt_cls_names = {}
    
    obj_gt_cls_ids = {}
    prd_gt_cls_ids = {}

    if cfg.BINARY_LOSS:
        binary_preds = {}

    
    for i, input_data in enumerate(dataloader_test):
        for key in input_data:
            if key != 'roidb':
                input_data[key] = list(map(Variable, input_data[key]))

        if len(input_data['im_info'])!=cfg.NUM_GPUS:
            print(len(input_data['im_info']))

        net_outputs_dict = maskRCNN(**input_data)
        
        for triplet_name in net_outputs_dict.keys():
            net_outputs = deepcopy(net_outputs_dict[triplet_name])
            
            for gpu_i in range(cfg.NUM_GPUS):
                if triplet_name not in boxes:
                    boxes[triplet_name] = []
                    scores[triplet_name] = []
                    scores_ori[triplet_name] = []

                    human_boxes[triplet_name] = []
                    human_scores[triplet_name] = []
                    human_scores_ori[triplet_name] = []

                    obj_gt_cls_names[triplet_name] = []
                    prd_gt_cls_names[triplet_name] = []

                    obj_gt_cls_ids[triplet_name] = []
                    prd_gt_cls_ids[triplet_name] = []
                    
                    file_names[triplet_name] = []
                    human_file_names[triplet_name] = []
                    
                    if cfg.BINARY_LOSS:
                        binary_preds[triplet_name] = []

                boxes[triplet_name] += [box[  (gpu_i)*cfg.TRAIN.BATCH_SIZE_PER_IM : (gpu_i+1)*cfg.TRAIN.BATCH_SIZE_PER_IM, :] for box in net_outputs['predictions']['box']]
                scores[triplet_name] += [score[  (gpu_i)*cfg.TRAIN.BATCH_SIZE_PER_IM : (gpu_i+1)*cfg.TRAIN.BATCH_SIZE_PER_IM] for score in net_outputs['predictions']['score']]
                scores_ori[triplet_name] += [score_ori[  (gpu_i)*cfg.TRAIN.BATCH_SIZE_PER_IM : (gpu_i+1)*cfg.TRAIN.BATCH_SIZE_PER_IM] for score_ori in net_outputs['predictions']['score_ori']]
                
                assert len(net_outputs['predictions']['box'][0])==cfg.TRAIN.BATCH_SIZE_PER_IM*cfg.NUM_GPUS
                assert len(net_outputs['predictions']['score'][0])==cfg.TRAIN.BATCH_SIZE_PER_IM*cfg.NUM_GPUS
                assert len(net_outputs['predictions']['score_ori'][0])==cfg.TRAIN.BATCH_SIZE_PER_IM*cfg.NUM_GPUS

                file_name = blob_utils.deserialize(net_outputs['predictions']['files'].numpy())
                obj_gt_cls_name = blob_utils.deserialize(net_outputs['predictions']['obj_gt_cls_name'].numpy())
                prd_gt_cls_name = blob_utils.deserialize(net_outputs['predictions']['prd_gt_cls_name'].numpy())
                obj_gt_cls = blob_utils.deserialize(net_outputs['predictions']['obj_gt_cls'].numpy())
                prd_gt_cls = blob_utils.deserialize(net_outputs['predictions']['prd_gt_cls'].numpy())

                file_names[triplet_name] += file_name
                obj_gt_cls_names[triplet_name] += obj_gt_cls_name
                prd_gt_cls_names[triplet_name] += prd_gt_cls_name
                obj_gt_cls_ids[triplet_name] += obj_gt_cls
                prd_gt_cls_ids[triplet_name] += prd_gt_cls
                
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['predictions']['files'].numpy())))
                net_outputs['predictions']['files'] = net_outputs['predictions']['files'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['predictions']['obj_gt_cls_name'].numpy())))
                net_outputs['predictions']['obj_gt_cls_name'] = net_outputs['predictions']['obj_gt_cls_name'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['predictions']['prd_gt_cls_name'].numpy())))
                net_outputs['predictions']['prd_gt_cls_name'] = net_outputs['predictions']['prd_gt_cls_name'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['predictions']['obj_gt_cls'].numpy())))
                net_outputs['predictions']['obj_gt_cls'] = net_outputs['predictions']['obj_gt_cls'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['predictions']['prd_gt_cls'].numpy())))
                net_outputs['predictions']['prd_gt_cls'] = net_outputs['predictions']['prd_gt_cls'][len_gpu_i:]

                if cfg.BINARY_LOSS:
                    binary_preds[triplet_name] += [binary_pred[  (gpu_i)*2 : (gpu_i+1)*2] for binary_pred in net_outputs['predictions']['binary_pred']]
                

                # human
                num_roi = cfg.MAX_NUM_HUMAN
                human_boxes[triplet_name] += [box[  (gpu_i)*num_roi : (gpu_i+1)*num_roi, :] for box in net_outputs['human_predictions']['box']]
                human_scores[triplet_name] += [score[  (gpu_i)*num_roi : (gpu_i+1)*num_roi] for score in net_outputs['human_predictions']['score']]
                human_scores_ori[triplet_name] += [score_ori[  (gpu_i)*num_roi : (gpu_i+1)*num_roi] for score_ori in net_outputs['human_predictions']['score_ori']]

                human_file_name = blob_utils.deserialize(net_outputs['human_predictions']['files'].numpy())
                human_obj_gt_cls_name = blob_utils.deserialize(net_outputs['human_predictions']['obj_gt_cls_name'].numpy())
                human_prd_gt_cls_name = blob_utils.deserialize(net_outputs['human_predictions']['prd_gt_cls_name'].numpy())
                obj_gt_cls = blob_utils.deserialize(net_outputs['human_predictions']['obj_gt_cls'].numpy())
                prd_gt_cls = blob_utils.deserialize(net_outputs['human_predictions']['prd_gt_cls'].numpy())
                human_file_names[triplet_name] += human_file_name
                
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['human_predictions']['files'].numpy())))
                net_outputs['human_predictions']['files'] = net_outputs['human_predictions']['files'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['human_predictions']['obj_gt_cls_name'].numpy())))
                net_outputs['human_predictions']['obj_gt_cls_name'] = net_outputs['human_predictions']['obj_gt_cls_name'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['human_predictions']['prd_gt_cls_name'].numpy())))
                net_outputs['human_predictions']['prd_gt_cls_name'] = net_outputs['human_predictions']['prd_gt_cls_name'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['human_predictions']['obj_gt_cls'].numpy())))
                net_outputs['human_predictions']['obj_gt_cls'] = net_outputs['human_predictions']['obj_gt_cls'][len_gpu_i:]
                len_gpu_i = len(blob_utils.serialize(blob_utils.deserialize(net_outputs['human_predictions']['prd_gt_cls'].numpy())))
                net_outputs['human_predictions']['prd_gt_cls'] = net_outputs['human_predictions']['prd_gt_cls'][len_gpu_i:]


                assert file_name==human_file_name
                assert obj_gt_cls_name==human_obj_gt_cls_name
                assert prd_gt_cls_name==human_prd_gt_cls_name

            assert len(scores[triplet_name])==len(scores_ori[triplet_name])==len(boxes[triplet_name])==len(file_names[triplet_name])
            assert len(human_scores[triplet_name])==len(human_boxes[triplet_name])==len(human_file_names[triplet_name])
            assert len(file_names[triplet_name])==len(obj_gt_cls_names[triplet_name])==len(prd_gt_cls_names[triplet_name])
        
    predictions_all_triplet = {}
    human_predictions_all_triplet = {}

    for triplet_name in net_outputs_dict.keys():
        predictions = {}
        for i, file_name in enumerate(file_names[triplet_name]):
            predictions[file_name] = {}
            predictions[file_name]['boxes'] = boxes[triplet_name][i]
            predictions[file_name]['scores'] = scores[triplet_name][i]
            predictions[file_name]['scores_ori'] = scores_ori[triplet_name][i]
            predictions[file_name]['obj_gt_cls_names'] = obj_gt_cls_names[triplet_name][i]
            predictions[file_name]['prd_gt_cls_names'] = prd_gt_cls_names[triplet_name][i]
            predictions[file_name]['obj_gt_cls_ids'] = obj_gt_cls_ids[triplet_name][i]
            predictions[file_name]['prd_gt_cls_ids'] = prd_gt_cls_ids[triplet_name][i]
            if cfg.BINARY_LOSS:
                predictions[file_name]['binary_preds'] = binary_preds[triplet_name][i]
        predictions_all_triplet[triplet_name] = predictions

        # human
        human_predictions = {}
        for i, file_name in enumerate(human_file_names[triplet_name]):
            human_predictions[file_name] = {}
            human_predictions[file_name]['boxes'] = human_boxes[triplet_name][i]
            human_predictions[file_name]['scores'] = human_scores[triplet_name][i]
            human_predictions[file_name]['scores_ori'] = human_scores_ori[triplet_name][i]
        human_predictions_all_triplet[triplet_name] = human_predictions

    eval_input = {}
    eval_input['predictions_object_bbox'] = predictions_all_triplet
    eval_input['predictions_human_bbox'] = human_predictions_all_triplet
    eval_input['video_name_triplet_dict'] = maskRCNN.module.video_name_triplet_dict
    

    # ------------------------------------------------------------------------------------------------------------
    # Compute Recall and mAP
    # ------------------------------------------------------------------------------------------------------------
    if 'vhico' in args.dataset:
        if not cfg.EVAL_MAP:
            frame_recall_phrase_ko = vhico_eval(cfg, eval_subset=eval_subset, eval_input=eval_input, GT_PATH_TEST=GT_PATH_TEST, GT_PATH_UNSEEN=GT_PATH_UNSEEN)
            
            test_stats.tb_log_stats({'frame_recall_phrase_ko': frame_recall_phrase_ko}, step)
            if frame_recall_phrase_ko>best_eval_result:
                is_best = True
                best_eval_result = frame_recall_phrase_ko
                print('best test frame_recall_phrase_ko is %.4f at step %d' % (frame_recall_phrase_ko, step) )
        else:
            mAP_result = vhico_eval(cfg, eval_subset=eval_subset, eval_input=eval_input, GT_PATH_TEST=GT_PATH_TEST, GT_PATH_UNSEEN=GT_PATH_UNSEEN)

    ## set the model to training mode
    maskRCNN.train()

    return is_best, best_eval_result 


