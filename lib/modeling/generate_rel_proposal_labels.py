from torch import nn

from core.config import cfg
from datasets import json_dataset_rel
import roi_data.fast_rcnn_rel
import pdb

class GenerateRelProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obj_rois, det_rois, roidb, im_info, human_gts=None):
        
        im_scales = im_info.data.numpy()[:, 2]
        
        ## add obj_boxes and human_gt_boxes in roidb
        human_box_list = []
        obj_box_list = []
        for i, entry in enumerate(roidb):
            inv_im_scale = 1. / im_scales[i]
            idx = np.where(obj_rois[:, 0] == i)[0]
            
            obj_box_list.append(obj_rois[idx, 1:] * inv_im_scale)
            entry['obj_boxes'] = obj_box_list[i]

            if cfg.USE_HUMAN:
                human_box_list.append(human_gts[idx, 1:])
                entry['human_gt_boxes'] = human_box_list[i]
            

        
        output_blob_names = ['human_gt_rois', 'obj_rois', 'rel_rois', 'prd_gt_cls', 'obj_gt_cls']
        blobs = {k: [] for k in output_blob_names}
        
        roi_data.fast_rcnn_rel.add_rel_blobs(blobs, im_scales, roidb)

        return blobs