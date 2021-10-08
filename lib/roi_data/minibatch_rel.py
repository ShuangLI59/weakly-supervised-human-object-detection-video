import numpy as np
import cv2

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn
import os
import pdb

def get_minibatch_blob_names(is_training=True):

    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']

    if cfg.RPN.RPN_ON:
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    
    blobs = {'data': [], 'human_mask': [], 'human_box': [], 'im_info': [], 'roidb': []}
    # Get the input image blob
    im_blob, blob_mask, im_scales, blob_human_boxes = _get_image_blob(roidb)
    blobs['data'] = im_blob
    blobs['human_mask'] = blob_mask
    blobs['human_box'] = blob_human_boxes

    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    
    return blobs, True



def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    
    frames_info = roidb['frames_info']
    video_info = roidb['video_info']

    num_images = len(frames_info)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []

    processed_masks = []
    processed_human_boxes = []

    for i in range(num_images):
        # im = cv2.imread( os.path.join(video_info['data_root'], frames_info[i]['file_name']) )
        im = cv2.imread(  frames_info[i]['file_name']) 
        assert im is not None, \
            'Failed to read image \'{}\''.format(os.path.join(video_info['data_root'], frames_info[i]['file_name']))
        
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale, mask, human_box_scales = blob_utils.prep_im_for_blob(im, frames_info[i], cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
        processed_masks.append(mask[0])
        processed_human_boxes.append(human_box_scales[0])

    blob, blob_mask, blob_human_boxes = blob_utils.im_list_to_blob(processed_ims, processed_masks, processed_human_boxes)

    return blob, blob_mask, im_scales, blob_human_boxes










