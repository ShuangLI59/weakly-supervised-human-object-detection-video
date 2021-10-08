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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import cv2
import pdb

from core.config import cfg


def get_image_blob(im, target_scale, target_max_size):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, [target_scale], target_max_size
    )
    blob = im_list_to_blob(processed_im)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)


def im_list_to_blob(ims, masks, human_boxes):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    
    
    if not isinstance(ims, list):
        ims = [ims]

    if not isinstance(masks, list):
        masks = [masks]
        
    max_shape = get_max_shape([im.shape[:2] for im in ims])

    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    

    max_num_box = cfg.MAX_NUM_HUMAN
        
    blob_mask = np.zeros((num_images, max_shape[0], max_shape[1]), dtype=np.float32)
    blob_human_boxes = np.zeros((num_images, max_num_box, 5), dtype=np.float32)
    for i in range(num_images):
        tem = np.zeros([max_num_box, 5])
        if len(human_boxes[i])>max_num_box:
            human_boxes[i] = human_boxes[i][:max_num_box]
        tem[:len(human_boxes[i]), 1:5] = human_boxes[i]
        tem[:, 0] = len(human_boxes[i])*np.ones(max_num_box)
        blob_human_boxes[i] = tem
        

    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        blob_mask[i, 0:im.shape[0], 0:im.shape[1]] = masks[i]
    
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob, blob_mask, blob_human_boxes


def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
    return max_shape


def prep_im_for_blob(im, frames_info, pixel_means, target_sizes, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means # array([[[102.9801, 115.9465, 122.7717]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    
    ims = []
    im_scales = []
    mask_scales = []
    human_box_scales = []

    
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        
    
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    
        mask_tem=np.zeros([frames_info['densepose_mask'].shape[0], frames_info['densepose_mask'].shape[1], 1]) 
        mask_tem[:,:,0]=frames_info['densepose_mask']
        mask_scale = cv2.resize(mask_tem, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        human_box_scale = []
        for i,tem_box in enumerate(frames_info['densepose_boxes']): # (3, 5)
            box_mask_tem=np.zeros([frames_info['densepose_mask'].shape[0], frames_info['densepose_mask'].shape[1], 1])
            tem_box = tem_box[:4].astype(int)

            box_mask_tem[tem_box[1]:tem_box[3], tem_box[0]:tem_box[2], :] = 1
            box_mask_scale = cv2.resize(box_mask_tem, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

            try:
                xx,yy=np.where(box_mask_scale==1)
                xx_min = xx.min()
                xx_max = xx.max()
                yy_min = yy.min()
                yy_max = yy.max()
            except:
                xx_min = 0
                yy_min = 0
                xx_max = im_resized.shape[0]
                yy_max = im_resized.shape[1]
                
            human_box_scale.append(np.array([yy_min, xx_min, yy_max, xx_max]))

        human_box_scale = np.array(human_box_scale)


        ims.append(im_resized)
        im_scales.append(im_scale)
        mask_scales.append(mask_scale)
        human_box_scales.append(human_box_scale)

    return ims, im_scales, mask_scales, human_box_scales


def get_im_blob_sizes(im_shape, target_sizes, max_size):
    """Calculate im blob size for multiple target_sizes given original im shape
    """
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_sizes = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_sizes.append(np.round(im_shape * im_scale))
    return np.array(im_sizes)


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """Calculate target resize scale
    """
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
