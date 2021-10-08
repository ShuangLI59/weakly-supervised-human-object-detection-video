import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data.minibatch_rel import get_minibatch
import utils.blob as blob_utils
import random
import pdb

class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True, dataset=None):
        self._roidb = roidb
        self._roidb_keys = list(roidb.keys())
        self._num_classes = num_classes
        self.training = training
        self.dataset = dataset

        key_set = set()
        self.key_len = {}
        for key in roidb.keys():
            key_set.add(key.split("|")[0])

            if key.split("|")[0] not in self.key_len:
                self.key_len[key.split("|")[0]] = 0
            self.key_len[key.split("|")[0]] +=1 

        self._roidb_keys = list(key_set)
        self.DATA_SIZE = len(self._roidb_keys)
        print("Set the data size to be: ", self.DATA_SIZE)


    def __getitem__(self, index):
        video_len = self.key_len[self._roidb_keys[index]]

        if cfg.VIDEO_FRAME==1:
            duration = [0]
        elif cfg.VIDEO_FRAME%2==1:
            duration = range(video_len//2-cfg.VIDEO_FRAME//2-1, video_len//2+cfg.VIDEO_FRAME//2)
        else:
            assert cfg.VIDEO_FRAME%2==0
            duration = range(video_len//2-cfg.VIDEO_FRAME//2, video_len//2+cfg.VIDEO_FRAME//2)
        assert len(duration) == cfg.VIDEO_FRAME
        

        blobs_list = []
        for i in duration:
            single_db = self._roidb[self._roidb_keys[index]+"|{}".format(i)]
            blobs, valid = get_minibatch(single_db)
            
            for key in blobs:
                if (key != 'roidb') and (key != 'dataset_name'):
                    blobs[key] = blobs[key].squeeze(axis=0)

            blobs['roidb'] = blob_utils.serialize(blobs['roidb'])
            blobs_list.append(blobs)

        return blobs_list


    def __len__(self):
        return self.DATA_SIZE



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """

    list_of_blobs = sum(list_of_blobs, [])
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]

    for i in range(0, len(list_of_blobs), cfg.VIDEO_FRAME * cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.VIDEO_FRAME * cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.VIDEO_FRAME * cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])
    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list
