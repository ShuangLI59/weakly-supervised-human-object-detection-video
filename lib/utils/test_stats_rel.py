#!/usr/bin/env python2

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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import datetime
import numpy as np

from core.config import cfg
from utils.logging_rel import log_stats
from utils.logging_rel import SmoothedValue
from utils.timer import Timer
import utils.net as nu
import pdb

class TestStats(object):
    """Track vital training statistics."""

    def __init__(self, misc_args, log_period=20, tensorboard_logger=None, is_training=True):
        # Output logging period in SGD iterations
        self.misc_args = misc_args
        self.LOG_PERIOD = log_period
        self.tblogger = tensorboard_logger
        self.tb_ignored_keys = ['val_iter', 'val_eta', 'val_time', 'val_lr', 'val_backbone_lr']
        self.iter_timer = Timer()
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 20
        def create_smoothed_value():
            return SmoothedValue(self.WIN_SZ)
        self.smoothed_losses = defaultdict(create_smoothed_value)
        self.smoothed_metrics = defaultdict(create_smoothed_value)
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        
        self.inner_total_loss = []
        self.inner_losses = defaultdict(list)
        self.inner_metrics = defaultdict(list)

        self.is_training = is_training

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self, model_out, inner_iter=None):
        model_out['val_losses'] = {}
        model_out['val_metrics'] = {}
        for k, loss in model_out['losses'].items():
            model_out['val_losses']['val_'+k] = loss
        for k, metric in model_out['metrics'].items():
            model_out['val_metrics']['val_'+k] = metric

        del model_out['losses']
        del model_out['metrics']

        # """Update tracked iteration statistics."""
        # if inner_iter is not None and self.misc_args.iter_size > 1:
        #     # For the case of using args.iter_size > 1
        #     return self._UpdateIterStats_inner(model_out, inner_iter)

        # Following code is saved for compatability of train_net.py and iter_size==1
        total_loss = 0
        
        for k, loss in model_out['val_losses'].items():
            assert loss.shape[0] == cfg.NUM_GPUS
            loss = loss.mean(dim=0, keepdim=True)
            total_loss += loss
            loss_data = loss.data[0]
            model_out['val_losses'][k] = loss
            self.smoothed_losses[k].AddValue(loss_data)

        model_out['val_total_loss'] = total_loss  # Add the total loss for back propagation
        self.smoothed_total_loss.AddValue(total_loss.data[0])
        
        for k, metric in model_out['val_metrics'].items():
            metric = metric.mean(dim=0, keepdim=True)
            self.smoothed_metrics[k].AddValue(metric.data[0])


    def _mean_and_reset_inner_list(self, attr_name, key=None):
        """Take the mean and reset list empty"""
        if key:
            mean_val = sum(getattr(self, attr_name)[key]) / self.misc_args.iter_size
            getattr(self, attr_name)[key] = []
        else:
            mean_val = sum(getattr(self, attr_name)) / self.misc_args.iter_size
            setattr(self, attr_name, [])
        return mean_val

    def LogIterStats(self, cur_iter):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0):
            stats = self.GetStats(cur_iter)
            log_stats(stats, self.misc_args, self.is_training)
            
    def tb_log_stats(self, stats, cur_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self.tb_log_stats(v, cur_iter)
                else:
                    self.tblogger.add_scalar(k, v, cur_iter)

    def GetStats(self, cur_iter):
        eta_seconds = self.iter_timer.average_time * (cfg.SOLVER.MAX_ITER - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        stats = OrderedDict(
            val_iter=cur_iter + 1,  # 1-indexed
            val_time=self.iter_timer.average_time,
            val_eta=eta,
            val_loss=self.smoothed_total_loss.GetMedianValue()
        )
        stats['val_metrics'] = OrderedDict()
        for k in sorted(self.smoothed_metrics):
            stats['val_metrics'][k] = self.smoothed_metrics[k].GetMedianValue()

        head_losses = []
        for k, v in self.smoothed_losses.items():
            head_losses.append((k, v.GetMedianValue()))
        stats['val_head_losses'] = OrderedDict(head_losses)

        return stats
