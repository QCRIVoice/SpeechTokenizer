# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import torch.nn as nn


class TimeReconstructLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_metric = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss_metric(pred, target)
