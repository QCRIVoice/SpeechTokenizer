# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import numpy as np
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger("collate")
torch.multiprocessing.set_start_method('spawn', force=True)

def collate_fn(batch):
    
    x = []
    x_teacher = []
    lid_label = []

    for b in batch:
        if b[0] is not None and b[1] is not None:
            x.append(torch.from_numpy(b[0]))
            x_teacher.append(torch.from_numpy(b[1]))
            lid_label.append(b[2])

    x_batch = pad_sequence(x, batch_first=True, padding_value=0).unsqueeze(1).float()

    x_teacher_batch = pad_sequence(x_teacher, batch_first=True, padding_value=0)
    x_teacher_batch = x_teacher_batch.to(torch.float)  # (B, T, C) -> (B, C, T)
    lid_label_batch = torch.stack(lid_label)
    
    return x_batch,x_teacher_batch,lid_label_batch
