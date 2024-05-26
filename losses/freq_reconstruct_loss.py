# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import torch.nn as nn
import torchaudio

class FreqReconstructLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, target):
        loss = 0.0
        scales = range(5,12)
        for i in scales:
            # Compute mel-spectrograms
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=2**i,
                hop_length=2**(i//4),
                n_mels=32,
                normalized=True
            )

            S_pred = mel_spec(pred.cpu())
            S_target = mel_spec(target.cpu())

            # Compute L1 and L2 losses
            L1_loss = self.l1_loss(S_pred,S_target)
            L2_loss = self.l2_loss(S_pred,S_target)
            
            loss += L1_loss + L2_loss
        return(loss)
