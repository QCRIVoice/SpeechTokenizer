# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import logging
import os
import torch
from typing import List

import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("dataset")


class STDataset(Dataset):
    def __init__(
            self,
            languages: List,
            data_dir: str,
    ):
        self.raw_blocks,self.teacher_blocks = self._load_blocks(data_dir,languages)
        self.raw_offsets,self.teacher_offsets = self._load_offsets(data_dir,languages)
        assert len(self.raw_blocks) == len(self.raw_offsets)
        assert len(self.teacher_blocks) == len(self.teacher_offsets)
        # check len
        for i in range(len(self.raw_blocks)):
            assert self.raw_blocks[i].shape[0] == self.raw_offsets[i][-1]
            assert self.teacher_blocks[i].shape[0] == self.teacher_offsets[i][-1]
            

        self.n_examples = np.cumsum([0] + [offset.shape[0] - 1 for offset in self.raw_offsets])

    def __len__(self):
        return self.n_examples[-1]

    def __getitem__(self, idx):
        # find which block
        block_id = -1
        for n in range(len(self.n_examples) - 1):
            if self.n_examples[n] <= idx < self.n_examples[n + 1]:
                block_id = n
                break
        assert 0 <= block_id < len(self.raw_blocks), f"Failed to find {idx}"
        block_offset = idx - self.n_examples[block_id]

        ##raw_features
        raw_start = self.raw_offsets[block_id][block_offset]
        raw_end = self.raw_offsets[block_id][block_offset + 1]
        raw = self.raw_blocks[block_id][raw_start:raw_end]
        # teacher features
        teacher_start = self.teacher_offsets[block_id][block_offset]
        teacher_end = self.teacher_offsets[block_id][block_offset+1]
        teacher = self.teacher_blocks[block_id][teacher_start:teacher_end]

        lid_label = torch.zeros(len(self.n_examples)-1,dtype=torch.long)
        lid_label[block_id] = 1


        return (raw,teacher,lid_label)

    @staticmethod
    def _load_blocks(feat_dir,languages: str) -> List[np.ndarray]:

        raw_file_names= [f"{feat_dir}/raw_{language}.npy" for language in languages ]
        teacher_file_names = [f"{feat_dir}/teacher_{language}.npy" for language in languages ]
        raw_blocks = [np.load(name,mmap_mode='r') for name in raw_file_names]
        teacher_blocks = [np.load(name,mmap_mode='r') for name in teacher_file_names]
        return raw_blocks,teacher_blocks

    @staticmethod
    def _load_offsets(feat_dir,languages: str):
        def load_lens(file_name: str):
            with open(file_name, mode="r") as fp:
                res = fp.read().strip().split("\n")
            # for easy use. [res[i], res[i+1]) denotes the range for ith element
            res = [0] + [int(r) for r in res]
            logger.info(f"for {file_name}, lens are {res}")
            return np.cumsum(res, dtype=int)

        raw_file_names= [f"{feat_dir}/raw_{language}.len" for language in languages ]
        teacher_file_names = [f"{feat_dir}/teacher_{language}.len" for language in languages ]
        raw_file_lens = []
        teacher_file_lens = []

        for name in raw_file_names:
            raw_file_lens.append(load_lens(name))
        for name in teacher_file_names:
            teacher_file_lens.append(load_lens(name))
        # raw_file_lens = load_lens(raw_file)
        # teacher_file_lens = load_lens(teacher_file)
        return raw_file_lens,teacher_file_lens  
