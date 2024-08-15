# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on fairseq (https://github.com/facebookresearch/fairseq)

# ref: https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/simple_kmeans/feature_utils.py

import logging
import os
import sys

import tqdm
from npy_append_array import NpyAppendArray
import librosa
import pickle
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)


def dump_feature(reader, generator, num, nshard, rank, feat_dir):
    iterator = generator()

    x_path = f"{feat_dir}/raw_audio.pickle"
    x_teacher_path = f"{feat_dir}/teacher_embeddings.pickle"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(x_path):
        os.remove(x_path)
    
    if os.path.exists(x_teacher_path):
        os.remove(x_teacher_path)    

    x_list = []
    x_teacher_list = []
    # x = NpyAppendArray(x_path)
    # x_teacher = NpyAppendArray(x_teacher_path)
    for path, nsample in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path, nsample)
        print(feat.shape)
        x_teacher_list.append(feat.cpu().numpy())
        raw_audio,_ = librosa.load(path)
        x_list.append(raw_audio)
    
    print(f"length of raw_audio data: {len(x_list)}")
    with open(x_path, "wb") as output_file:
        pickle.dump(x_list, output_file)
    with open(x_teacher_path, "wb") as output_file_2:
        pickle.dump(x_teacher_list, output_file_2)
    # x.close()
    # x_teacher.close()
        
    logger.info("finished successfully")


