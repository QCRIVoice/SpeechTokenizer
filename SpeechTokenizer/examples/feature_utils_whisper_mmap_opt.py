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
import webrtcvad
import numpy as np
import soundfile as sf

from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")

def get_speech_segments(audio, sample_rate, aggressiveness=2):

    vad = webrtcvad.Vad(aggressiveness)
    
    # Ensure the audio is in 16-bit PCM format
    if audio.dtype != np.int16:
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

    # Generate audio frames
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
    
    # Filter out non-speech frames
    speech_frames = [frame for frame in frames if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate)]
    
    # Concatenate speech frames
    speech_audio = np.concatenate(speech_frames, axis=0)
    
    return speech_audio

def get_trimmed_audio_raw(audio_data, segment_length):
    
    # Calculate the total number of samples
    total_samples = len(audio_data)
    
    trimmed_segments = []

    # Adjust segment length if the total samples are less than the desired segment length
    if total_samples <= segment_length:
        # segment_length = total_samples
        # trimmed_segments.append(audio_data)
        return None
    else:
        for i in range(0, total_samples - segment_length+1, segment_length + 1):
            start_sample = i
            if start_sample + segment_length <= total_samples:
                trimmed_segments.append(audio_data[start_sample:start_sample+segment_length])
            # else:
            #     trimmed_segments.append(audio_data[start_sample:total_samples])
    return trimmed_segments

def get_trimmed_audio_teacher(audio_data, segment_length):
    
    # Calculate the total number of samples
    total_samples = audio_data.size()[1]
    
    trimmed_segments = []

    # Adjust segment length if the total samples are less than the desired segment length
    if total_samples <= segment_length:
        # segment_length = total_samples
        # trimmed_segments.append(audio_data.detach().cpu().numpy())
        return None
    else:
        for i in range(0, total_samples - segment_length+1, segment_length + 1):
            start_sample = i
            if start_sample + segment_length <= total_samples:
                trimmed_segments.append(audio_data[:,start_sample:start_sample+segment_length,:].detach().cpu().numpy())
            # else:
            #     trimmed_segments.append(audio_data[:,start_sample:total_samples,:].detach().cpu().numpy())
    return trimmed_segments

def get_path_iterator(tsv):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)


def dump_feature(reader, generator, num, feat_dir, language):
    iterator = generator()

    chunk_size=2000
    x_path = f"{feat_dir}/raw_{language}.npy"
    x_teacher_path = f"{feat_dir}/teacher_{language}.npy"
    x_len = f"{feat_dir}/raw_{language}.len"
    x_teacher_len = f"{feat_dir}/teacher_{language}.len"
    
    # os.makedirs(feat_dir, exist_ok=True)
    # if os.path.exists(x_path):
    #     os.remove(x_path)
    # if os.path.exists(x_teacher_path):
    #     os.remove(x_teacher_path)   
    # if os.path.exists(x_len):
    #     os.remove(x_len)
    # if os.path.exists(x_teacher_len):
    #     os.remove(x_teacher_len)  

    raw_feat = NpyAppendArray(x_path)
    teacher_feat = NpyAppendArray(x_teacher_path)
    
    with open(x_len, "w") as raw_len, open(x_teacher_len, "w") as teacher_len:
        x_list = []
        x_teacher_list = []
        
        for path, nsample in tqdm.tqdm(iterator, total=num):
            audio, sr = sf.read(path)
            max_len_teacher = 80
            max_len_raw = 64000 
            max_len_1500 = 1200000  # whisper can handle max 1500 embedding perembedding= 50msec
            
            raw_list = []
            teacher_list = []
            if len(audio) > max_len_1500:
                whisper_input_list = get_trimmed_audio_raw(audio, max_len_1500)
                for i in range(len(whisper_input_list)):
                    feat = reader.get_feats(whisper_input_list[i])
                    (raw_list.extend(get_trimmed_audio_raw(whisper_input_list[i], max_len_raw)) if get_trimmed_audio_raw(whisper_input_list[i], max_len_raw) is not None else None )
                    (teacher_list.extend(get_trimmed_audio_teacher(feat, max_len_teacher)) if get_trimmed_audio_teacher(feat, max_len_teacher)  is not None else None)
            else:     
                feat = reader.get_feats(audio)
                (raw_list.extend(get_trimmed_audio_raw(audio, max_len_raw)) if get_trimmed_audio_raw(audio, max_len_raw) is not None else None)
                (teacher_list.extend(get_trimmed_audio_teacher(feat, max_len_teacher)) if get_trimmed_audio_teacher(feat, max_len_teacher) is not None else None)
            
            if len(raw_list) < len(teacher_list):
                teacher_list = teacher_list[:len(raw_list)]
            elif len(teacher_list) < len(raw_list):
                raw_list = raw_list[:len(teacher_list)]
            
            x_list.extend(raw_list)
            x_teacher_list.extend(teacher_list)
            
            if len(x_list) >= chunk_size:
                write_data_chunk(raw_feat, teacher_feat, raw_len, teacher_len, x_list, x_teacher_list)
                x_list.clear()
                x_teacher_list.clear()
        
        # Write any remaining data
        if x_list:
            write_data_chunk(raw_feat, teacher_feat, raw_len, teacher_len, x_list, x_teacher_list)
        
        logger.info(f"Processed {num} samples for language {language}")
        

def write_data_chunk(raw_feat, teacher_feat, raw_len, teacher_len, x_list, x_teacher_list):
    x_teacher_list = [np.ascontiguousarray(np.squeeze(feat)) for feat in x_teacher_list]
    for i in range(len(x_list)):
        raw_feat.append(x_list[i])
        teacher_feat.append(x_teacher_list[i])
        raw_len.write(f"{len(x_list[i])}\n")
        teacher_len.write(f"{len(x_teacher_list[i])}\n") 



