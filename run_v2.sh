#!/bin/sh


stage=2
layer=32


if [ ${stage} == 1 ]; then
    python3 examples/dump_feature.py \
    --model_type hubert \
    --tsv_path ./data/train/train.tsv \
    --ckpt_path ./checkpoint/hubert_large_ll60k.pt \
    --layer ${layer} \
    --feat_dir ./out_features/train/
fi

if [ ${stage} == 2 ]; then
    OMP_NUM_THREADS=12 python -m torch.distributed.run --nproc_per_node=4 train_mmap_v2.py \
        -c ./train_config/config_2l.yaml \
        --tag Whisper_32layer_ar+en_ddp \
        --exp_root exp 
fi

if [ ${stage} == 3 ]; then
    repcodec /alt/qvoice/RepCodec/out_features/train \
    --model /alt/qvoice/RepCodec/exp/Hubert_myst_21/checkpoint-200000steps.pkl \
    --tsv_path /alt/qvoice/Speechtokenizer/SpeechTokenizer/data/train/train.tsv \
    --model_config_path /alt/qvoice/RepCodec/exp/Hubert_myst_21/config.yml \
    --out_dir out_discrete_features/2l_checkpoint/train/
fi
