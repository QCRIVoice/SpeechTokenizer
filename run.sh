#!/bin/sh


stage=1
layer=21


if [ ${stage} == 1 ]; then
    python3 examples/dump_feature.py \
    --model_type hubert \
    --tsv_path /alt/qvoice/Speechtokenizer/SpeechTokenizer/data/train/train.tsv \
    --ckpt_path /alt/qvoice/RepCodec/checkpoint/hubert_large_ll60k.pt \
    --layer ${layer} \
    --feat_dir /alt/qvoice/Speechtokenizer/SpeechTokenizer/out_features/train/
fi

if [ ${stage} == 2 ]; then
    python train.py \
        -c /alt/qvoice/Speechtokenizer/SpeechTokenizer/training_config/config_new.yaml \
        --tag Hubert_myst_21 \
        --exp_root exp 
fi

if [ ${stage} == 3 ]; then
    repcodec /alt/qvoice/RepCodec/out_features/train \
    --model /alt/qvoice/RepCodec/exp/Hubert_myst_21/checkpoint-200000steps.pkl \
    --tsv_path /alt/qvoice/Speechtokenizer/SpeechTokenizer/data/train/train.tsv \
    --model_config_path /alt/qvoice/RepCodec/exp/Hubert_myst_21/config.yml \
    --out_dir out_discrete_features/2l_checkpoint/train/
fi
