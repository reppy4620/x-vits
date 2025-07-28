#!/bin/bash

bin_dir=../../../../src/x_vits/bin

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false uv run ${bin_dir}/train.py \
    path=ljspeech \
    generator.text_encoder.num_vocab=151 \
    dataset=single_en \
    mel=ljspeech
