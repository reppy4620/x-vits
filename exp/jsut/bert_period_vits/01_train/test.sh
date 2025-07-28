#!/bin/bash

bin_dir=../../../../src/x_vits/bin

model_name=bert_period_vits

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 uv run ${bin_dir}/test.py \
    generator=${model_name} \
    lit_module=${model_name} \
    generator.text_encoder.context_channels=256
