#!/bin/bash

bin_dir=../../../../src/x_vits/bin

model_name=bert_period_vits

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false uv run ${bin_dir}/synthesize.py \
    generator=${model_name} \
    lit_module=${model_name} \
    generator.text_encoder.context_channels=256

