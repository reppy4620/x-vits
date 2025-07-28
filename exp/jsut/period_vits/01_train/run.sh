#!/bin/bash

bin_dir=../../../../src/x_vits/bin

model_name=period_vits

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false uv run ${bin_dir}/train.py \
    generator=${model_name} \
    lit_module=period_vits
