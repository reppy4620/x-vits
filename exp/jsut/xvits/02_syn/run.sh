#!/bin/bash

bin_dir=../../../../src/x_vits/bin

HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false uv run ${bin_dir}/synthesize.py
