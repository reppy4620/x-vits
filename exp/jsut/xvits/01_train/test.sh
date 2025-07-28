#!/bin/bash

bin_dir=../../../../src/x_vits/bin

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 uv run ${bin_dir}/test.py
