#!/bin/bash

out_dir=/path/to/out_dir
mkdir -p ${out_dir}/{transcription,wav,alignment}

# Extract transcription and save it to a file
metadata_file=/path/to/LJSpeech/LJSpeech-1.1/metadata.csv
while IFS="|" read -r bname tmp transcription
do
    echo ${bname}
    echo "$transcription" > "${out_dir}/transcription/${bname}.txt"
done < ${metadata_file}

# (Optional) Copy wav files
wav_dir=/path/to/LJSpeech/LJSpeech-1.1/wavs
# alignment file is extracted by montreal forced aligner
# https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
alignment_dir=/path/to/alignment

cd ${out_dir}/wav
for fpath in ${wav_dir}/*.wav; do
    bname=$(basename ${fpath} .wav)
    echo ${bname}
    ln -s ${fpath} ${bname}.wav
done
cd -

cd ${out_dir}/alignment
for fpath in ${alignment_dir}/*.TextGrid; do
    bname=$(basename ${fpath} .TextGrid)
    echo ${bname}
    ln -s ${fpath} ${bname}.TextGrid
done
cd -

# Preprocess : Extract CF0, VUV and duration from alignment files
bin_dir=../../../../src/x_vits/bin
HYDRA_FULL_ERROR=1 uv run ${bin_dir}/preprocess.py \
    path=ljspeech \
    mel=ljspeech \
    preprocess.type=LJSPEECH \
    path.data_root=${out_dir}
