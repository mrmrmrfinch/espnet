#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# Yongyi Zang, November 2022, Modified from https://github.com/espnet/espnet/compare/master...2022fall_new_task_tutorial.

# set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_feature_extraction=false # Skip feature extraction stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
datadir=data         # Directory to put data after stage 1, data prep.
dumpdir=dump         # Directory to dump features.
inference_nj=4      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Feature extraction related
fs=16000      # Sampling rate
downmix="mix" # Downmixing type (mix/first/concat)
feature_type="raw" # Feature type (raw, mel or linear)
n_mels=80     # Number of mel basis
n_fft=512     # FFT size
n_shift=256   # Shift size
feature_extract_dir=${dumpdir}/${feature_type} # Directory to dump features
audio_length=5 # Audio length, in seconds
trunc_mode="center" # Truncation mode for feature extraction (left, center, right, random)
pad_mode="tile" # Padding mode for feature extraction (zero, tile)

# Training related
training_config="conf/temp.json" # Training config

# TODO: add help message

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
# run_args=$(pyscripts/utils/print_args.py $0 "$@")
# . utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Anti-spoofing verification with ASVspoof2019LA:"
    log "Help message hasn't been implemented yet."
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# ========================== Main stages start from here. ==========================
if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation."
        # local/data.sh ${local_data_opts} 
        # At this moment, only ASVspoof2019LA is supported; therefore, no need for data opts
        local/data.sh
    fi
else
    log "Skip the data preparation stages"
fi
# ========================== Data preparation is done here. ==========================
# ========================== Feature extraction starts from here. ==========================
if ! "${skip_feature_extraction}"; then
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Stage 2: Feature extraction."
        log "Extracting ${feature_type} features for train, dev and eval."

        if [ ! -d ${feature_extract_dir} ]; then
            mkdir -p ${feature_extract_dir}
        fi

        python3 local/extract_features.py \
            --feature_type ${feature_type} \
            --fs ${fs} \
            --downmix ${downmix} \
            --n_mels ${n_mels} \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --data_dir ${datadir} \
            --output_dir ${feature_extract_dir} \
            --audio_length ${audio_length} \
            --trunc_mode ${trunc_mode} \
            --pad_mode ${pad_mode}

        log "Finished feature extraction."
    fi
else
    log "Skip the feature extraction stages"
fi
# ========================== Feature extraction is done here. ==========================
# ========================== Training starts from here. ==========================
if ! "${skip_train}"; then
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Training."
        log "Training ASV model."
        python3 local/train.py \
            --data_dir ${datadir} \
            --feature_dir ${feature_extract_dir} \
            --exp_dir ${expdir} \
            --config ${training_config} \
            --ngpu ${ngpu}
        log "Finished training."
    fi
fi
# ========================== Training is done here. ==========================
# ========================== Decoding and evaluation starts from here. ==========================