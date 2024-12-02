#!/bin/bash
set -eou pipefail

input_file_path=./data/direct_request.jsonl
model="gemini-1.5-flash-001"

output_dir="exp/figure1/reliability/$model"
augmentations_path="$output_dir/working_augs_with_file.jsonl"
file_name=direct_request_search_steps.jsonl

alm_temperature=1.0 # must be this since that is the temperature used to generate the working_augs
alm_n_samples=5 # this was 200 in the paper
n_requests=5 # this is 159 in the paper

mkdir -p $output_dir

python3 -m experiments.audio_reliability.convert_bon_to_working_augs --augmentations_path $augmentations_path --input_file_path $input_file_path --n_requests $n_requests --file_name $file_name --model $model

python3 -m experiments.audio_reliability.run_specific_vector_augmentation --input_file_path $input_file_path --output_dir $output_dir --augmentations_path $augmentations_path --alm_temperature $alm_temperature --only_run_on_broken_example True --alm_n_samples $alm_n_samples --use_same_file True
