#!/bin/bash
set -eou pipefail

input_file_path=./data/direct_request.jsonl

# model
models="gemini-1.5-flash-001"
# other models: "gemini-1.5-pro-001", "gpt-4o-realtime-preview-2024-10-01" (DiVA unsupported in this codebase)
temperature=1.0

# audio bon specific
chosen_augmentations="speed pitch speech noise volume music" # remove speech noise and music if you want to replicate faster and not install Kaldi
sigma=0.25

# note that overall N = num_concurrent_k * n_steps
num_concurrent_k=10 # in paper this is 60
n_steps=8 # in paper this is 120
n_requests=5 # in paper this is 159
request_ids="31 69 58 83 4 72 151 14 81 38" # This is a set of easy ids for quick replication, for full set use $(seq 0 158)

# baseline runs
baseline_temperature=1.0
n_samples=40 # in paper this is 5000

# NOTE: when running gemini please run the delete files script in parallel to avoid storage quota issues
# python3 -m bon.apis.inference.gemini.run_delete_gemini_files


for model in $models; do

    # run bon jailbreak for each specific id
    for choose_specific_id in $request_ids; do

        output_dir=./exp/bon/audio/${model}/${choose_specific_id}

        if [ -f $output_dir/done_$n_steps ]; then
            echo "Skipping $output_dir because it is already done"
            continue
        fi

        python -m bon.attacks.run_audio_bon \
            --input_file_path $input_file_path \
            --output_dir $output_dir \
            --enable_cache False \
            --alm_model $model \
            --alm_temperature $temperature \
            --chosen_augmentations $chosen_augmentations \
            --sigma $sigma \
            --choose_specific_id $choose_specific_id \
            --num_concurrent_k $num_concurrent_k \
            --n_steps $n_steps
    done

    # run repeated sample baseline
    output_dir=./exp/baselines/audio

    python3 -m bon.attacks.run_baseline \
        --dataset_path $input_file_path \
        --output_dir $output_dir \
        --model $model \
        --modality audio \
        --temperature $baseline_temperature \
        --n_samples $n_samples \
        --enable_cache False \
        --request_ids "$request_ids"
done
