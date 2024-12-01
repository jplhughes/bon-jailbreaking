#!/bin/bash
set -eou pipefail

input_file_path=./data/direct_request.jsonl

alm_model="gemini-1.5-flash-001"
alm_n_samples=1
alm_temperature=1.0

chosen_augmentations="speed pitch speech noise volume music" # remove speech noise and music if you want to replicate faster and not install Kaldi

sigma=0.25
alg_seed=0
asr_threshold=0.01

# note that overall N = num_concurrent_k * n_steps
num_concurrent_k=10 # in paper this is 60
n_steps=8 # in paper this is 120
n_requests=5 # in paper this is 159

# baseline runs
temperature=1.0
n_samples=40 # in paper this is 5000

for alm_model in "gemini-1.5-flash-001"; do

    # run bon jailbreak for each specific id
    for choose_specific_id in $(seq 0 $(($n_requests - 1))); do

        output_dir=./exp/figure1/${alm_model}/${choose_specific_id}

        python -m almj.attacks.audio.run_bon_jailbreaking --input_file_path $input_file_path --output_dir $output_dir --enable_cache False --openai_num_threads 120 --gemini_num_threads 120 --alm_model $alm_model --seed $alg_seed --alm_temperature $alm_temperature --alm_n_samples $alm_n_samples --chosen_augmentations $chosen_augmentations --sigma $sigma --choose_specific_id $choose_specific_id  --asr_threshold $asr_threshold --num_concurrent_k $num_concurrent_k --n_steps $n_steps --run_batch False
    done

    # run repeated sample baseline
    output_dir=./exp/figure1/baselines/audio-only

    python3 -m almj.attacks.run_baseline \
        --input_file $input_file_path \
        --output_dir $output_dir \
        --model_id $alm_model \
        --text_input "none" \
        --audio_input "audio_file" \
        --temperature $temperature \
        --n_samples $n_samples \
        --enable_cache False \
        --stop_idx $n_requests

    # run text-only baseline
    output_dir=./exp/figure1/baselines/text-only

    python3 -m almj.attacks.run_baseline \
        --input_file $input_file_path \
        --output_dir $output_dir \
        --model_id $alm_model \
        --text_input "rewrite" \
        --audio_input "none" \
        --temperature $temperature \
        --n_samples $n_samples \
        --stop_idx $n_requests \
        --enable_cache False
done
