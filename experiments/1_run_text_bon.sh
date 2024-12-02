#!/bin/bash
set -eou pipefail

input_file_path=./data/direct_request.jsonl

models="gpt-4o-mini"
# other models: "gpt-4o", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "gemini-1.5-pro-001", "gemini-1.5-flash-001", "GraySwanAI/Llama-3-8B-Instruct-RR", "meta-llama/Meta-Llama-3-8B-Instruct", "cygnet"
# need to set up HuggingFace inference endpoint for GraySwan and Meta models (and add URLs to HUGGINGFACE_MODELS in almj/apis/inference/huggingface.py), also put HF_API_KEY in SECRETS
# for cygnet, you need a GRAYSWAN_API_TOKEN in SECRETS

temperature=1.0

# note that overall N = num_concurrent_k * n_steps
num_concurrent_k=10 # in paper this is 60
n_steps=8 # in paper this is 120
n_requests=5 # in paper this is 159
request_ids="31 69 58 83 4 72 151 14 81 38" # This is a set of easy ids for quick replication, for full set use $(seq 0 158)

# turn down to avoid rate limiting
gemini_num_threads=10

# baseline runs
baseline_temperature=1.0
n_samples=40 # in paper this is 5000

for model in $models; do

    model_str=${model//\//-}

    # run bon jailbreak for each specific id
    for choose_specific_id in $request_ids; do

        output_dir=./exp/bon/text/${model_str}/${choose_specific_id}

        if [ -f $output_dir/done_$n_steps ]; then
            echo "Skipping $output_dir because it is already done"
            continue
        fi

        python -m almj.attacks.run_text_bon \
            --input_file_path $input_file_path \
            --output_dir $output_dir \
            --enable_cache False \
            --gemini_num_threads $gemini_num_threads \
            --lm_model $model \
            --lm_temperature $temperature \
            --choose_specific_id $choose_specific_id \
            --num_concurrent_k $num_concurrent_k \
            --n_steps $n_steps
    done

    # run repeated sample baseline
    output_dir=./exp/baselines/text

    python3 -m almj.attacks.run_baseline \
        --dataset_path $input_file_path \
        --output_dir $output_dir \
        --model $model \
        --modality text \
        --temperature $baseline_temperature \
        --n_samples $n_samples \
        --enable_cache False \
        --request_ids "$request_ids"
done
