#!/bin/bash
set -eou pipefail

n_steps=1
batch_size=4
limit_batches=1 # in the paper we don't limit batches
target_model="gemini-1.5-flash-001"
semaphore_limit=5
verbose=False
attack_type="audio"
request_type="audio"

input_file=./data/direct_request.jsonl
output_dir="./exp/prepair"

init_attack_paths="pair/e2e/init/first.jinja"

for init_attack_path in "${init_attack_paths[@]}"; do
    if [ "$init_attack_path" == "empty" ]; then
        init_attack_stem="empty"
    else
        init_attack_stem=$(basename "${init_attack_path%.*}")
    fi

    file_name="init-${init_attack_stem}.jsonl"

    python -m almj.attacks.run_prepair \
        --input_file $input_file \
        --direct_requests_path $input_file \
        --verbose $verbose \
        --output_dir "$output_dir" \
        --file_name "$file_name" \
        --n_steps $n_steps \
        --batch_size $batch_size \
        --init_attack_path "$init_attack_path" \
        --attack_type "$attack_type" \
        --request_type "$request_type" \
        --target_model "$target_model" \
        --semaphore_limit $semaphore_limit \
        --limit_batches $limit_batches
done
