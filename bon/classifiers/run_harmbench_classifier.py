import argparse
import asyncio
import pathlib

import bon.classifiers.run_classifier


async def main(input_dir: pathlib.Path):
    output_dir = input_dir / "classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.jsonl"):
        print(input_file)
        file_name = f"{input_file.stem}.jsonl"
        cfg = bon.classifiers.run_classifier.ExperimentConfig(
            output_dir=output_dir,
            response_input_file=input_file,
            file_name=file_name,
            model_outputs_tag=None,
            model_output_tag="completion",
            classifier_fields=dict(
                behavior="behavior_str",
                assistant_response="completion",
            ),
            classifier_template="harmbench/harmbench-gpt-4.jinja",
            classifier_models=("gpt-4o",),
            max_tokens=5,
            temperature=0,
            n_samples=1,
            get_logprobs=False,
            anthropic_num_threads=50,
            openai_fraction_rate_limit=0.9,
            seed=0,
            print_prompt_and_response=False,
        )
        cfg.setup_experiment(log_file_prefix="run-harmbench-classifier")
        await bon.classifiers.run_classifier.main(cfg=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classifier on input directory")
    parser.add_argument("--input_dir", type=pathlib.Path, help="Path to the input directory")
    args = parser.parse_args()
    asyncio.run(main(args.input_dir))
