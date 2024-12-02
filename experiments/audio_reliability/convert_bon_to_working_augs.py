import argparse
from pathlib import Path

import pandas as pd

from almj.utils.shotgun_utils import process_single_shotgun


def main(args):
    augmentations_path = Path(args.augmentations_path)
    input_file_path = Path(args.input_file_path)
    df = pd.read_json(input_file_path, lines=True)
    df_augs = process_single_shotgun(
        augmentations_path, "bon-replicate", args.model, df, args.n_requests, args.file_name
    )
    df_augs.to_json(augmentations_path, lines=True, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process augmentations and save to JSON")
    parser.add_argument("--augmentations_path", type=str, required=True, help="Path to the augmentations file")
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input file")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--n_requests", type=int, required=True, help="Number of requests")
    parser.add_argument("--file_name", type=str, required=True, help="File name for the output")

    args = parser.parse_args()
    main(args)
