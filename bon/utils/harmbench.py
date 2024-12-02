import json
import pathlib

import pandas as pd


def _get_harmbench_behaviors():
    df_val_behaviors = pd.read_csv(
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "e2d308a711b77f5e6096ef7991e6b295a14f79d8"
        "/data/behavior_datasets/harmbench_behaviors_text_val.csv"
    )
    df_val_behaviors["split"] = "val"

    df_test_behaviors = pd.read_csv(
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "e2d308a711b77f5e6096ef7991e6b295a14f79d8"
        "/data/behavior_datasets/harmbench_behaviors_text_test.csv"
    )
    df_test_behaviors["split"] = "test"

    df_behaviors = pd.concat([df_val_behaviors, df_test_behaviors]).reset_index(drop=True)
    return df_behaviors


def _get_harmbench_static_attacks_for_api_models(
    harmbench_data_dir: pathlib.Path = pathlib.Path("/mnt/jailbreak-defense/exp/harmbench_data_from_authors"),
    verbose: bool = False,
):
    """Gets pre-generated Harmbench attacks for api models."""
    attack_to_test_cases = {
        "DirectRequest": "DirectRequest/default/test_cases/test_cases.json",
        "EnsembleGCG": "EnsembleGCG/*/test_cases/test_cases.json",
        "HumanJailbreaks": "HumanJailbreaks/*/test_cases/test_cases.json",
        "PAIR": "PAIR/*/test_cases/test_cases.json",
        "PAP": "PAP/*/test_cases/test_cases.json",
        "TAP": "TAP/*/test_cases/test_cases.json",
        "ZeroShot": "ZeroShot/mixtral_attacker_llm/test_cases/test_cases.json",
    }

    dfs = []
    for attack, pattern in attack_to_test_cases.items():
        if verbose:
            print(attack, pattern)
        test_case_paths = harmbench_data_dir.glob(pattern)
        for file in test_case_paths:
            with open(file) as f:
                data = json.load(f)
            test_cases = []
            for behavior, test_cases_behavior in data.items():
                for test_case in test_cases_behavior:
                    test_cases.append({"behavior": behavior, "rewrite": test_case})
            df_ = pd.DataFrame(test_cases)
            df_["model"] = file.parent.parent.name
            df_["attack"] = attack
            dfs.append(df_)

    return pd.concat(dfs).reset_index(drop=True).rename(columns={"behavior": "behavior_id"})


def get_harmbench_data(verbose: bool = False):
    df_behaviors = _get_harmbench_behaviors()
    df_all = _get_harmbench_static_attacks_for_api_models(verbose=verbose)

    bid_to_base_request = dict(df_behaviors[["BehaviorID", "Behavior"]].values)
    bid_to_functional_cat = dict(df_behaviors[["BehaviorID", "FunctionalCategory"]].values)
    bid_to_semantic_cat = dict(df_behaviors[["BehaviorID", "SemanticCategory"]].values)
    bid_to_split = dict(df_behaviors[["BehaviorID", "split"]].values)

    df_all["behavior_str"] = df_all["behavior_id"].map(bid_to_base_request)
    df_all["functional_category"] = df_all["behavior_id"].map(bid_to_functional_cat)
    df_all["semantic_category"] = df_all["behavior_id"].map(bid_to_semantic_cat)
    df_all["split"] = df_all["behavior_id"].map(bid_to_split)

    # Drop duplicates
    df_all = df_all.drop_duplicates(subset=["rewrite"])

    return df_all
