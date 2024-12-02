# BoN Jailbreaking
Contains code useful for jailbreaking LLMs.

## Repository setup

### Environment

To set up the development environment for this project,
follow the steps below:

1. First install `micromamba` if you haven't already.
You can follow the installation instructions here:
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html. The Homebrew / automatic installation method is recommended.
Or just run:
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
2. Next, you need to run the following from the root of the repository:
    ```bash
    micromamba env create -n bon python=3.11.7
    micromamba activate bon
    pip install -r requirements.txt
    pip install -e .
    (git clone git@github.com:facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop)
    ```

3. Install Kaldi if on a new machine: ./scripts/install_kaldi.sh

### Secrets

You should create a file called `SECRETS` at the root of the repository
with the following contents:
```
# Required
OPENAI_API_KEY=<your-key>
OPENAI_ORG=<openai-org-id>
GOOGLE_API_KEY=<your-key>
GOOGLE_PROJECT_ID=<GCP-project-name>
GOOGLE_PROJECT_REGION=<GCP-project-region>
HF_API_KEY=<your-key> # required for Llama3 and Circuit Breaking
GRAYSWAN_API_KEY=<your-key> # required for Cygnet
ELEVENLABS_API_KEY=<your-key> # required for ElevenLabs TTS (used in PrePAIR)
```

## Replicate experiments

To replicate the experiments in the paper, run the scripts in the `experiments` directory. For example, to replicate Figure 1, run the following:

```bash
./experiments/1_run_text_bon.sh
```

## Human data

We release our dataset of human verbalized jailbreaks from Harmbench. This inlcudes 308 PAIR, 307 TAP and 159 direct requests. These are in git LFS, so to download please run:
```bash
git lfs install
git lfs pull
```

Then unzip the data:
```bash
unzip data/human_data.zip
```

Use pandas to open the jsonl file to see the jailbreak (located in "rewrite" column), audio files and other metadata such as the attack type.
```python
import pandas as pd

df = pd.read_json('human_data/verbalized_requests.jsonl', lines=True)
```

