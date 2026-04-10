# Check Signature Verifier

Lightweight proof of concept for offline signature verification on checks and similar documents.

This repo demonstrates a simple end-to-end workflow:
- preprocess signature images into a normalized `224x224` format
- train a Siamese neural network on genuine vs different-writer samples
- compare two signatures and return a similarity score plus verdict
- present the result through a local Gradio demo

This project is a **POC**, not a production fraud-decisioning system.

## Repo Contents

- `data_preprocessing.py`: preprocessing pipeline for signature extraction, cleanup, cropping, and resize
- `model.py`: Siamese model definition using a `ResNet-18` backbone
- `train.py`: training script for pair generation, contrastive loss, and checkpoint save
- `inference.py`: loads a checkpoint and compares two signature images
- `demo_app.py`: Gradio UI for local demo use
- `case_study.md`: short presentation-friendly summary of the project and results

## Requirements

- Python `3.11+`
- Local install of the packages in `requirements.txt`

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Expected Data Layout

Training expects class folders under `data/samples/`.

Example:

```text
data/
  samples/
    genuine/
      genuine_1.png
      genuine_2.png
      ...
    other/
      other_1.png
      other_2.png
      ...
```

The current POC uses:
- `genuine/` for the reference writer
- `other/` for a different writer

## Train The Model

Run a first training pass with:

```bash
python train.py --epochs 10 --batch-size 8
```

This will:
- preprocess the sample images
- build positive and negative training pairs
- train the Siamese model with contrastive loss
- save `checkpoint.pt` in the repo root

## Run Inference From The CLI

Compare two signatures directly:

```bash
python inference.py "data\samples\genuine\genuine_1.png" "data\samples\genuine\genuine_2.png"
```

Example output:

```text
Distance: 0.0853
Similarity: 95.7%
Verdict: match
```

Compare a genuine sample against a different writer:

```bash
python inference.py "data\samples\genuine\genuine_1.png" "data\samples\other\other_1.png"
```

## Launch The Demo

Start the Gradio app with:

```bash
python demo_app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://127.0.0.1:7860
```

The demo allows you to:
- upload a reference signature
- upload a questioned signature
- view the processed crops used by the model
- see similarity, distance, and a `match` / `review` / `mismatch` verdict

## Current Status

The current repo supports:
- preprocessing
- model training
- checkpoint-based inference
- a working local demo

The current repo does **not** yet include:
- large-scale training data
- FAR / FRR / EER evaluation scripts
- robust arbitrary check-image localization
- production deployment, governance, or calibrated operating thresholds

## Positioning

This repo should be presented as:
- a workflow demo
- a technical proof of concept
- a starting point for a future pilot

It should **not** be presented as:
- a bank-grade fraud model
- a production-ready signature verification platform
- validated evidence of enterprise error rates

## Suggested Next Steps

- expand the labeled dataset
- test on masked real check images
- add evaluation scripts for threshold tuning and error-rate measurement
- decide whether future scope is assist-only triage or stronger automation
