# Interview Simulator — Llama 3.1-8B Instruct MVP

This repository contains a minimal MVP to interact with the Llama-3.1-8B-Instruct model hosted on Hugging Face. The goal is a simple interview chat you can run locally.

Quick setup (PowerShell on Windows):

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

1. Set your Hugging Face API key (in the same PowerShell session):

```powershell
$env:HF_API_KEY = "your_hf_api_key_here"
```

1. Run the CLI chat (script to be added):

```powershell
python mvp/run_chat.py
```

Notes

- The project will call the Hugging Face inference API for `meta-llama/Llama-3.1-8B-Instruct`.
- Keep your API key secret. For production, use a secrets manager or environment variables persisted securely.

Run the model locally (CUDA 12.1)

If you want to run `meta-llama/Llama-3.1-8B-Instruct` locally you need a machine with sufficient GPU memory (multiple high-memory GPUs or an adapter like 48+ GB of VRAM). Install PyTorch built for CUDA 12.1. Example commands (PowerShell):

```powershell
# Example: install torch for CUDA 12.1 (check the official PyTorch site for the correct wheel command for Windows + CUDA 12.1)
# This is an example placeholder; replace with the exact command from https://pytorch.org/get-started/locally/
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Interview Simulator — Mistral 7B Instruct (local)

This project runs a minimal interview CLI using a local LLM. The default local model is
`mistralai/Mistral-7B-Instruct-v0.3`. You can either download weights locally and set
`LOCAL_MODEL_PATH`, or run the CLI with `--mock` to test without weights.

Quick setup (PowerShell)

1) Create and activate a venv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Install a torch wheel that matches your CUDA (if you have a GPU). Example for CUDA 12.1 — replace with the correct command from PyTorch docs:

```powershell
# Example only — pick the exact wheel for your platform
pip install --index-url https://download.pytorch.org/whl/cu121 torch --upgrade
```

3) (Optional) Download the model locally

If you want to run the real model locally, download it to a folder and set `LOCAL_MODEL_PATH`.
If the HF repo is gated, request/accept access on the model page before attempting the download.

```powershell
# Authenticate with HF if needed
pip install --upgrade huggingface-hub
huggingface-cli login

# Download the repo (this can be many GB)
python .\scripts\download_model.py --repo mistralai/Mistral-7B-Instruct-v0.3 --out C:\models\mistral-7b-instruct

# Set LOCAL_MODEL_PATH for this session and run the CLI
$env:LOCAL_MODEL_PATH = 'C:\models\mistral-7b-instruct'
python -m mvp.run_chat
```

4) Run immediately with a mock backend (no download required)

```powershell
python -m mvp.run_chat --mock
```

Notes
- The Mistral 7B model still requires significant disk space and memory. On GPUs with constrained RAM, consider quantized weights and `bitsandbytes` support (not included here).
- If the HF model is gated, be sure to request access at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

Files added by this minimal scaffold:
- `scripts/download_model.py` — helper wrapper around `huggingface_hub.snapshot_download`.
- `mvp/backend_local.py` — local transformers loader (uses `LOCAL_MODEL_PATH` or HF id).
- `mvp/backend_mock.py` — a tiny canned interviewer for development.
- `mvp/run_chat.py` — simple CLI (use `--mock` to avoid downloading weights).
- `mvp/trial.py` — scripted short interview harness.

If you'd like, I can add quantified loading (bitsandbytes) or a small web UI next.
