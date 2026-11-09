# AI Text Watermarking Lab

A portfolio research project investigating statistical text watermarks with reproducible local experiments.

## Workspace

Open this folder in VS Code and select `.venv\Scripts\python.exe` as the Python interpreter.

```powershell
.\.venv\Scripts\python.exe scripts\check_setup.py
```

The initial model is `HuggingFaceTB/SmolLM2-360M-Instruct`, downloaded to `models/smollm2-360m-instruct`. It is an ungated small model for quick experiments; Gemma is a later option requiring license/access review. Model files are excluded from Git.

## Research direction

1. White box: compare generation with watermarking enabled and disabled, including logits.
2. Controlled access: put the local generator behind a CLI, keeping ground truth separate from the experiment.
3. Black box: study output-only detection and robustness, measuring sample requirements and uncertainty.

Track detection, effects of controlled rewriting, and whether restricted observations reveal useful watermark information. Key recovery is a research question, not a promised outcome.

## Important assumptions to investigate

- A simple green-list watermark and SynthID are different algorithms; implement and label them separately.
- A z-score or p-value is not a posterior percentage probability that text is watermarked.
- Twenty repeated completions do not establish secret-key recovery.
- Fixed phrases can leave little room for watermark bias. Compare constrained and open-ended prompts.
- This chat cannot expose private internal thinking tokens or supply a known commercial watermark key. Use a local backend for experiments with known ground truth.
- Credit AI assistance and list only tools actually used in the final report.

## Sources

- https://ai.google.dev/responsible/docs/safeguards/synthid
- https://huggingface.co/blog/synthid-text
- https://github.com/google-deepmind/synthid-text

See `HANDOFF.md` for setup state and next steps.

The local `setup.ps1` installs the GPU build of PyTorch and project dependencies, then runs the offline smoke check. Its local status is written to `SETUP_STATUS.md`. The model and NLTK datasets are local downloads and are excluded from Git. To download the model again:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-360M-Instruct', local_dir='models/smollm2-360m-instruct', allow_patterns=['*.json','*.safetensors','*.txt','*.md'])"
```
