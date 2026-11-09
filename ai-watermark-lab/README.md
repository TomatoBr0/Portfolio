# AI Text Watermarking Lab

A portfolio research project investigating statistical text watermarks with reproducible local experiments.

## Launch the terminal lab

From the activated environment in this folder:

```powershell
python -m lab.app
```

Start with **Learn** for a three-part walkthrough, then choose your prompt, key,
watermark bias, green fraction, temperature, token limit, sampling seed, and pair count
in **Experiment**. Hover over settings for explanations. Completing the tutorial
remembers your choice locally; F1 reopens it anytime. Ctrl+Q exits.

The **Evidence** tab shows measured cumulative detection scores, same-prefix
next-token probabilities, each generated sample, and green-token highlighting.
Use the sample selector to inspect another pair. Dimmed tokens have repeated
contexts and are excluded from detection scoring. Every run saves JSON under
`outputs/`, including settings, token IDs, model provenance, scores, and timing.
Run records include the experiment key for reproducibility; these are lab keys.

The first method is an educational SHA-256-seeded green-list baseline using the
previous token as context. It is separate from SynthID. The backend, method, and
interface live in separate modules so more methods and models can be added.
Currently, detection requires this scheme and the matching key; it is not a
universal AI-text detector. Scores use an approximate, uncalibrated null model.
The p-value is not a probability of AI authorship. Repeating context and reusing
paired seeds limit independence; a few demonstration pairs do not establish a
false-positive rate or reliable detection threshold.

Checks: `python -m pytest` and `python -m ruff check lab tests`.

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
