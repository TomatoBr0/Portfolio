# Continue in VS Code

## Goal and source

This is a high-school portfolio research lab on AI text watermarking, based on the author's brainstorming notes from November 9, 2026.

The brainstorm proposes white-box local model experiments, a controlled CLI chat, and output-only black-box tests. Priorities: detection, robustness to rewriting, and investigating secret-key recovery. It suggests SynthID, logits comparisons, n-gram statistics, synonym replacement, and repeated single-word completions. These are proposed experiments, not established results.

Additional question: could predictable pangram completions reveal a watermark, perhaps through differences in thinking tokens? Keep this as a hypothesis; local controlled output experiments are needed, and private thinking tokens are unavailable here.

## Environment

- Windows, Python 3.11; dedicated `.venv` in this folder.
- NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM.
- VS Code interpreter and extension recommendations configured.
- Setup verified on November 9, 2026: PyTorch 2.14.1+cu126, Transformers 4.57.6, CUDA available on the RTX 3060 Ti. Offline SmolLM2 generation and `pip check` passed. Package versions are recorded in requirements-lock.txt. Local logs/status are ignored by Git.
- Model weights and NLTK datasets are downloaded under models/ and data/. Textual, plotext, statistics tools, Jupyter, pytest, and Ruff are installed. VS Code Python, Jupyter, and Ruff extensions are installed.
- First TUI implemented: launch `python -m lab.app`. Three tutorial pages lead to editable paired OFF/ON experiments and measured evidence charts. The tutorial completion preference is local in data/preferences.json.
- Educational greenlist-v1 lives in lab/greenlist.py, paired generation/export in lab/experiment.py, Textual UI in lab/app.py. Each generated sample has token IDs, scores, timing and settings in an ignored outputs/run-*.json file.
- GPU runner and full TUI worker smoke checks passed. Unit tests verify RNG isolation, key-dependent lists, repeated-context exclusion and the score formula. Short setup runs are not validation of detection accuracy.

## User's latest product direction

Build a visually striking TUI research tool with graphs, charts, and mathematical explanations. Display actual measured experiment data. Never present fabricated chart values as results. Distinguish observed results, analytical expectations, and any explicitly labeled demonstrations. Use Textual with plotext/textual-plotext and Rich. Attach model, seed, sampling configuration, sample counts, and watermark settings to experiment records. Repository: https://github.com/TomatoBr0/Portfolio, project folder: ai-watermark-lab.

Explain each step before running commands. Let the user choose meaningful configuration settings. The user specifically chose ai-watermark-lab as the folder name. Model and TUI settings should be discussed before implementation.

## Implementation next steps

1. Have the user launch `python -m lab.app`, read Learn and choose first settings. Explain each change before running it.
2. Collect held-out controls and enough samples to measure false-positive rates before setting detection thresholds.
3. Extend to SynthID and additional backend adapters later; do not conflate these methods.
4. Add run-history loading and comparison as future work. Current UI shows the current session's latest run; earlier JSON files are preserved.

The user requested setup and downloads before reopening in VS Code. Do not assume they requested a website or that key recovery is feasible.
