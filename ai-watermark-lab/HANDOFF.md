# Continue in VS Code

## Goal and source

This is a high-school portfolio research lab on AI text watermarking, based on the author's brainstorming notes from November 9, 2026.

The brainstorm proposes white-box local model experiments, a controlled CLI chat, and output-only black-box tests. Priorities: detection, robustness to rewriting, and investigating secret-key recovery. It suggests SynthID, logits comparisons, n-gram statistics, synonym replacement, and repeated single-word completions. These are proposed experiments, not established results.

Additional question: could predictable pangram completions reveal a watermark, perhaps through differences in thinking tokens? Keep this as a hypothesis; local controlled output experiments are needed, and private thinking tokens are unavailable here.

## Environment

- Windows, Python 3.11; dedicated `.venv` in this folder.
- NVIDIA GeForce RTX 3060 Ti with 8 GB VRAM.
- VS Code interpreter and extension recommendations configured.
- Dependencies and a small ungated model are being installed in the setup session; consult `SETUP_STATUS.md` for verified completion.
- No experiment code or research results yet.

## Next work

## User's latest product direction

Build a visually striking TUI research tool with graphs, charts, and mathematical explanations. Display actual measured experiment data. Never present fabricated chart values as results. Distinguish observed results, analytical expectations, and any explicitly labeled demonstrations. Use Textual with plotext/textual-plotext and Rich. Attach model, seed, sampling configuration, sample counts, and watermark settings to experiment records. Repository: https://github.com/TomatoBr0/Portfolio, project folder: ai-watermark-lab.

Explain each step before running commands. Let the user choose meaningful configuration settings. The user specifically chose ai-watermark-lab as the folder name. Model and TUI settings should be discussed before implementation.

## Implementation next steps

1. Run `scripts/check_setup.py` and inspect `SETUP_STATUS.md`.
2. Define a first measurable experiment: fixed prompts, watermark on/off, fixed sampling settings, held-out controls, false-positive measurement.
3. Build a small green-list baseline separately from SynthID. Use the Transformers SynthID implementation for its own experiment.
4. Save configurations, seeds, model revision, outputs, and results. Avoid calling p-values confidence percentages.

The user requested setup and downloads before reopening in VS Code. Do not assume they requested a website or that key recovery is feasible.
