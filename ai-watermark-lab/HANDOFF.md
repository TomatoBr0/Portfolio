# Continue in VS Code

## Goal

Build a high-school portfolio research tool for AI text watermarks. Original roadmap:
white-box, controlled CLI access, then output-only experiments; detection, rewriting
robustness, and investigating whether key recovery is feasible. Do not promise key recovery.

## User preferences / latest direction

- Explain steps before running commands; let the user choose meaningful configuration.
- Fixed-screen TUI: short explanation on the LEFT, actual data on the RIGHT; Back/Next
  switches views. No long scrolling dashboard. The user is done with the tutorial;
  launch directly into the lab. Longer explanations/answers and sample tables paginate.
- Keep a flat ASCII/Catppuccin terminal appearance inspired by terminal-web-tui.
- Ground descriptions in current graph values. Use descriptive measurement names:
  pattern score, green hits, scored tokens, green fraction. Do not label z as likelihood.
- User supplied https://www.youtube.com/watch?v=Cmi-1QSaptA plus a summary about
  key/previous-token hashing, favored tokens, bias and the low-entropy trade-off.
  Use that framing; don't claim to have watched it. First-token entropy is now recorded.

## Current implementation

Launch `python -m lab.app` from this folder's activated .venv. Seven fixed views:
setup, tuning, run, summary, score graph, probability chart, answer excerpts.
Experiments: paired OFF/ON, bulk baseline with disjoint calibration/evaluation seeds,
and strength sweep (0, .75, configured bias, 2.5). Bulk defaults 40 calibration OFF
and 20 fresh test pairs. Target false alarms defaults .05. Calibration alone chooses
an upper order-statistic threshold; held-out rates include approximate Wilson intervals.
These are model/prompt/settings-specific estimates, not universal detector accuracy.
Stop + save retains completed samples and marks the run cancelled. Load latest opens
the newest local JSON. Model/tokenizer stay cached across experiments.

`lab/app.py` UI; `lab/experiment.py` runner/settings; `lab/suites.py` sample plans and
stats; `lab/greenlist.py` educational scheme/detector. `lab/teaching.py` earlier detailed
templates remain available but aren't the default UI. Outputs are ignored JSON files.
Every row stores prompt, phase, seed, strength, token IDs, scores, timing, first-step
entropy. Probability-chart source sample/bias is recorded. Repeated contexts are not
counted again by the detector. The scheme is separate from SynthID.

## Environment / verification

Windows, Python 3.11, RTX 3060 Ti (8 GB), CUDA PyTorch 2.14.1+cu126,
Transformers 4.57.6. Local SmolLM2-360M-Instruct weights (~724 MB) and NLTK resources
are downloaded. Textual, plotext, data tools, pytest and Ruff installed; VS Code
Python, Jupyter and Ruff extensions installed.

Tests cover RNG isolation, score formula, repeated-context exclusion, narrative
substitution, 80x24/120x45 paging, calibration/test separation, threshold leakage,
Wilson uncertainty and sweep plans. Real GPU baseline (20 calibration + 5 test pairs)
and sweep smoke runs passed, as did their graphs/export UI. These short runs verify
software behavior, not scientific detection performance. Run pytest and Ruff after edits.

## Repository

https://github.com/TomatoBr0/Portfolio ; project is ai-watermark-lab/.
Use connected GitHub tools for writes: local git push opens an authentication selector.
Public git fetch works with `git -c credential.helper= -c core.askPass= fetch origin`.
Models, environment, datasets, logs and outputs are ignored. Old portfolio-project
folder is obsolete; the user will delete it manually after a policy-blocked deletion.
