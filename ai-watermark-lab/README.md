# AI Text Watermarking Lab

A local terminal workbench for inspecting statistical text watermarks with real model data.

## Start

From the activated Python environment in this folder:

```powershell
python -m lab.app
```

The fixed-screen layout puts short, data-aware explanations on the left and controls,
graphs, or answers on the right. Back/Next changes views. Row lists, long answers,
and longer explanations have page buttons. The app opens straight into the lab;
there is no required tutorial. F1 returns to setup, Ctrl+Q exits.

## Experiments

- **Paired OFF/ON:** same prompt, model, and sampling seed; compare ordinary generation with a green-list watermark.
- **Bulk baseline + held-out test:** generate OFF calibration answers, freeze a threshold, then measure false alarms on fresh OFF answers and detections on fresh ON answers. Default: 40 calibration answers + 20 test pairs = 80 total answers. Prompts are sampled from the user's prompt plus a small built-in bank; calibration and test sampling seeds are disjoint. Counts and the false-alarm target are editable.
- **Strength sweep:** compare a regular OFF control against ON bias levels 0, 0.75, the configured bias, and 2.5 (deduplicated). Repeats reuse prompts and sampling seeds across strengths. The graph shows measured mean pattern scores, not accuracy.

On the Run page, check the sample/token budget before starting. Stop + save retains
completed answers, marks the run cancelled, and discards the unfinished answer.
Load latest saved run restores the most recent JSON export. Loaded runs show plain
text unless a tokenizer is already loaded; new live runs can highlight token colors.

## Measurements and limits

The educational `greenlist-v1` method hashes a key and previous token, selects a
fixed-size token subset, and adds bias to favored logits. It is separate from SynthID.
The pattern score is the standard z-score with descriptive labels: excess green hits
divided by expected fluctuation. Repeated previous-token contexts are excluded from
scoring. A z-score or approximate normal-tail p-value is not a watermark probability.

Bulk calibration chooses the `ceil((n+1)*(1-target))` ordered OFF score and flags
scores **strictly above** that threshold. Only calibration rows choose the threshold;
held-out results do not tune it. Too few valid calibration samples produce no finite
threshold or rates. Held-out false-alarm and detection rates include approximate 95%
Wilson intervals. These rates apply to this model, fixed watermark key, token limit,
and small prompt mixture. Reusing evaluation results to tune settings stops being a
fresh held-out evaluation. Zero observed false alarms does not mean a zero error rate.

The probability chart uses the stored prefix and bias of one identified marked
sample, comparing OFF/ON probabilities at the same context. Only four candidates
are shown. Each answer records first-step entropy (bits) before and after the bias;
this describes uncertainty at that step, not the whole answer or its writing quality.

## Files and verification

`lab/app.py`: fixed-screen UI. `lab/experiment.py`: local generation and JSON export.
`lab/greenlist.py`: watermark and matching-key detector. `lab/suites.py`: sample plans,
calibration, and held-out analysis. `lab/teaching.py`: earlier detailed explanation
templates retained for reuse.

Raw data is saved to ignored `outputs/*.json`: prompts, seeds, token IDs, generation
settings, model revision, software versions, scores, timings, and analysis. Experiment
keys are stored for reproducibility. `.venv`, models, datasets and logs stay out of Git.

```powershell
python -m pytest -q
python -m ruff check lab tests
python scripts\check_setup.py
```

The downloaded starter model is `HuggingFaceTB/SmolLM2-360M-Instruct` (360M parameters,
about 724 MB of weights). `setup.ps1` installs GPU PyTorch, dependencies, the model and
NLTK resources, then runs a local generation check. Setup status/logs are local files.
The environment was verified on an NVIDIA RTX 3060 Ti with 8 GB VRAM.

## References

- [SynthID overview](https://ai.google.dev/responsible/docs/safeguards/synthid)
- [Transformers generation](https://huggingface.co/docs/transformers/generation_strategies)
- [Finite-sample quantile background](https://www.stat.berkeley.edu/~ryantibs/papers/classconf.pdf)
- [NIST proportion confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
- [User-supplied introductory video](https://www.youtube.com/watch?v=Cmi-1QSaptA) (the user's summary informed the token-color and low-uncertainty explanation).

Future work: additional models/method adapters, SynthID experiments, robustness to
rewriting, and restricted-access experiments. This is not a universal AI-text detector.
