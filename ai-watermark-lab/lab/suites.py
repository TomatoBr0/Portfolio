"""Experiment plans and held-out baseline analysis, independent of the UI/model."""
import math
import random
import statistics

PROMPTS = [
    "Explain how a volcanic island forms, step by step.",
    "Describe how a garden changes through the four seasons.",
    "Explain how a bicycle's gears help a rider climb a hill.",
    "Describe the journey of a drop of water through the water cycle.",
    "Write a short story about finding an old map in a library.",
    "Explain how to plan and cook a simple meal for friends.",
]


def make_plan(settings):
    choices = random.Random(settings.seed)
    prompts = list(dict.fromkeys([settings.prompt, *PROMPTS]))
    jobs = []

    def add(phase, pair, marked, bias, prompt, seed):
        jobs.append({"phase": phase, "pair": pair, "marked": marked,
                     "bias": bias, "prompt": prompt, "seed": seed})

    if settings.kind == "baseline":
        for i in range(settings.calibration_samples):
            add("calibration", i + 1, False, 0, choices.choice(prompts), settings.seed + i)
        for i in range(settings.evaluation_pairs):
            prompt = choices.choice(prompts)
            seed = settings.seed + settings.calibration_samples + i
            for marked in (False, True):
                add("evaluation", i + 1, marked, settings.bias if marked else 0, prompt, seed)
    elif settings.kind == "strength":
        strengths = sorted(set([0.0, 0.75, settings.bias, 2.5]))
        for i in range(settings.pairs):
            prompt = choices.choice(prompts)
            add("sweep", i + 1, False, 0, prompt, settings.seed + i)
            for bias in strengths:
                add("sweep", i + 1, True, bias, prompt, settings.seed + i)
    else:
        for i in range(settings.pairs):
            for marked in (False, True):
                add("paired", i + 1, marked, settings.bias if marked else 0, settings.prompt, settings.seed + i)
    return jobs


def wilson(successes, total):
    if not total:
        return {"count": successes, "total": 0, "rate": None, "interval95": None}
    rate = successes / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"count": successes, "total": total, "rate": rate,
            "interval95": [max(0, center - half), min(1, center + half)]}


def analyze_baseline(rows, target):
    calibration = sorted(row["z"] for row in rows
                         if row["phase"] == "calibration" and not row["marked"] and row["z"] is not None)
    rank = math.ceil((len(calibration) + 1) * (1 - target))
    # Never substitute a finite threshold when the sample resolution is too low.
    threshold = calibration[rank - 1] if calibration and rank <= len(calibration) else None
    evaluation = [row for row in rows if row["phase"] == "evaluation" and row["z"] is not None]
    off = [row for row in evaluation if not row["marked"]]
    on = [row for row in evaluation if row["marked"]]
    return {
        "calibration_count": len(calibration), "target_false_alarm_rate": target,
        "threshold": threshold, "rule": "flag when pattern score is strictly greater than threshold",
        "calibration_mean": statistics.mean(calibration) if calibration else None,
        "calibration_stdev": statistics.stdev(calibration) if len(calibration) > 1 else None,
        "false_alarms": wilson(sum(row["z"] > threshold for row in off), len(off)) if threshold is not None else None,
        "detection": wilson(sum(row["z"] > threshold for row in on), len(on)) if threshold is not None else None,
        "unscored": sum(row["z"] is None for row in rows),
        "note": "Threshold uses calibration OFF scores only. Evaluation seeds are disjoint. "
                "Rates apply to this model, key, token limit and prompt mixture; not universal detection accuracy. "
                "Wilson intervals are approximate and assume independent samples within each rate. "
                "ON/OFF samples within a pair share a seed. Repeated tuning on these test results invalidates a fresh-test interpretation.",
    }


def analyze_strength(rows):
    groups = []
    for bias in sorted({row["bias"] for row in rows if row["marked"]}):
        selected = [row for row in rows if row["marked"] and row["bias"] == bias and row["z"] is not None]
        scores = [row["z"] for row in selected]
        groups.append({"bias": bias, "samples": len(scores),
                       "mean_score": statistics.mean(scores) if scores else None,
                       "stdev_score": statistics.stdev(scores) if len(scores) > 1 else None})
    return groups
