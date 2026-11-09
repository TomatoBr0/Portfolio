"""Plain-language explanations tied to measured experiment values."""

SETTINGS_HELP = """### What am I choosing?

- **Secret key:** a number that chooses which tokens get favored. It is a lab
  password shared by the generator and detector. Changing it changes the lists.
- **Bias / strength:** how hard we nudge favored tokens. 0 applies no nudge.
  1.5 is a starting point, not 1.5% and not a guarantee of detection.
- **Green fraction:** how much of the vocabulary is favored at each step.
  0.25 means 25% of token choices, not 25% of the finished text.
- **Temperature:** how adventurous the sampling is. Lower values concentrate
  probability on high-scoring tokens; higher values spread it out.
- **Max tokens:** how long each answer can be. Tokens are pieces of words or
  punctuation, so this is not a word count. Answers can stop early.
- **Sampling seed:** the starting point of the random-number generator. We reuse
  it within each pair to make the comparison more controlled. This is separate
  from the watermark key.
- **OFF/ON pairs:** each pair contains one regular answer and one watermarked
  answer. More pairs show how much results vary, but take longer.

**What happens when I run?** We give the same prompt to the same local model twice.
OFF uses ordinary sampling. ON adds the green-list nudge. Both answers are AI
generated; this experiment compares watermarking, not AI versus human writing.
"""

TABLE_HELP = """### 1 / Start with the measurements

Each row is a real generated answer. **OFF/ON is the known setting we used**,
not a detector prediction. A **pair** is one OFF answer and one ON answer.

**Tokens** is the answer length including a final stop token, if emitted.
**Scored tokens** counts text tokens used by the detector; repeated previous-token
contexts and the final stop token are excluded. **Green hits** counts favored tokens
among those scored. An ordinary answer can contain green tokens by chance.

**Pattern score (z)** measures excess green hits relative to a simple chance baseline.
For example, a score of 2 means two estimated standard deviations above the baseline;
it does not mean 2% or twice as likely. **Chance tail (approximate p-value)** is the
one-sided normal-tail calculation from that score: a smaller value means a more
unusual result under the simplified chance model. **Tokens/sec** is generation speed,
not detection accuracy. This small run does not establish a false-positive rate.
"""

TRACE_HELP = """### 2 / Did the pattern build up as the answer grew?

Read left to right: **horizontal axis = generated token position**,
**vertical axis = cumulative pattern score (z)**.
At each position we rescore all eligible tokens seen so far, rather than scoring
only that single token. Cyan is watermark OFF; green is watermark ON.

**Zero** is the simple baseline expectation. Positive values mean more green hits
than expected; negative values mean fewer. A climbing green line means the ON
sample is accumulating evidence of our pattern. Both lines can wobble, overlap,
or cross. Flat steps can come from skipped repeated contexts.

There is no calibrated pass/fail line yet. One high score is not proof, and a low
score does not prove absence. Use the sample selector below to change the pair.
"""

PROBABILITY_HELP = """### 3 / How did the watermark change the next choice?

This is a microscope on **one step**, immediately after the prompt. We take the
same model scores, apply the selected temperature, and compare probabilities
with and without the bias. It is a different view from the whole-answer pattern-score chart.

Each numbered group is one candidate token; the table lists its exact text.
Cyan = its probability with watermark OFF. Green = its probability with ON.
The **Green?** column tells you whether that candidate belongs to the favored list;
the green bar color simply identifies the ON measurement for every candidate.

Favored tokens gain probability and others lose some because probabilities
must sum to 100%. We show only top candidates: the visible bars need not sum
to 100%. A larger chance does not mean that token was necessarily chosen.
"""

TEXT_HELP = """### 4 / Read the actual answer

Choose an OFF or ON sample below. The plain text lets you judge whether the
answer still reads well. Under it, green text marks tokens favored by the key;
pale text marks other tokens. Dim tokens have repeated contexts and were not
counted again. Both OFF and ON answers can have green highlights.

The colors are an inspection overlay. The watermark itself is a statistical
pattern in token choices, not invisible characters embedded in the text.

**Try next:** keep your prompt, key, seed and other settings fixed, then set bias
to 0 and run again. ON now applies no boost, so the pair should match in this
local controlled setup. Then try bias 2.0 and compare the scores and writing.
Short runs can be noisy; more held-out OFF samples are needed to assess mistakes.
"""


def explain_sample(result: dict, row: dict) -> str:
    label = "ON" if row["marked"] else "OFF"
    n, hits = row["scored_tokens"], row["green_hits"]
    if not n:
        return f"## Your selected sample / pair {row['pair']} / {label}\nNo eligible text tokens were available to score."
    fraction = result["actual_green_fraction"]
    expected = n * fraction
    difference = hits - expected
    direction = "more" if difference >= 0 else "fewer"
    deviation = (n * fraction * (1 - fraction)) ** 0.5
    text = (
        f"## Your selected sample / pair {row['pair']} / {label}\n\n"
        f"We scored **{n} text tokens** and found **{hits} green hits** "
        f"({hits / n:.1%} of scored tokens). The simple chance baseline expects "
        f"{fraction:.1%} of {n}, or **{expected:.2f} hits**. "
        f"That is **{abs(difference):.2f} {direction}** than expected.\n\n"
        f"The baseline's estimated standard deviation (its typical scale of "
        f"random fluctuation) is {deviation:.2f} hits. "
        f"So **pattern score (z)** = (green hits - expected hits) / expected fluctuation "
        f"= ({hits} - {expected:.2f}) / {deviation:.2f} = **{row['z']:.2f}**. "
        "It measures distance from the baseline, not a percentage.\n\n"
        f"**Chance tail (approximate p-value) = {row['approx_p']:.3g}.** Under the simplified chance model, "
        "this is the approximate one-sided tail area at this pattern score or higher. "
        "It is not the probability this answer is watermarked. The null has not "
        "been empirically calibrated, especially for short samples.\n\n"
    )
    pair = [sample for sample in result["rows"] if sample["pair"] == row["pair"]]
    off = next((sample for sample in pair if not sample["marked"]), None)
    on = next((sample for sample in pair if sample["marked"]), None)
    if off and on and off["z"] is not None and on["z"] is not None:
        comparison = "higher" if on["z"] > off["z"] else "lower" if on["z"] < off["z"] else "equal"
        text += (
            f"**Pair comparison:** OFF pattern score = {off['z']:.2f}; ON pattern score = {on['z']:.2f}. "
            f"The ON score is {comparison}. "
            + ("That is the direction we hope to see from the watermark boost. "
               if comparison == "higher" else "This pair does not show a higher ON score. That is a result worth recording. ")
            + "One pair cannot tell us how reliable the method is.\n\n"
        )
    return text


def explain_trace(result: dict, row: dict) -> str:
    """Fill the graph's reading guide from this pair's measured trace."""
    samples = [sample for sample in result["rows"] if sample["pair"] == row["pair"]]
    lines = [f"### In this graph / pair {row['pair']}"]
    for sample in samples:
        trace = sample["trace"]
        label = "green ON" if sample["marked"] else "cyan OFF"
        if not trace:
            lines.append(f"The {label} sample has no eligible text to plot.")
            continue
        peak = max(trace, key=lambda point: point["z"])
        endpoint = trace[-1]
        lines.append(
            f"The **{label} line** ends at token position **{endpoint['position']}**, "
            f"at **pattern score (z) = {endpoint['z']:.2f}**. Its highest point is "
            f"**{peak['z']:.2f} at position {peak['position']}**. "
            f"The final score uses {sample['green_hits']} green hits from "
            f"{sample['scored_tokens']} scored tokens."
        )
    lines.append("The highest point and final point can differ: later tokens can strengthen or weaken the pattern. A peak selected after inspecting the graph is not a separate significance test.")
    return TRACE_HELP + "\n\n" + "\n\n".join(lines)


def explain_probabilities(result: dict) -> str:
    """Refer to a visible bar group and identify its corresponding table row."""
    items = result["first_step"][:8]
    if not items:
        return PROBABILITY_HELP + "\n\nNo first-step probabilities were recorded."
    index, item = max(enumerate(items), key=lambda pair: abs(pair[1]["after"] - pair[1]["before"]))
    difference = (item["after"] - item["before"]) * 100
    token = repr(item["token"]).replace("`", "\\`")
    direction = "increased" if difference > 0 else "decreased" if difference < 0 else "stayed unchanged"
    text = (
        f"### In this graph / candidate {index + 1}\n\n"
        f"Look at **group {index + 1}**, token **`{token}`** (the same numbered row "
        f"in the table). Its cyan OFF bar is **{item['before']:.3%}**; its green "
        f"ON bar is **{item['after']:.3%}**. Its probability {direction}"
        + (f" by **{abs(difference):.3f} percentage points**." if difference else ".")
        + f" This token is **{'in' if item['green'] else 'outside'}** the favored list.\n\n"
        f"These values use bias **{result['settings']['bias']}** and temperature "
        f"**{result['settings']['temperature']}** at the first step of pair 1. "
        "This chart stays on that shared first-step comparison when you select other "
        "pairs; it is not a probability breakdown of their entire answers."
    )
    return PROBABILITY_HELP + "\n\n" + text


def explain_table(result: dict) -> str:
    count = len(result["rows"])
    tokens = sum(row["generated_tokens"] for row in result["rows"])
    return TABLE_HELP + (
        f"\n\n**In this table:** {count} answers from {result['settings']['pairs']} "
        f"OFF/ON pair(s), with **{tokens} generated tokens** in total. "
        f"The configured limit was **{result['settings']['tokens']} per answer**. "
        "The chart below follows whichever pair you select in section 4."
    )
