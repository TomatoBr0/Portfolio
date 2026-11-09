from lab.teaching import explain_probabilities, explain_sample, explain_trace


def sample(marked, hits, z):
    return {"pair": 2, "marked": marked, "scored_tokens": 40, "green_hits": hits,
            "z": z, "approx_p": 0.03,
            "trace": [{"position": 3, "z": 1.1}, {"position": 50, "z": z}]}


def test_explanations_use_selected_counts_and_do_not_promise_success():
    off, on = sample(False, 20, 3.65), sample(True, 5, -1.83)
    result = {"actual_green_fraction": 0.25, "rows": [off, on]}
    text = explain_sample(result, on)
    assert "pair 2 / ON" in text
    assert "**40 text tokens**" in text
    assert "**5 green hits**" in text
    assert "**10.00 hits**" in text
    assert "**5.00 fewer**" in text
    assert "does not show a higher ON score" in text
    trace = explain_trace(result, on)
    assert "**50**" in trace
    assert "**z = -1.83**" in trace
    assert "**1.10 at position 3**" in trace


def test_probability_explanation_uses_percentage_points_and_candidate_number():
    result = {"first_step": [{"token": " a", "green": True, "before": 0.1, "after": 0.15}],
              "settings": {"bias": 1.5, "temperature": 0.8}}
    text = explain_probabilities(result)
    assert "group 1" in text
    assert "10.000%" in text
    assert "15.000%" in text
    assert "5.000 percentage points" in text
    assert "first step of pair 1" in text
