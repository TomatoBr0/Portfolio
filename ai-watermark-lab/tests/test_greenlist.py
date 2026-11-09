import math

import torch

from lab.greenlist import GreenList, GreenListProcessor


def test_greenlist_does_not_consume_generation_randomness():
    torch.manual_seed(100)
    expected = torch.rand(4)
    torch.manual_seed(100)
    first = GreenList(100, 42)
    assert first.ids(7) == GreenList(100, 42).ids(7)
    assert first.ids(7) != GreenList(100, 43).ids(7)
    assert len(set(first.ids(7))) == 25
    assert torch.equal(expected, torch.rand(4))


def test_repeated_contexts_do_not_inflate_score():
    method = GreenList(100, 42)
    detection = method.detect([7, 7, 7, 7], previous=7)
    assert detection["scored_tokens"] == 1
    assert sum(token["scored"] for token in detection["trace"]) == 1
    assert method.detect([], 7)["z"] is None


def test_score_matches_hand_calculation():
    method = GreenList(100, 42)
    token = method.ids(9)[0]
    result = method.detect([token], previous=9)
    assert result["green_hits"] == 1
    assert math.isclose(result["z"], (1 - 0.25) / math.sqrt(0.25 * 0.75))


def test_bias_changes_only_green_logits_and_zero_bias_is_control():
    class Tokenizer:
        def decode(self, ids):
            return str(ids[0])
    method = GreenList(100, 42)
    scores = torch.zeros(1, 100)
    processor = GreenListProcessor(method, 1.5, 0.8, Tokenizer())
    actual = processor(torch.tensor([[9]]), scores)
    expected = scores.clone()
    expected[:, list(method.ids(9))] = 1.5
    assert torch.equal(actual, expected)
    assert torch.equal(scores, torch.zeros_like(scores))
    control = GreenListProcessor(method, 0, 0.8, Tokenizer())
    assert torch.equal(control(torch.tensor([[9]]), scores), scores)
