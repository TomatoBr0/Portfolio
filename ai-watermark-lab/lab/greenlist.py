"""Educational keyed green-list baseline; deliberately separate from SynthID."""
import hashlib
import math
from collections import OrderedDict

import torch


class GreenList:
    """Select a fixed-size vocabulary subset from the key and previous token.

    A local CPU generator keeps selection independent of generation's RNG.
    This is an educational scheme, not a production cryptographic design.
    """

    name = "greenlist-v1"

    def __init__(self, vocab_size: int, key: int, fraction: float = 0.25):
        self.vocab_size = vocab_size
        self.key = key
        self.size = int(vocab_size * fraction)
        if not 0 < self.size < vocab_size:
            raise ValueError("Green list must include some, but not all, tokens.")
        self.fraction = self.size / vocab_size
        self._cache = OrderedDict()

    def ids(self, previous: int) -> tuple[int, ...]:
        if previous in self._cache:
            self._cache.move_to_end(previous)
            return self._cache[previous]
        digest = hashlib.sha256(f"{self.name}:{self.key}:{previous}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = tuple(torch.randperm(self.vocab_size, generator=generator)[:self.size].tolist())
        self._cache[previous] = selected
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return selected

    def detect(self, tokens: list[int], previous: int) -> dict:
        seen = set()
        hits = 0
        scored = 0
        trace = []
        for position, token in enumerate(tokens, 1):
            # Only the first occurrence of each context contributes evidence.
            # Repeated phrases must not inflate the nominal sample size.
            included = previous not in seen
            green = token in self.ids(previous)
            if included:
                scored += 1
                hits += int(green)
                seen.add(previous)
            z = (hits - self.fraction * scored) / math.sqrt(
                scored * self.fraction * (1 - self.fraction)
            )
            trace.append({"position": position, "token_id": token, "green": green,
                          "scored": included, "z": z})
            previous = token
        z = trace[-1]["z"] if trace else None
        return {"scored_tokens": scored, "green_hits": hits, "z": z,
                "approx_p": 0.5 * math.erfc(z / math.sqrt(2)) if z is not None else None,
                "trace": trace}


class GreenListProcessor:
    """Transformers-compatible logits hook; records a shared-prefix comparison."""

    def __init__(self, method: GreenList, bias: float, temperature: float, tokenizer,
                 on_token=None):
        self.method = method
        self.bias = bias
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.on_token = on_token
        self.first_step = []
        self.steps = 0

    def __call__(self, input_ids, scores):
        green_ids = self.method.ids(int(input_ids[0, -1]))
        boosted = scores.clone()
        boosted[:, list(green_ids)] += self.bias
        if self.steps == 0:
            before = torch.softmax(scores[0].float() / self.temperature, dim=-1)
            after = torch.softmax(boosted[0].float() / self.temperature, dim=-1)
            candidates = set(before.topk(8).indices.tolist() + after.topk(8).indices.tolist())
            green_set = set(green_ids)
            self.first_step = [
                {"token_id": token, "token": self.tokenizer.decode([token]),
                 "green": token in green_set, "before": float(before[token]),
                 "after": float(after[token])}
                for token in sorted(candidates, key=lambda token: -float(before[token]))
            ]
        self.steps += 1
        if self.on_token:
            self.on_token(self.steps)
        return boosted
