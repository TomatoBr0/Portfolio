"""Run paired, local experiments and preserve their measured evidence."""
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    prompt: str = "Explain how a volcanic island forms, step by step."
    key: int = 42
    bias: float = 1.5
    fraction: float = 0.25
    temperature: float = 0.8
    tokens: int = 128
    seed: int = 123
    pairs: int = 1

    def validate(self):
        if not self.prompt.strip():
            raise ValueError("Enter a prompt first.")
        if not 16 <= self.tokens <= 512:
            raise ValueError("Use 16 to 512 generated tokens.")
        if not 1 <= self.pairs <= 10:
            raise ValueError("Use 1 to 10 pairs per run.")
        if not math.isfinite(self.bias) or not 0 <= self.bias <= 5:
            raise ValueError("Bias must be between 0 and 5.")
        if not math.isfinite(self.fraction) or not 0.05 <= self.fraction <= 0.95:
            raise ValueError("Green fraction must be between 0.05 and 0.95.")
        if not math.isfinite(self.temperature) or not 0.1 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0.1 and 2.")
        if not 0 <= self.seed <= 2**32 - 11:
            raise ValueError("Seed must be between 0 and 4294967285.")


class Runner:
    """Cache one backend; method and UI concerns stay separate for future adapters."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def run(self, settings: Settings, progress=lambda message: None):
        settings.validate()
        progress("Loading PyTorch and watermark method...")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from .greenlist import GreenList, GreenListProcessor

        model_path = ROOT / "models" / "smollm2-360m-instruct"
        if self.model is None:
            progress("Loading the local model into memory...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        tokenizer, model = self.tokenizer, self.model
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": settings.prompt}], add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        if prompt_ids.shape[-1] + settings.tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt plus generated tokens exceeds this model's context window.")
        method = GreenList(model.get_output_embeddings().weight.shape[0],
                           settings.key, settings.fraction)
        rows = []
        first_step = []
        for pair in range(settings.pairs):
            for marked in (False, True):
                label = "watermark ON" if marked else "watermark OFF"
                def token_progress(step, pair=pair, label=label):
                    if step == 1 or step % 8 == 0 or step == settings.tokens:
                        progress(f"Pair {pair + 1}/{settings.pairs} | {label} | token {step}/{settings.tokens}")
                processor = GreenListProcessor(
                    method, settings.bias if marked else 0, settings.temperature,
                    tokenizer, token_progress,
                )
                set_seed(settings.seed + pair)
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                started = perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                        max_new_tokens=settings.tokens, do_sample=True,
                        temperature=settings.temperature, top_k=0, top_p=1.0,
                        repetition_penalty=1.0, logits_processor=[processor],
                        pad_token_id=tokenizer.eos_token_id,
                    )
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - started
                ids = output[0, prompt_ids.shape[-1]:].tolist()
                # End-of-sequence is a stopping signal, not displayed text evidence.
                evidence_ids = list(ids)
                if evidence_ids and evidence_ids[-1] == tokenizer.eos_token_id:
                    evidence_ids.pop()
                detection = method.detect(evidence_ids, int(prompt_ids[0, -1]))
                rows.append({"pair": pair + 1, "marked": marked, "seed": settings.seed + pair,
                             "token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True),
                             "generated_tokens": len(ids), "seconds": elapsed,
                             "tokens_per_second": len(ids) / elapsed, **detection})
                if pair == 0 and marked:
                    first_step = processor.first_step
        config_bytes = (model_path / "config.json").read_bytes()
        metadata_path = model_path / ".cache/huggingface/download/model.safetensors.metadata"
        try:
            source_revision = metadata_path.read_text().splitlines()[0]
        except (OSError, IndexError):
            source_revision = None
        result = {
            "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
            "method": method.name, "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "local_model_revision": getattr(model.config, "_commit_hash", None),
            "download_source_revision": source_revision,
            "sampling": {"do_sample": True, "top_k": 0, "top_p": 1.0,
                         "repetition_penalty": 1.0},
            "device": str(model.device),
            "gpu": torch.cuda.get_device_name(0) if model.device.type == "cuda" else None,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__},
            "settings": asdict(settings), "actual_green_fraction": method.fraction,
            "prompt_token_ids": prompt_ids[0].tolist(), "rows": rows,
            "first_step": first_step,
            "limitations": ["Approximate z/p values; null is not empirically calibrated.",
                             "Only first occurrence of each previous-token context is scored.",
                             "Paired outputs share sampling seeds and are not independent replicates.",
                             "Matching-key detection for this method only; not a universal AI detector."],
        }
        folder = ROOT / "outputs"
        folder.mkdir(exist_ok=True)
        path = folder / f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        progress("Experiment saved.")
        return result, path
