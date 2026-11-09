"""Run local watermark experiments and preserve measured evidence."""
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .suites import PROMPTS, analyze_baseline, analyze_strength, make_plan

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
    kind: str = "paired"
    calibration_samples: int = 40
    evaluation_pairs: int = 20
    target_false_alarm: float = 0.05

    def validate(self):
        if self.kind not in ("paired", "baseline", "strength"):
            raise ValueError("Choose paired, baseline, or strength.")
        if not self.prompt.strip():
            raise ValueError("Enter a prompt first.")
        if not 16 <= self.tokens <= 512:
            raise ValueError("Use 16 to 512 generated tokens.")
        if not 1 <= self.pairs <= 50:
            raise ValueError("Use 1 to 50 pairs/repeats.")
        if not 20 <= self.calibration_samples <= 500 or not 5 <= self.evaluation_pairs <= 250:
            raise ValueError("Use 20-500 calibration answers and 5-250 evaluation pairs.")
        if not math.isfinite(self.target_false_alarm) or not 0.01 <= self.target_false_alarm <= 0.2:
            raise ValueError("Target false alarms must be 0.01 to 0.20 (1% to 20%).")
        if self.kind == 'baseline' and math.ceil((self.calibration_samples + 1) * (1 - self.target_false_alarm)) > self.calibration_samples:
            raise ValueError(f"This false-alarm target needs at least {math.ceil(1 / self.target_false_alarm) - 1} calibration samples.")
        if not math.isfinite(self.bias) or not 0 <= self.bias <= 5:
            raise ValueError("Bias must be between 0 and 5.")
        if not math.isfinite(self.fraction) or not 0.05 <= self.fraction <= 0.95:
            raise ValueError("Green fraction must be between 0.05 and 0.95.")
        if not math.isfinite(self.temperature) or not 0.1 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0.1 and 2.")
        if not 0 <= self.seed <= 2**32 - 1001:
            raise ValueError("Seed must be between 0 and 4294966295.")


class Runner:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def run(self, settings, progress=lambda message: None, cancelled=lambda: False):
        settings.validate()
        progress("Loading local model...")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from .greenlist import GreenList, GreenListProcessor

        model_path = ROOT / "models" / "smollm2-360m-instruct"
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        tokenizer, model = self.tokenizer, self.model
        method = GreenList(model.get_output_embeddings().weight.shape[0], settings.key, settings.fraction)
        jobs = make_plan(settings)
        rows, first_step = [], []
        probability_sample = None
        encoded = {}
        interrupted = False
        for job_index, job in enumerate(jobs):
            if cancelled():
                interrupted = True
                break
            prompt = job['prompt']
            if prompt not in encoded:
                encoded[prompt] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
            prompt_ids = encoded[prompt]
            if prompt_ids.shape[-1] + settings.tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt plus answer exceeds this model's context window.")
            label = 'ON' if job['marked'] else 'OFF'

            def token_progress(step, job_index=job_index, label=label, phase=job['phase']):
                if cancelled():
                    raise InterruptedError("Stopped by user")
                if step == 1 or step % 8 == 0:
                    progress(f"{phase} | sample {job_index + 1}/{len(jobs)} | {label} | token {step}/{settings.tokens}")

            processor = GreenListProcessor(method, job['bias'], settings.temperature, tokenizer, token_progress)
            set_seed(job['seed'])
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            started = perf_counter()
            try:
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                        max_new_tokens=settings.tokens, do_sample=True,
                        temperature=settings.temperature, top_k=0, top_p=1.0,
                        repetition_penalty=1.0, logits_processor=[processor],
                        pad_token_id=tokenizer.eos_token_id,
                    )
            except InterruptedError:
                interrupted = True
                break
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = perf_counter() - started
            ids = output[0, prompt_ids.shape[-1]:].tolist()
            evidence = ids[:-1] if ids and ids[-1] == tokenizer.eos_token_id else ids
            detection = method.detect(evidence, int(prompt_ids[0, -1]))
            rows.append({**job, 'token_ids': ids, 'prompt_token_ids': prompt_ids[0].tolist(),
                         'first_token_entropy_before': processor.entropy_before,
                         'first_token_entropy_after': processor.entropy_after,
                         'text': tokenizer.decode(ids, skip_special_tokens=True),
                         'generated_tokens': len(ids), 'seconds': elapsed,
                         'tokens_per_second': len(ids) / elapsed, **detection})
            if not first_step and job['marked'] and (settings.kind != 'strength' or job['bias'] > 0):
                first_step = processor.first_step
                probability_sample = {'row_index': len(rows) - 1, 'bias': job['bias'],
                                      'prompt': job['prompt'], 'seed': job['seed']}
        metadata = model_path / '.cache/huggingface/download/model.safetensors.metadata'
        try:
            revision = metadata.read_text().splitlines()[0]
        except (OSError, IndexError):
            revision = None
        result = {
            'schema_version': 2, 'created_at': datetime.now(UTC).isoformat(),
            'kind': settings.kind, 'status': 'cancelled' if interrupted else 'complete',
            'planned_samples': len(jobs), 'completed_samples': len(rows),
            'method': method.name, 'model': 'HuggingFaceTB/SmolLM2-360M-Instruct',
            'model_config_sha256': hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest(),
            'download_source_revision': revision, 'device': str(model.device),
            'gpu': torch.cuda.get_device_name(0) if model.device.type == 'cuda' else None,
            'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__},
            'settings': asdict(settings), 'actual_green_fraction': method.fraction,
            'prompt_bank': list(dict.fromkeys([settings.prompt, *PROMPTS])) if settings.kind != 'paired' else [settings.prompt],
            'sampling': {'do_sample': True, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0},
            'rows': rows, 'first_step': first_step, 'probability_sample': probability_sample,
            'limitations': ['Matching-key educational detector, not universal AI detection.',
                            'Normal-tail p-values are approximate; baseline evaluation uses an empirical threshold.',
                            'Results apply to this fixed model, key, settings and prompt mixture.'],
        }
        if settings.kind == 'baseline':
            result['baseline'] = analyze_baseline(rows, settings.target_false_alarm)
        if settings.kind == 'strength':
            result['strength'] = analyze_strength(rows)
        path = ROOT / 'outputs' / f"{settings.kind}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        progress(f"{'Stopped; completed samples saved' if interrupted else 'Complete'} | {len(rows)}/{len(jobs)} samples")
        return result, path
