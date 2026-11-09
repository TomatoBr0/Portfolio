"""Check local GPU, watermark imports, and offline model loading."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "smollm2-360m-instruct"
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("SynthID configuration:", SynthIDTextWatermarkingConfig.__name__)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
inputs = tokenizer("A useful science experiment", return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print("Offline generation:", tokenizer.decode(output[0], skip_special_tokens=True))
print("Setup check passed.")
