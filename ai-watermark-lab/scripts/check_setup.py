"""Check local GPU, watermark imports, and offline model loading."""
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.panel import Panel


def main():
    console = Console()
    started = perf_counter()
    console.print("\n[bold cyan]AI Watermark Lab[/] [dim]/ setup check[/]\n")
    with console.status("Loading PyTorch and Transformers...", spinner="dots") as status:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            SynthIDTextWatermarkingConfig,
        )

        status.update("Checking GPU access...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[green]OK[/] PyTorch: {torch.__version__}")
        console.print(f"[green]OK[/] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
        console.print(f"[green]OK[/] SynthID: {SynthIDTextWatermarkingConfig.__name__}")

        model_path = Path(__file__).resolve().parents[1] / "models" / "smollm2-360m-instruct"
        status.update("Loading tokenizer from disk...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        status.update("Loading model weights from disk...")
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        status.update(f"Moving model to {device.upper()}...")
        model = model.to(device).eval()
        inputs = tokenizer("A useful science experiment", return_tensors="pt").to(device)
        status.update("Generating a short sample...")
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=12, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        sample = tokenizer.decode(output[0], skip_special_tokens=True)

    console.print(Panel(sample, title="Offline generation", border_style="cyan"))
    console.print(f"[bold green]Setup check passed.[/] [dim]{perf_counter() - started:.1f}s[/]\n")


if __name__ == "__main__":
    main()
