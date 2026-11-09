"""Teach first, then explore measured watermark experiments in the terminal."""
import json

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual_plotext import PlotextPlot

from .experiment import ROOT, Runner, Settings

LESSONS = [
    """# 01 / A model chooses tokens

A token is a piece of text: a word, part of a word, or punctuation. The model assigns
scores (**logits**) to possible next tokens. Softmax turns scores into probabilities;
sampling chooses one token. This repeats to form a response.

**What you'll measure:** actual next-token probabilities from your local model.
The charts stay empty until you run an experiment.

This lab begins with an educational green-list watermark. More methods and model
backends can be added later. It does not identify every kind of AI-written text.
""",
    """# 02 / A tiny nudge leaves a pattern

Our **key + previous token** deterministically select a random-looking subset of
the vocabulary: the green list. The list changes as the context changes.

We add a **bias** to green-token logits. Other tokens remain possible. A stronger
bias usually makes the pattern easier to measure, but can affect the writing.

**Green fraction** controls the share of vocabulary selected. **Temperature** controls
sampling randomness. These are different knobs. You'll be able to change both.

The probability chart will compare watermark OFF and ON at the **same first-token
context**, using the selected temperature. These are measured probabilities, not
invented examples.
""",
    """# 03 / Measure evidence, don't guess a percentage

The matching-key detector rebuilds each green list and counts green-token hits.
It scores only the first occurrence of each previous-token context, so repetition
doesn't keep adding the same evidence.

**z = (G - gamma × N) / sqrt(N × gamma × (1 - gamma))**

G = green hits; N = scored tokens; gamma = actual green fraction.
A larger positive z means more green hits than this simple null expects.

The displayed one-sided **p-value is an approximation**, not the probability the
text is AI-written or watermarked. Short samples are noisy. We need many held-out
unmarked samples to measure false-positive rates before claiming a reliable detector.

Next: choose a prompt and settings. The lab generates a watermark-OFF/ON pair with
the same sampling seed, saves both outputs, and plots their actual measurements.
The paired samples are controlled comparisons, not independent trials.
""",
]


class WatermarkLab(App):
    TITLE = "Watermark Lab"
    SUB_TITLE = "Learn the mechanism. Measure the evidence."
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("f1", "learn", "Learn")]
    CSS = """
    Screen { background: #090f1b; }
    Header { background: #142139; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }
    .intro { color: #50d9cb; margin-bottom: 1; }
    .card { background: #101d30; border: round #294562; padding: 1 2; margin-bottom: 1; }
    #lesson { min-height: 17; }
    .actions { height: 3; margin: 1 0; }
    Button { margin-right: 1; }
    #prompt { height: 6; margin-bottom: 1; }
    .settings-row { height: 6; }
    .field { width: 1fr; margin-right: 1; }
    .field Label { height: 1; }
    .field Input { width: 100%; }
    #status { height: auto; min-height: 2; color: #50d9cb; }
    #spinner { height: 1; }
    #measurements { height: 12; margin-bottom: 1; }
    PlotextPlot { height: 19; border: round #294562; margin-bottom: 1; }
    #probabilities { height: 15; margin-bottom: 1; }
    #sample-picker { margin-bottom: 1; }
    #sample-text { height: 12; margin-bottom: 1; }
    #token-evidence { height: auto; max-height: 12; overflow-y: auto; }
    #run-meta { height: auto; }
    """

    def __init__(self):
        super().__init__()
        self.lesson = 0
        self.runner = Runner()
        self.result = None
        self.run_path = None
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="learn", id="tabs"):
            with TabPane("01 Learn", id="learn"):
                with VerticalScroll():
                    yield Static("WATERMARK LAB  /  A GUIDED RESEARCH WORKSPACE", classes="intro")
                    yield Markdown(LESSONS[0], id="lesson", classes="card")
                    yield Static("Lesson 1 of 3", id="lesson-count")
                    with Horizontal(classes="actions"):
                        yield Button("Back", id="back", disabled=True)
                        yield Button("Next lesson", id="next", variant="primary")
                        yield Button("Open lab", id="open-lab")
            with TabPane("02 Experiment", id="experiment"):
                with VerticalScroll():
                    yield Markdown("## Green-list baseline\nChange a setting, run a paired comparison, inspect the evidence.", classes="card")
                    yield Label("Prompt / what should the model write about?")
                    yield TextArea(Settings().prompt, id="prompt")
                    with Horizontal(classes="settings-row"):
                        for name, label, value, tip in [
                            ("key", "Secret key", "42", "Selects the green lists. Detector must use the same key."),
                            ("bias", "Bias / strength", "1.5", "Added to green logits. 0 means no watermark boost; range 0 to 5."),
                            ("fraction", "Green fraction", "0.25", "Share of vocabulary selected; range 0.05 to 0.95."),
                        ]:
                            with VerticalScroll(classes="field"):
                                yield Label(label)
                                yield Input(value, id=name, tooltip=tip)
                    with Horizontal(classes="settings-row"):
                        for name, label, value, tip in [
                            ("temperature", "Temperature", "0.8", "Randomness of sampling; range 0.1 to 2."),
                            ("tokens", "Max tokens", "128", "Maximum generated tokens per sample; 16 to 512. May end early."),
                            ("seed", "Sampling seed", "123", "Controls generation randomness, separate from the watermark key."),
                            ("pairs", "OFF/ON pairs", "1", "1 to 10 pairs. Each pair uses the next sampling seed."),
                        ]:
                            with VerticalScroll(classes="field"):
                                yield Label(label)
                                yield Input(value, id=name, tooltip=tip)
                    yield Static("Hover over a setting for its meaning. First run loads the model; later runs reuse it.", classes="intro")
                    with Horizontal(classes="actions"):
                        yield Button("Run experiment", id="run", variant="success")
                    yield LoadingIndicator(id="spinner")
                    yield Static("Ready. No experiment has run yet.", id="status", markup=False)
            with TabPane("03 Evidence", id="evidence"):
                with VerticalScroll():
                    yield Static("No measured data yet. Run an experiment to fill these views.", id="run-meta", classes="card", markup=False)
                    yield DataTable(id="measurements")
                    yield PlotextPlot(id="z-chart")
                    yield Markdown("### The math\nz = (G − γN) / √(Nγ(1−γ)). Only unique previous-token contexts are scored. The one-sided normal-tail p-value is approximate and uncalibrated; it is not a watermark probability.")
                    yield PlotextPlot(id="prob-chart")
                    yield DataTable(id="probabilities")
                    yield Select([], prompt="Choose generated text", id="sample-picker")
                    yield TextArea(read_only=True, id="sample-text")
                    yield Static("Green highlighting will mark preferred tokens. Repeated contexts remain visible but are excluded from scoring.", id="token-evidence", classes="card")
        yield Footer()

    def on_mount(self):
        self.theme = "textual-dark"
        self.query_one("#spinner").display = False
        self.query_one("#measurements", DataTable).add_columns(
            "Pair", "Watermark", "Tokens", "Scored N", "Green G", "z", "Approx p", "tok/s",
        )
        self.query_one("#probabilities", DataTable).add_columns("Token", "Green?", "OFF probability", "ON probability")
        for chart_id, title in [("z-chart", "Detection trace / waiting for measured data"),
                                ("prob-chart", "First-token probabilities / waiting for measured data")]:
            chart = self.query_one(f"#{chart_id}", PlotextPlot).plt
            chart.theme("dark")
            chart.title(title)
        try:
            preferences = json.loads((ROOT / "data" / "preferences.json").read_text())
            if preferences.get("tutorial_completed"):
                self.call_after_refresh(self.open_lab)
        except (OSError, ValueError):
            pass

    def action_learn(self):
        self.set_focus(None)
        self.query_one("#tabs", TabbedContent).active = "learn"
        self.call_after_refresh(self.query_one("#next", Button).focus)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed):
        button_id = event.button.id
        if button_id == "back":
            self.lesson = max(0, self.lesson - 1)
        elif button_id == "next":
            if self.lesson == len(LESSONS) - 1:
                self.open_lab(completed=True)
                return
            self.lesson += 1
        elif button_id == "open-lab":
            self.open_lab(completed=False)
            return
        elif button_id == "run":
            self.start_run()
            return
        else:
            return
        self.query_one("#lesson", Markdown).update(LESSONS[self.lesson])
        self.query_one("#lesson-count", Static).update(f"Lesson {self.lesson + 1} of {len(LESSONS)}")
        self.query_one("#back", Button).disabled = self.lesson == 0
        self.query_one("#next", Button).label = "Enter the lab" if self.lesson == 2 else "Next lesson"

    def open_lab(self, completed=False):
        self.set_focus(None)
        self.query_one("#tabs", TabbedContent).active = "experiment"
        self.call_after_refresh(self.query_one("#prompt", TextArea).focus)
        if completed:
            try:
                path = ROOT / "data" / "preferences.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"tutorial_completed": True}), encoding="utf-8")
            except OSError as error:
                self.notify(f"Could not save tutorial preference: {error}", severity="warning")

    def start_run(self):
        if self.busy:
            return
        try:
            values = {name: self.query_one(f"#{name}", Input).value for name in
                      ("key", "bias", "fraction", "temperature", "tokens", "seed", "pairs")}
            settings = Settings(
                prompt=self.query_one("#prompt", TextArea).text,
                key=int(values["key"]), bias=float(values["bias"]), fraction=float(values["fraction"]),
                temperature=float(values["temperature"]), tokens=int(values["tokens"]),
                seed=int(values["seed"]), pairs=int(values["pairs"]),
            )
            settings.validate()
        except ValueError as error:
            self.notify(str(error), title="Check your settings", severity="error")
            return
        self.busy = True
        self.query_one("#run", Button).disabled = True
        self.query_one("#spinner").display = True
        self.run_experiment(settings)

    @work(thread=True, exclusive=True)
    def run_experiment(self, settings):
        try:
            result, path = self.runner.run(
                settings, lambda message: self.call_from_thread(self.update_status, message),
            )
            self.call_from_thread(self.show_result, result, path)
        except Exception as error:
            self.call_from_thread(self.update_status, f"Experiment failed: {error}")
        finally:
            self.call_from_thread(self.finish_run)

    def update_status(self, message):
        self.query_one("#status", Static).update(message)

    def finish_run(self):
        self.busy = False
        self.query_one("#run", Button).disabled = False
        self.query_one("#spinner").display = False

    def show_result(self, result, path):
        self.result, self.run_path = result, path
        self.query_one("#run-meta", Static).update(
            f"{result['method']} | {result['model']} | {result['gpu'] or result['device']}\n"
            f"{len(result['rows'])} measured samples | key {result['settings']['key']} | "
            f"bias {result['settings']['bias']} | seed {result['settings']['seed']}\n"
            f"Saved: {path}\nExploratory results; false-positive rate has not been calibrated."
        )
        table = self.query_one("#measurements", DataTable)
        table.clear()
        for row in result["rows"]:
            table.add_row(str(row["pair"]), "ON" if row["marked"] else "OFF",
                          str(row["generated_tokens"]), str(row["scored_tokens"]),
                          str(row["green_hits"]), f"{row['z']:.2f}" if row["z"] is not None else "n/a",
                          f"{row['approx_p']:.3g}" if row["approx_p"] is not None else "n/a",
                          f"{row['tokens_per_second']:.1f}")
        probabilities = self.query_one("#probabilities", DataTable)
        probabilities.clear()
        for item in result["first_step"]:
            probabilities.add_row(repr(item["token"]), "yes" if item["green"] else "no",
                                  f"{item['before']:.4%}", f"{item['after']:.4%}")
        prob_widget = self.query_one("#prob-chart", PlotextPlot)
        plot = prob_widget.plt
        plot.clear_figure()
        plot.theme("dark")
        items = result["first_step"][:8]
        plot.multiple_bar([str(i + 1) for i in range(len(items))],
                          [[item["before"] * 100 for item in items],
                           [item["after"] * 100 for item in items]],
                          labels=["OFF", "ON"], color=["cyan", "green"])
        plot.title("Same-prefix next-token probabilities / top candidates")
        plot.xlabel("Candidate rank (token names in table below)")
        plot.ylabel("Probability (%)")
        prob_widget.refresh()
        picker = self.query_one("#sample-picker", Select)
        picker.set_options([(f"Pair {row['pair']} / watermark {'ON' if row['marked'] else 'OFF'}", i)
                            for i, row in enumerate(result["rows"])])
        picker.value = 0
        self.show_sample(0)
        self.set_focus(None)
        self.query_one("#tabs", TabbedContent).active = "evidence"
        self.call_after_refresh(self.query_one("#measurements", DataTable).focus)

    @on(Select.Changed, "#sample-picker")
    def sample_changed(self, event: Select.Changed):
        if self.result is not None and isinstance(event.value, int):
            self.show_sample(event.value)

    def show_sample(self, index):
        row = self.result["rows"][index]
        self.query_one("#sample-text", TextArea).load_text(row["text"])
        evidence = Text()
        tokenizer = self.runner.tokenizer
        for token in row["trace"]:
            style = "green" if token["green"] else "white"
            if not token["scored"]:
                style += " dim"
            evidence.append(tokenizer.decode([token["token_id"]]), style=style)
        self.query_one("#token-evidence", Static).update(evidence)
        widget = self.query_one("#z-chart", PlotextPlot)
        plot = widget.plt
        plot.clear_figure()
        plot.theme("dark")
        for sample in self.result["rows"]:
            if sample["pair"] == row["pair"] and sample["trace"]:
                plot.plot([token["position"] for token in sample["trace"]],
                          [token["z"] for token in sample["trace"]],
                          label="ON" if sample["marked"] else "OFF",
                          color="green" if sample["marked"] else "cyan")
        plot.title(f"Pair {row['pair']} / cumulative detection evidence")
        plot.xlabel("Generated token position")
        plot.ylabel("z-score (approximate null)")
        widget.refresh()


if __name__ == "__main__":
    WatermarkLab().run()
