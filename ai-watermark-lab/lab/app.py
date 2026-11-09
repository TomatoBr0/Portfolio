"""Teach first, then explore measured watermark experiments in the terminal."""
import json

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
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
from .teaching import (
    PROBABILITY_HELP,
    SETTINGS_HELP,
    TABLE_HELP,
    TEXT_HELP,
    TRACE_HELP,
    explain_probabilities,
    explain_sample,
    explain_table,
    explain_trace,
)

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

**Pattern score = (green hits - expected hits) / expected fluctuation**

Expected hits = green fraction * scored tokens. Expected fluctuation is the
baseline's standard deviation: sqrt(scored tokens * green fraction * (1 - green fraction)).
The pattern score is conventionally called a **z-score**. Higher positive scores
mean more green hits than this simple chance baseline expects, not a likelihood percentage.

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
    Screen { background: #1e1e2e; color: #cdd6f4; }
    #masthead { height: 3; padding: 1 2 0 2; color: #b4befe; text-style: bold; }
    #workspace { height: 1fr; padding: 0 1; }
    #sidebar { width: 23; height: 1fr; border: ascii #45475a; padding: 1; margin-right: 1; }
    #sidebar .side-title { color: #a6adc8; margin-bottom: 1; }
    #sidebar Button { width: 100%; min-width: 0; text-align: left; margin: 0 0 1 0; }
    #sidebar Button.selected { background: #cdd6f4; color: #1e1e2e; }
    #sidebar .side-note { color: #a6adc8; margin-top: 1; height: auto; }
    TabbedContent { height: 1fr; width: 1fr; }
    TabPane { padding: 1; border: ascii #45475a; }
    Tabs { background: #1e1e2e; }
    Tab { background: #313244; color: #cdd6f4; padding: 0 1; }
    Tab.-active { background: #cdd6f4; color: #1e1e2e; text-style: bold; }
    Underline { display: none; }
    .intro { color: #a6adc8; margin-bottom: 1; }
    .card { background: #1e1e2e; border: ascii #45475a; padding: 1; margin-bottom: 1; }
    Markdown { background: #1e1e2e; color: #cdd6f4; }
    MarkdownH1, MarkdownH2, MarkdownH3 { background: #1e1e2e; color: #b4befe; border: none; }
    #lesson { min-height: 17; }
    .actions { height: 3; margin: 1 0; }
    #learn .actions { dock: bottom; background: #1e1e2e; margin: 0; height: 2; padding-top: 1; }
    Button { margin-right: 1; height: 1; min-width: 12; padding: 0 1; border: none; background: #313244; color: #cdd6f4; text-style: none; }
    Button:hover { background: #45475a; }
    Button:focus { background: #cdd6f4; color: #1e1e2e; text-style: bold; }
    Button.-primary, Button.-success { background: #b4befe; color: #1e1e2e; }
    Button:disabled { background: #313244; color: #6c7086; opacity: 1; }
    Input, TextArea, SelectCurrent { background: #1e1e2e; color: #cdd6f4; border: ascii #45475a; }
    Input:focus, TextArea:focus, SelectCurrent:focus { border: ascii #b4befe; }
    DataTable { background: #1e1e2e; }
    DataTable > .datatable--header { background: #313244; color: #b4befe; text-style: bold; }
    DataTable > .datatable--cursor { background: #cdd6f4; color: #1e1e2e; }
    Footer { background: #313244; color: #cdd6f4; }
    Footer > .footer--key { background: #45475a; color: #b4befe; }
    #prompt { height: 6; margin-bottom: 1; }
    .settings-row { height: 6; }
    .field { width: 1fr; margin-right: 1; }
    .field Label { height: 1; }
    .field Input { width: 100%; }
    #status { height: auto; min-height: 2; color: #a6e3a1; }
    #spinner { height: 1; }
    #measurements { height: 12; margin-bottom: 1; }
    PlotextPlot { height: 19; border: ascii #45475a; margin-bottom: 1; }
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

    def on_resize(self, event):
        sidebar = self.query("#sidebar")
        if sidebar:
            sidebar.first().display = event.size.width >= 100

    @on(TabbedContent.TabActivated, "#tabs")
    def navigation_changed(self, event: TabbedContent.TabActivated):
        for name in ("learn", "experiment", "evidence"):
            self.query_one(f"#nav-{name}", Button).set_class(event.pane.id == name, "selected")

    @staticmethod
    def style_plot(plot):
        plot.theme("dark")
        plot.canvas_color((30, 30, 46))
        plot.axes_color((30, 30, 46))
        plot.ticks_color((166, 173, 200))

    def compose(self) -> ComposeResult:
        yield Static("WATERMARK LAB  /  local research terminal", id="masthead")
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Static("WORKSPACE", classes="side-title")
                yield Button("[1] Learn", id="nav-learn")
                yield Button("[2] Experiment", id="nav-experiment")
                yield Button("[3] Evidence", id="nav-evidence")
                yield Static("-------------------\nMETHOD\ngreenlist-v1\n\nMODEL\nSmolLM2 / 360M\n\nDATA\nLocal measurements\n\nF1   Learn\n^Q   Quit", classes="side-note")
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
                        yield Markdown(SETTINGS_HELP, classes="card")
                        with Horizontal(classes="actions"):
                            yield Button("Run experiment", id="run", variant="success")
                        yield LoadingIndicator(id="spinner")
                        yield Static("Ready. No experiment has run yet.", id="status", markup=False)
                with TabPane("03 Evidence", id="evidence"):
                    with VerticalScroll():
                        yield Static("No measured data yet. Run an experiment to fill these views.", id="run-meta", classes="card", markup=False)
                        yield Markdown("## Your results, explained\nRun an experiment first. This section will use your actual token counts to explain the score.", id="result-guide", classes="card")
                        yield Markdown(TABLE_HELP, id="table-guide")
                        yield DataTable(id="measurements")
                        yield Markdown(TRACE_HELP, id="trace-guide")
                        yield PlotextPlot(id="z-chart")
                        yield Markdown("### The math, with names\n**Pattern score = (green hits - expected hits) / expected fluctuation.**\n\nExpected hits = scored tokens * green fraction. Expected fluctuation = sqrt(scored tokens * green fraction * (1 - green fraction)). This score is the standard **z-score**. It tells you distance from the baseline, not likelihood. The chance-tail value is an approximate **p-value**, not a watermark probability.")
                        yield Markdown(PROBABILITY_HELP, id="probability-guide")
                        yield PlotextPlot(id="prob-chart")
                        yield DataTable(id="probabilities")
                        yield Markdown(TEXT_HELP)
                        yield Select([], prompt="Choose generated text", id="sample-picker")
                        yield TextArea(read_only=True, id="sample-text")
                        yield Static("Green highlighting will mark preferred tokens. Repeated contexts remain visible but are excluded from scoring.", id="token-evidence", classes="card")
        yield Footer()

    def on_mount(self):
        self.theme = "catppuccin-mocha"
        self.query_one("#spinner").display = False
        self.query_one("#measurements", DataTable).add_columns(
            "Pair", "Watermark", "Tokens", "Scored tokens", "Green hits", "Pattern score", "Chance tail", "Tokens/sec",
        )
        self.query_one("#probabilities", DataTable).add_columns("#", "Token", "Green?", "OFF probability", "ON probability")
        for chart_id, title in [("z-chart", "Detection trace / waiting for measured data"),
                                ("prob-chart", "First-token probabilities / waiting for measured data")]:
            chart = self.query_one(f"#{chart_id}", PlotextPlot).plt
            self.style_plot(chart)
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
        self.query_one("#next", Button).focus()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed):
        button_id = event.button.id
        if button_id == "nav-learn":
            self.action_learn()
            return
        elif button_id == "nav-experiment":
            self.open_lab()
            return
        elif button_id == "nav-evidence":
            self.set_focus(None)
            self.query_one("#tabs", TabbedContent).active = "evidence"
            self.query_one("#measurements", DataTable).focus()
            return
        elif button_id == "back":
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
        self.query_one("#prompt", TextArea).focus()
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
        self.query_one("#table-guide", Markdown).update(explain_table(result))
        self.query_one("#probability-guide", Markdown).update(explain_probabilities(result))
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
        for rank, item in enumerate(result["first_step"], 1):
            probabilities.add_row(str(rank), repr(item["token"]), "yes" if item["green"] else "no",
                                  f"{item['before']:.4%}", f"{item['after']:.4%}")
        prob_widget = self.query_one("#prob-chart", PlotextPlot)
        plot = prob_widget.plt
        plot.clear_figure()
        self.style_plot(plot)
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
        evidence_view = self.query_one("#evidence", TabPane).query_one(VerticalScroll)
        evidence_view.focus()
        evidence_view.scroll_home(animate=False)

    @on(Select.Changed, "#sample-picker")
    def sample_changed(self, event: Select.Changed):
        if self.result is not None and isinstance(event.value, int):
            self.show_sample(event.value)

    def show_sample(self, index):
        row = self.result["rows"][index]
        self.query_one("#result-guide", Markdown).update(explain_sample(self.result, row))
        self.query_one("#trace-guide", Markdown).update(explain_trace(self.result, row))
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
        self.style_plot(plot)
        for sample in self.result["rows"]:
            if sample["pair"] == row["pair"] and sample["trace"]:
                plot.plot([token["position"] for token in sample["trace"]],
                          [token["z"] for token in sample["trace"]],
                          label="ON" if sample["marked"] else "OFF",
                          color="green" if sample["marked"] else "cyan")
        plot.title(f"Pair {row['pair']} / cumulative detection evidence")
        plot.xlabel("Generated token position")
        plot.ylabel("Pattern score (z)")
        widget.refresh()


if __name__ == "__main__":
    WatermarkLab().run()
