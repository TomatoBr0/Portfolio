"""A fixed-screen terminal workbench with paged controls and measured evidence."""
import json
import math
import threading
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    Select,
    Static,
    TextArea,
)
from textual_plotext import PlotextPlot

from .experiment import ROOT, Runner, Settings
from .suites import make_plan

PAGES = ['setup', 'tuning', 'run', 'summary', 'chart', 'probabilities', 'text']
TITLES = ['Choose experiment', 'Tune settings', 'Run / baseline setup', 'Measurements', 'Score graph', 'Token choices', 'Generated answers']


class WatermarkLab(App):
    TITLE = 'Watermark Lab'
    BINDINGS = [('ctrl+q', 'quit', 'Quit'), ('f1', 'setup', 'Setup')]
    CSS = '''
    Screen { background: #1e1e2e; color: #cdd6f4; overflow: hidden hidden; }
    #heading { height: 2; padding: 0 2; color: #b4befe; text-style: bold; }
    #body { height: 1fr; padding: 0 1; }
    #explanation { width: 30%; min-width: 23; border: ascii #45475a; padding: 1; margin-right: 1; overflow: hidden hidden; }
    #page-name { color: #b4befe; height: 2; text-style: bold; }
    #guide { height: 1fr; overflow: hidden hidden; }
    .guide-controls { height: 1; }
    .guide-controls Button { min-width: 3; width: 3; padding: 0; }
    #guide-count { width: 1fr; text-align: center; }
    #page-data { width: 1fr; height: 1fr; border: ascii #45475a; padding: 0 1; }
    ContentSwitcher { height: 1fr; }
    .page { height: 1fr; overflow: hidden hidden; }
    Label { height: 1; }
    Input, SelectCurrent, TextArea { background: #1e1e2e; border: ascii #45475a; color: #cdd6f4; }
    Input:focus, TextArea:focus { border: ascii #b4befe; }
    Input { height: 3; }
    Select { height: 3; }
    #prompt { height: 5; }
    .fields { height: 4; }
    .field { width: 1fr; margin-right: 1; }
    .button-row { height: 1; }
    Button { height: 1; border: none; min-width: 9; padding: 0 1; margin-right: 1; background: #313244; color: #cdd6f4; text-style: none; }
    Button:focus, Button.-primary { background: #cdd6f4; color: #1e1e2e; }
    Button:disabled { color: #6c7086; background: #313244; }
    #navigation { height: 2; padding: 0 1; }
    #step { width: 1fr; text-align: center; }
    #status { height: 2; padding: 0 2; color: #a6e3a1; overflow: hidden hidden; }
    #spinner { height: 1; }
    PlotextPlot { height: 1fr; }
    DataTable { height: 1fr; background: #1e1e2e; }
    DataTable > .datatable--header { background: #313244; color: #b4befe; }
    #probabilities-table { height: 7; }
    #run-plan { height: 1fr; overflow: hidden hidden; }
    #excerpt { height: 1fr; overflow: hidden hidden; }
    #excerpt-count { height: 1; color: #a6adc8; }
    Footer { background: #313244; }
    '''

    def __init__(self):
        super().__init__()
        self.page = 0
        self.runner = Runner()
        self.result = None
        self.run_path = None
        self.busy = False
        self.stop_event = threading.Event()
        self.sample = 0
        self.table_page = 0
        self.excerpt_page = 0
        self.guide_text = ''
        self.guide_page = 0

    def fields(self, items):
        with Horizontal(classes='fields'):
            for name, label, value in items:
                with Vertical(classes='field'):
                    yield Label(label)
                    yield Input(value, id=name)

    def compose(self) -> ComposeResult:
        yield Static('WATERMARK LAB / experiment workbench', id='heading')
        with Horizontal(id='body'):
            with Vertical(id='explanation'):
                yield Static('', id='page-name')
                yield Static('', id='guide', markup=False)
                with Horizontal(classes='guide-controls'):
                    yield Button('<', id='prev-guide')
                    yield Static('', id='guide-count')
                    yield Button('>', id='next-guide')
            with Vertical(id='page-data'):
                with ContentSwitcher(initial='setup', id='pages'):
                    with Vertical(id='setup', classes='page'):
                        yield Label('Experiment type')
                        yield Select([('Paired OFF / ON', 'paired'), ('Bulk baseline + held-out test', 'baseline'), ('Watermark strength sweep', 'strength')], value='paired', allow_blank=False, id='kind')
                        yield Label('Prompt / bulk also uses a small varied prompt bank')
                        yield TextArea(Settings().prompt, id='prompt')
                        yield from self.fields([('tokens', 'Max tokens', '128'), ('pairs', 'Pairs / sweep repeats', '1')])
                        yield Button('Load latest saved run', id='load')
                    with Vertical(id='tuning', classes='page'):
                        yield from self.fields([('key', 'Secret key', '42'), ('seed', 'Sampling seed', '123')])
                        yield from self.fields([('bias', 'Bias / strength', '1.5'), ('fraction', 'Green fraction', '0.25')])
                        yield from self.fields([('temperature', 'Temperature', '0.8')])
                    with Vertical(id='run', classes='page'):
                        yield from self.fields([('calibration_samples', 'Calibration OFF samples', '40'), ('evaluation_pairs', 'Fresh test pairs', '20')])
                        yield from self.fields([('target_false_alarm', 'Target false alarms (0.05 = 5%)', '0.05')])
                        yield Static('', id='run-plan', markup=False)
                        with Horizontal(classes='button-row'):
                            yield Button('Run experiment', id='run-button', variant='primary')
                            yield Button('Stop + save', id='stop', disabled=True)
                        yield LoadingIndicator(id='spinner')
                    with Vertical(id='summary', classes='page'):
                        yield DataTable(id='measurements', cell_padding=0)
                        with Horizontal(classes='button-row'):
                            yield Button('Prev rows', id='prev-rows')
                            yield Static('', id='row-count')
                            yield Button('Next rows', id='next-rows')
                    with Vertical(id='chart', classes='page'):
                        yield Select([], prompt='Select sample / pair', id='sample-picker')
                        yield PlotextPlot(id='score-chart')
                    with Vertical(id='probabilities', classes='page'):
                        yield PlotextPlot(id='prob-chart')
                        yield DataTable(id='probabilities-table', cell_padding=0)
                    with Vertical(id='text', classes='page'):
                        yield Select([], prompt='Select answer', id='text-picker')
                        yield Static('', id='excerpt', markup=False)
                        yield Static('', id='excerpt-count')
                        with Horizontal(classes='button-row'):
                            yield Button('Prev excerpt', id='prev-text')
                            yield Button('Next excerpt', id='next-text')
        yield Static('Ready. Choose an experiment; no lesson required.', id='status', markup=False)
        with Horizontal(id='navigation'):
            yield Button('< Back', id='back')
            yield Static('', id='step')
            yield Button('Next >', id='next', variant='primary')
        yield Footer()

    def on_mount(self):
        self.theme = 'catppuccin-mocha'
        self.query_one('#spinner').display = False
        self.query_one('#measurements', DataTable).add_columns('Sample', 'Set', 'Bias', 'Scored', 'Green', 'Score', 'tok/s')
        self.query_one('#probabilities-table', DataTable).add_columns('#', 'Token', 'Favored?', 'OFF %', 'ON %')
        self.update_page()

    def action_setup(self):
        self.page = 0
        self.update_page()

    def on_resize(self, event):
        if self.is_mounted:
            self.call_after_refresh(self.render_guide)
            if self.result:
                self.call_after_refresh(self.render_excerpt)
                self.call_after_refresh(self.render_table)

    def settings(self):
        values = {}
        for name in ['key', 'seed', 'tokens', 'pairs', 'calibration_samples', 'evaluation_pairs']:
            values[name] = int(self.query_one(f'#{name}', Input).value)
        for name in ['bias', 'fraction', 'temperature', 'target_false_alarm']:
            values[name] = float(self.query_one(f'#{name}', Input).value)
        settings = Settings(prompt=self.query_one('#prompt', TextArea).text, kind=self.query_one('#kind', Select).value, **values)
        settings.validate()
        return settings

    def update_page(self):
        self.set_focus(None)
        self.query_one('#pages', ContentSwitcher).current = PAGES[self.page]
        self.query_one('#page-name', Static).update(TITLES[self.page])
        self.query_one('#step', Static).update(f'{self.page + 1} / {len(PAGES)}')
        self.query_one('#back', Button).disabled = self.page == 0
        self.query_one('#next', Button).disabled = self.page == len(PAGES) - 1
        self.update_guide()
        if PAGES[self.page] == 'run':
            self.update_plan()
        target = {'setup': '#kind', 'tuning': '#key', 'run': '#run-button', 'summary': '#next', 'chart': '#sample-picker', 'probabilities': '#next', 'text': '#text-picker'}[PAGES[self.page]]
        self.query_one(target).focus()
        if self.result:
            self.call_after_refresh(self.render_excerpt)

    @on(Input.Changed)
    def settings_changed(self, event):
        if self.is_mounted and PAGES[self.page] == 'run':
            self.update_plan()

    def update_plan(self):
        try:
            settings = self.settings()
            count = len(make_plan(settings))
            text = f'{settings.kind.upper()} / {count} planned answers\nUp to {count * settings.tokens:,} generated tokens.\nSaved locally as JSON. Stop keeps completed samples.'
            if settings.kind != 'baseline':
                text += '\nCalibration controls above apply only to Bulk baseline.'
            self.query_one('#run-plan', Static).update(text)
        except ValueError as error:
            self.query_one('#run-plan', Static).update(str(error))

    @on(Button.Pressed)
    def button_pressed(self, event):
        name = event.button.id
        if name in ('back', 'next'):
            self.page = max(0, min(len(PAGES) - 1, self.page + (1 if name == 'next' else -1)))
            self.update_page()
        elif name == 'run-button':
            self.start_run()
        elif name == 'stop':
            self.stop_event.set()
            self.update_status('Stopping at the next token; completed answers will be saved.')
        elif name in ('prev-rows', 'next-rows'):
            self.table_page = max(0, self.table_page + (1 if name == 'next-rows' else -1))
            self.render_table()
        elif name in ('prev-text', 'next-text'):
            self.excerpt_page = max(0, self.excerpt_page + (1 if name == 'next-text' else -1))
            self.render_excerpt()
        elif name in ('prev-guide', 'next-guide'):
            self.guide_page = max(0, self.guide_page + (1 if name == 'next-guide' else -1))
            self.render_guide()
        elif name == 'load':
            try:
                paths = sorted((ROOT / 'outputs').glob('*.json'), key=lambda p: p.stat().st_mtime)
                if not paths:
                    raise ValueError('No saved experiment found yet.')
                path = paths[-1]
                self.show_result(json.loads(path.read_text(encoding='utf-8')), path)
                self.update_status(f'Loaded {path.name}')
            except (OSError, ValueError, KeyError) as error:
                self.notify(str(error), severity='error')

    def start_run(self):
        if self.busy:
            return
        try:
            settings = self.settings()
        except ValueError as error:
            self.notify(str(error), title='Check settings', severity='error')
            return
        self.busy = True
        self.stop_event.clear()
        self.query_one('#run-button', Button).disabled = True
        self.query_one('#stop', Button).disabled = False
        self.query_one('#spinner').display = True
        self.run_experiment(settings)

    @work(thread=True, exclusive=True)
    def run_experiment(self, settings):
        try:
            result, path = self.runner.run(settings, lambda msg: self.call_from_thread(self.update_status, msg), self.stop_event.is_set)
            self.call_from_thread(self.show_result, result, path)
        except Exception as error:
            self.call_from_thread(self.update_status, f'Failed: {error}')
        finally:
            self.call_from_thread(self.finish_run)

    def update_status(self, text):
        self.query_one('#status', Static).update(text)

    def finish_run(self):
        self.busy = False
        self.query_one('#run-button', Button).disabled = False
        self.query_one('#stop', Button).disabled = True
        self.query_one('#spinner').display = False

    def show_result(self, result, path):
        self.result, self.run_path = result, Path(path)
        self.sample = self.table_page = self.excerpt_page = 0
        self.query_one('#excerpt', Static).update('No completed answer yet.')
        options = [(f"{i + 1} / {row.get('phase', 'paired')} / {'ON' if row['marked'] else 'OFF'} / bias {row.get('bias', result['settings']['bias'] if row['marked'] else 0)}", i) for i, row in enumerate(result['rows'])]
        for selector in ['#sample-picker', '#text-picker']:
            picker = self.query_one(selector, Select)
            picker.set_options(options)
            if options:
                picker.value = 0
        self.render_table()
        self.render_scores()
        self.render_probabilities()
        self.page = PAGES.index('summary')
        self.update_page()

    @on(Select.Changed)
    def selected(self, event):
        if event.select.id in ('sample-picker', 'text-picker') and isinstance(event.value, int) and self.result:
            if self.sample != event.value:
                self.sample = event.value
                self.excerpt_page = 0
                for selector in ['#sample-picker', '#text-picker']:
                    self.query_one(selector, Select).value = self.sample
            self.render_scores()
            self.render_excerpt()
            self.update_guide()

    def render_table(self):
        rows = self.result['rows'] if self.result else []
        # Keep rows on a page small enough for the current terminal height.
        count = max(1, self.size.height - 14)
        pages = max(1, math.ceil(len(rows) / count))
        self.table_page = min(self.table_page, pages - 1)
        table = self.query_one('#measurements', DataTable)
        table.clear()
        start = self.table_page * count
        for i, row in enumerate(rows[start:start + count], start + 1):
            table.add_row(str(i), row.get('phase', 'paired')[:5] + (' ON' if row['marked'] else ' OFF'), str(row.get('bias', self.result['settings']['bias'] if row['marked'] else 0)), str(row['scored_tokens']), str(row['green_hits']), f"{row['z']:.2f}" if row['z'] is not None else 'n/a', f"{row['tokens_per_second']:.0f}")
        self.query_one('#row-count', Static).update(f' {self.table_page + 1}/{pages} ')
        self.query_one('#prev-rows', Button).disabled = self.table_page == 0
        self.query_one('#next-rows', Button).disabled = self.table_page == pages - 1

    @staticmethod
    def style_plot(plot):
        plot.clear_figure()
        plot.theme('dark')
        plot.canvas_color((30, 30, 46))
        plot.axes_color((30, 30, 46))
        plot.ticks_color((166, 173, 200))

    def render_scores(self):
        widget = self.query_one('#score-chart', PlotextPlot)
        plot = widget.plt
        self.style_plot(plot)
        rows = self.result['rows']
        kind = self.result.get('kind', 'paired')
        if kind == 'baseline':
            for phase, marked, label, color in [('calibration', False, 'Calibration OFF', 'gray'), ('evaluation', False, 'Test OFF', 'cyan'), ('evaluation', True, 'Test ON', 'green')]:
                points = [(i + 1, row['z']) for i, row in enumerate(rows) if row.get('phase') == phase and row['marked'] == marked and row['z'] is not None]
                if points:
                    plot.scatter([p[0] for p in points], [p[1] for p in points], label=label, color=color, marker='dot')
            threshold = self.result.get('baseline', {}).get('threshold')
            if threshold is not None:
                plot.hline(threshold, color='yellow')
            plot.title('Bulk scores / yellow = frozen threshold')
            plot.xlabel('Recorded sample number')
        elif kind == 'strength':
            groups = [g for g in self.result.get('strength', []) if g['mean_score'] is not None]
            if groups:
                plot.plot([g['bias'] for g in groups], [g['mean_score'] for g in groups], marker='dot', color='green')
            plot.title('Watermark strength / measured mean score')
            plot.xlabel('Bias applied to favored tokens')
        elif rows:
            selected = rows[min(self.sample, len(rows) - 1)]
            for row in rows:
                if row['pair'] == selected['pair'] and row['trace']:
                    plot.plot([p['position'] for p in row['trace']], [p['z'] for p in row['trace']], label='ON' if row['marked'] else 'OFF', color='green' if row['marked'] else 'cyan')
            plot.title(f"Pair {selected['pair']} / accumulated evidence")
            plot.xlabel('Generated token position')
        plot.ylabel('Pattern score')
        widget.refresh()

    def render_probabilities(self):
        table = self.query_one('#probabilities-table', DataTable)
        table.clear()
        widget = self.query_one('#prob-chart', PlotextPlot)
        plot = widget.plt
        self.style_plot(plot)
        items = self.result.get('first_step', [])[:4]
        if items:
            plot.multiple_bar([str(i + 1) for i in range(len(items))], [[r['before'] * 100 for r in items], [r['after'] * 100 for r in items]], labels=['OFF', 'ON'], color=['cyan', 'green'])
            for i, item in enumerate(items, 1):
                table.add_row(str(i), repr(item['token']), 'yes' if item['green'] else 'no', f"{item['before']*100:.2f}", f"{item['after']*100:.2f}")
        source = self.result.get('probability_sample')
        plot.title(f"Sample {source['row_index'] + 1} / bias {source['bias']:g} / same prefix" if source else 'First marked sample / same-prefix choices')
        plot.ylabel('Probability (%)')
        widget.refresh()

    def render_excerpt(self):
        if not self.result or not self.result['rows']:
            return
        row = self.result['rows'][min(self.sample, len(self.result['rows']) - 1)]
        widget = self.query_one('#excerpt', Static)
        width = max(10, widget.content_size.width)
        height = max(1, widget.content_size.height)
        text = Text(row['text'])
        if self.runner.tokenizer and row.get('trace'):
            text = Text()
            for token in row['trace']:
                text.append(self.runner.tokenizer.decode([token['token_id']]), style=('green' if token['green'] else '#cdd6f4') + (' dim' if not token['scored'] else ''))
        lines = text.wrap(self.console, width)
        pages = max(1, math.ceil(len(lines) / height))
        self.excerpt_page = min(self.excerpt_page, pages - 1)
        widget.update(Text('\n').join(lines[self.excerpt_page * height:(self.excerpt_page + 1) * height]))
        self.query_one('#excerpt-count', Static).update(f'Excerpt {self.excerpt_page + 1}/{pages} | {row["generated_tokens"]} generated tokens')
        self.query_one('#prev-text', Button).disabled = self.excerpt_page == 0
        self.query_one('#next-text', Button).disabled = self.excerpt_page == pages - 1

    def update_guide(self):
        name = PAGES[self.page]
        guides = {
            'setup': 'Choose what to test.\n\nPaired: compare ordinary and watermarked answers.\n\nBulk: build a baseline, then measure mistakes on fresh answers.\n\nStrength: test several bias levels.\n\nNext opens the settings.',
            'tuning': 'Key chooses favored tokens. Seed controls sampling randomness.\n\nBias boosts favored choices; 0 turns the boost off.\n\nGreen fraction 0.25 favors 25% of vocabulary.\n\nTemperature controls randomness. These settings stay fixed during a baseline run.',
            'run': 'Bulk first generates calibration OFF answers. Only those scores set the threshold.\n\nFresh OFF/ON test pairs use different seeds. OFF flags are false alarms; ON flags are detections.\n\nA 5% target is a calibration setting, not promised accuracy.\n\nStop saves completed answers.',
            'summary': 'Run an experiment to populate this table.\n\nScored = eligible text tokens.\nGreen = favored-token hits.\nScore = excess green hits relative to the simple chance baseline.\n\nUse Prev/Next rows for more samples.',
            'chart': 'Run an experiment first. The graph will use actual measured scores.',
            'probabilities': 'The model predicts possible next tokens. The key colors some as favored, then bias nudges their probabilities.\n\nCyan = OFF. Green = ON. The numbered table matches the bars.\n\nThis view uses one shared prefix, not the whole answer.',
            'text': 'Choose an actual generated answer.\n\nUse Prev/Next excerpt to read it without scrolling.\n\nGreen = favored tokens. Dim = repeated contexts excluded from scoring. Saved runs loaded without a tokenizer show plain text.',
        }
        text = guides[name]
        result = self.result
        if result:
            rows = result['rows']
            if name == 'summary':
                text = self.result_summary()
            elif name == 'chart':
                if result.get('kind') == 'baseline':
                    b = result['baseline']
                    text = f"{b['calibration_count']} calibration OFF scores set the yellow threshold.\n\nGray = calibration. Cyan = fresh OFF. Green = fresh ON.\n\nAbove the line is flagged. A cyan point above it is a false alarm.\n\nHorizontal = sample order. Vertical = pattern score.\n\nThis applies to this prompt bank and settings."
                elif result.get('kind') == 'strength':
                    groups = [g for g in result.get('strength', []) if g['mean_score'] is not None]
                    text = 'Horizontal = watermark bias. Vertical = mean pattern score.\n\n' + '\n'.join(f"Bias {g['bias']:g}: score {g['mean_score']:.2f} ({g['samples']} samples)" for g in groups) + '\n\nRepeats share seeds across strengths. This is not a detection accuracy curve.'
                elif rows:
                    row = rows[self.sample]
                    text = f"Pair {row['pair']}\n\nHorizontal = token position. Vertical = pattern score so far.\n\nCyan = OFF; green = ON.\n\nSelected answer: {row['green_hits']} green hits / {row['scored_tokens']} scored tokens.\n\nFinal score: {row['z']:.2f}" if row['z'] is not None else 'No eligible text tokens to score.'
                    text += '\n\nHigher means more of our pattern, not an AI-likelihood percentage.'
            elif name == 'probabilities' and result.get('first_step'):
                item = result['first_step'][0]
                text = f"Group 1 is {item['token']!r}.\n\nOFF probability: {item['before']:.2%}\nON probability: {item['after']:.2%}\n\nKey + previous token choose the favored list. Bias nudges those choices.\n\nThese bars use the first marked sample's prefix. Only four candidates are shown.\n\nLow uncertainty means fewer plausible choices; forcing a preference can affect wording."
                source = result.get('probability_sample')
                if source:
                    text = text.replace("first marked sample's prefix", f"prefix of sample {source['row_index'] + 1} at bias {source['bias']:g}")
                if rows:
                    entropy = rows[source['row_index']].get('first_token_entropy_before') if source else next((r.get('first_token_entropy_before') for r in rows if r['marked']), None)
                    if entropy is not None:
                        text += f'\n\nFirst-step uncertainty: {entropy:.2f} bits.'
        if text != self.guide_text:
            self.guide_page = 0
        self.guide_text = text
        self.call_after_refresh(self.render_guide)

    def render_guide(self):
        widget = self.query_one('#guide', Static)
        lines = Text(self.guide_text).wrap(self.console, max(10, widget.content_size.width))
        height = max(1, widget.content_size.height)
        pages = max(1, math.ceil(len(lines) / height))
        self.guide_page = min(self.guide_page, pages - 1)
        widget.update(Text('\n').join(lines[self.guide_page * height:(self.guide_page + 1) * height]))
        self.query_one('#guide-count', Static).update(f'{self.guide_page + 1}/{pages}')
        self.query_one('#prev-guide', Button).disabled = self.guide_page == 0
        self.query_one('#next-guide', Button).disabled = self.guide_page == pages - 1

    def result_summary(self):
        result = self.result
        text = f"{result.get('status', 'complete').upper()}\n{len(result['rows'])}/{result.get('planned_samples', len(result['rows']))} answers saved.\n\n"
        if result.get('kind') == 'baseline':
            b = result['baseline']
            threshold = b['threshold']
            if threshold is None:
                return text + 'Not enough valid calibration samples for a finite threshold at this target. Increase calibration count.\n\nNo detection rates reported.'
            text += f'Threshold: score > {threshold:.2f}\nCalibration OFF: {b["calibration_count"]}\n\n'
            for key, label in [('false_alarms', 'False alarms'), ('detection', 'ON detected')]:
                rate = b[key]
                if rate['rate'] is None:
                    text += f'{label}: no scored test samples\n\n'
                else:
                    lo, hi = rate['interval95']
                    text += f'{label}: {rate["count"]}/{rate["total"]} ({rate["rate"]:.1%})\n95% interval: {lo:.0%}-{hi:.0%}\n\n'
            text += 'Small tests have wide uncertainty. These rates do not apply to other models or settings.'
        else:
            text += 'Scored = eligible tokens.\nGreen = favored hits.\nScore = excess green hits.\n\nOFF is ordinary AI output; ON applies a watermark boost.\n\nUse Next for graphs.\n\n' + self.run_path.name
        return text


if __name__ == '__main__':
    WatermarkLab().run()
