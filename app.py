"""SSR Engine: a stateless scorer for semantic similarity rating.

What this file is careful about, because the Space is public and its callers do not know each other:

  * no component that takes or returns a file, no session state, no cached examples;
  * run history, usage analytics, the monitoring page and telemetry are switched off;
  * handlers log an operation's name and duration and nothing a caller sent;
  * a failure inside the engine or a page handler is reported as one fixed sentence, so a library's
    own message (which may quote the text it failed on) reaches neither the caller's screen nor the logs.

Two things the web framework does on its own are outside that: its checks on a malformed, hand-built
request can print the value they rejected, and its upload route exists even though nothing here uses
it (capped at 1 kB, never read, purged within 30 minutes).

README.md says what that does and does not protect against.
"""
from __future__ import annotations

import spaces  # noqa: F401, I001 - first, before anything imports torch: on a ZeroGPU Space this is what makes the GPU usable

import functools
import json
import logging
import os
import re
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402

from ssr_engine import LIMITS, VERSION, EngineError, anchors, encoder, service  # noqa: E402

log = logging.getLogger("ssr_engine")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("uvicorn.access").disabled = True  # request lines carry session ids in their query strings
    logging.getLogger("httpx").setLevel(logging.WARNING)


_LABELLED = re.compile(r"^\s*([1-5])\s*[|\t]\s*(\S.*)$")

EXAMPLE_REACTIONS = "\n".join([
    "Not for me. I already have something that does this and I see no reason to switch.",
    "Maybe. I would need to see what it costs before I think about it any further.",
    "Yes, this solves a problem I have right now. I would sign up this week.",
])

EXAMPLE_LABELLED = "\n".join([
    "1 | No. I have no use for this and I would not look at it again.",
    "1 | I would never pay for this. It is of no interest to me.",
    "2 | Probably not. It looks fine, but it is not something I need.",
    "2 | I doubt I would bother. It does not seem worth the effort.",
    "3 | I could go either way. I would have to think about it.",
    "3 | Not sure. Some of it appeals to me and some of it does not.",
    "4 | I like this and would probably try it.",
    "4 | This looks useful. I expect I would go ahead.",
    "5 | Absolutely. I want this and would start today.",
    "5 | Yes, without question. This is exactly what I have been looking for.",
])

ABOUT = f"""
### What this is

Semantic similarity rating (SSR) turns a free-text reaction into a distribution over a five-point
scale. Each reaction is embedded and compared with anchor statements written for each scale point
(Maier et al., arXiv:2510.08338). Two mappings are offered so they can be compared: **paper**, as the
authors' reference code computes it, and **power**, a simpler mean-cosine variant.

### What happens to what you send

- Reactions and anchors are held in memory for the one request and then dropped.
- The app writes nothing you send to disk, a database or a log, and no result repeats your text.
- Nothing is cached per caller. The only cached vectors are those of the built-in anchors.
- Run history, analytics and the monitoring page are off. No part of the page takes a file.

### What that does not cover

- Hugging Face hosts this Space and can see traffic to it, as any host can.
- This is a shared public instance. For text you would not put on someone else's server, use
  **Duplicate this Space** (the copy is private by default and runs on your own hardware), or run
  the code yourself. The source is in this repository; what it does is checkable.

### Limits

Up to {LIMITS['texts']} reactions per request, {LIMITS['text_chars']} characters each; up to
{LIMITS['anchor_sets']} anchor sets of five statements; up to {LIMITS['labelled']} labelled reactions
for calibration. The encoder runs in float32 on a shared CPU, one request at a time.

Engine {VERSION}.
"""


def _run(name: str, operation, request):
    started = time.monotonic()
    try:
        result = operation(request)
    except EngineError as exc:
        log.info("%s refused", name)
        raise gr.Error(str(exc), print_exception=False) from None
    except Exception as exc:  # noqa: BLE001 - the message is dropped on purpose; see the module docstring
        log.warning("%s failed (%s)", name, type(exc).__name__)
        raise gr.Error("The engine could not complete this request.", print_exception=False) from None
    log.info("%s ok ms=%d", name, (time.monotonic() - started) * 1000)
    return result


def score(request: dict) -> dict:
    """Score reactions. Request: {"texts": [...], "anchors": [[five statements], ...] (optional),
    "mappings": ["paper", "power"] (optional), "temperature": {"paper": 1.0, "power": 1.0} (optional)}.
    Returns a distribution and expected score per reaction, by position; texts are never returned."""
    return _run("score", service.score, request)


def calibrate(request: dict) -> dict:
    """Fit each mapping's temperature on labelled reactions. Request: {"rows": [{"text": ..., "label": 1-5,
    "split": "calibration" | "holdout" (optional)}, ...], "anchors": ... (optional), "threshold": 1-5 (optional)}.
    Returns fitted temperatures, holdout metrics and each row's cosines, by position."""
    return _run("calibrate", service.calibrate, request)


def engine() -> dict:
    """What is running: the encoder, its precision, the mappings and the request limits. Takes nothing;
    call it before sending text if you need to be sure which encoder will read it."""
    return _run("engine", lambda _: service.about(), None)


def status() -> str:
    running = service.about()
    if not running["meaningful"]:
        return ("> **Stand-in encoder.** This instance is set to the offline stand-in, which matches wording, not meaning. "
                "Its scores are for trying the interface only.")
    return f"Encoder: `{running['encoder']}`, {running['dtype']}, on this Space's own hardware."


def _custom_anchors(text: str):
    if not (text or "").strip():
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise gr.Error("Your anchors are not valid JSON. Use a list of sets, each a list of five statements.", print_exception=False) from None


def _page(handler):
    """A handler behind the page. Whatever it raises that is not already a message for the caller becomes the fixed sentence."""
    @functools.wraps(handler)
    def run(*arguments):
        try:
            return handler(*arguments)
        except gr.Error:
            raise
        except Exception as exc:  # noqa: BLE001 - the message is dropped on purpose; see the module docstring
            log.warning("page failed (%s)", type(exc).__name__)
            raise gr.Error("The engine could not complete this request.", print_exception=False) from None
    return run


@_page
def ui_score(reactions: str, mappings: list[str], t_paper: float, t_power: float, custom: str):
    texts = [line.strip() for line in (reactions or "").splitlines() if line.strip()]
    if not mappings:
        raise gr.Error("Choose at least one mapping.", print_exception=False)
    request = {"texts": texts, "mappings": mappings, "temperature": {"paper": t_paper, "power": t_power}}
    request["temperature"] = {m: request["temperature"][m] for m in mappings}
    anchor_sets = _custom_anchors(custom)
    if anchor_sets is not None:
        request["anchors"] = anchor_sets
    result = score(request)
    table = [[row["index"] + 1, m, row[m]["expected_score"], *row[m]["pmf"]] for row in result["results"] for m in mappings]
    return table, result


@_page
def ui_calibrate(labelled: str, threshold: float | None, custom: str):
    rows = []
    for number, line in enumerate((labelled or "").splitlines(), start=1):
        if not line.strip():
            continue
        match = _LABELLED.match(line)
        if not match:
            raise gr.Error(f"Line {number}: start with the intended point (1 to 5), then |, then the reaction.", print_exception=False)
        rows.append({"label": int(match.group(1)), "text": match.group(2)})
    request = {"rows": rows}
    if threshold is not None:
        request["threshold"] = threshold
    anchor_sets = _custom_anchors(custom)
    if anchor_sets is not None:
        request["anchors"] = anchor_sets
    result = calibrate(request)
    table = [[r["setting"], r["temperature"], r["mae"], r["spearman"], r["argmax_accuracy"], r["spread"], r["min"], r["max"]] for r in result["holdout"]]
    return table, result


ANCHOR_HELP = "Optional. JSON: a list of sets, each a list of five statements from 1 (lowest) to 5 (highest). Leave empty for the built-in anchors."

with gr.Blocks(title="SSR Engine", analytics_enabled=False, delete_cache=(1800, 1800)) as demo:
    gr.Markdown("# SSR Engine\nFree-text reactions in, a five-point distribution out. Stateless: nothing you send is kept.")
    running = gr.Markdown()
    with gr.Tab("Score"):
        reactions = gr.Textbox(label="Reactions, one per line", lines=8, value=EXAMPLE_REACTIONS)
        with gr.Row():
            mappings = gr.CheckboxGroup(["paper", "power"], value=["paper"], label="Mapping")
            t_paper = gr.Number(1.0, label="Temperature, paper")      # no minimum here: the engine checks it, and its refusal quotes nothing
            t_power = gr.Number(1.0, label="Temperature, power")
        with gr.Accordion("Your own anchors", open=False):
            score_anchors = gr.Textbox(label="Anchors", info=ANCHOR_HELP, lines=6, placeholder=json.dumps(anchors.DEFAULT[:1]))
        score_button = gr.Button("Score", variant="primary")
        score_table = gr.Dataframe(headers=["#", "mapping", "expected", "p(1)", "p(2)", "p(3)", "p(4)", "p(5)"], interactive=False, label="Per reaction")
        score_json = gr.JSON(label="Full result", open=False)
        score_button.click(ui_score, [reactions, mappings, t_paper, t_power, score_anchors], [score_table, score_json], api_visibility="private")
    with gr.Tab("Calibrate"):
        gr.Markdown("Reactions whose intended point you know, one per line as `point | reaction`. Half fit each mapping's temperature; "
                    "the figures below come from the other half. Fit again whenever you change the encoder or the anchors.")
        labelled = gr.Textbox(label="Labelled reactions", lines=12, value=EXAMPLE_LABELLED)
        threshold = gr.Number(None, label="Pass threshold (optional, 1 to 5)")
        with gr.Accordion("Your own anchors", open=False):
            calibrate_anchors = gr.Textbox(label="Anchors", info=ANCHOR_HELP, lines=6)
        calibrate_button = gr.Button("Calibrate", variant="primary")
        calibrate_table = gr.Dataframe(headers=["setting", "T", "mean abs. error", "rank corr.", "top-point accuracy", "spread 5-1", "min", "max"],
                                       interactive=False, label="Holdout half")
        calibrate_json = gr.JSON(label="Full result", open=False)
        calibrate_button.click(ui_calibrate, [labelled, threshold, calibrate_anchors], [calibrate_table, calibrate_json], api_visibility="private")
    with gr.Tab("About and privacy"):
        gr.Markdown(ABOUT)

    demo.load(status, None, running, api_visibility="private")
    gr.api(engine, api_name="engine")
    gr.api(score, api_name="score")
    gr.api(calibrate, api_name="calibrate")

demo.queue(max_size=24, default_concurrency_limit=1, api_open=False)

LAUNCH = dict(show_error=True, run_history=False, enable_monitoring=False, mcp_server=False, ssr_mode=False,
              max_file_size="1kb", footer_links=["api"], state_session_capacity=64)

if __name__ == "__main__":
    configure_logging()
    encoder.get()            # load the weights before the first caller arrives
    demo.launch(**LAUNCH)
