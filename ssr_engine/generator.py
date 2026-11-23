"""Prompts to text: an open-weights chat model on the Space's own hardware. Off unless the deployment turns it on.

The public Space scores and does nothing else. A deployment enables a text model in one of two
ways, and a request never can:

    SSR_GENERATOR_MODEL    a model id, as a Space variable; it wins when both are present
    engine_config.json     {"generator_model": "<model id>"}, in the directory app.py is in

With neither, `load()` returns at once: no library is imported and no weight is read.

The model. Written for openai/gpt-oss-20b, a reasoning model that speaks the "harmony" format: it
answers on channels, first `analysis` (its reasoning) and then `final` (the answer). The tokenizer's
own chat template builds the prompt, asked for low reasoning effort. Its tokenizer ships no response
parser (the config has no `response_template`), so `final_text` reads the channel markers: it keeps
the final channel and drops the rest.

The weights. The model is published with its expert layers in MXFP4, about 14 GB. Running them as
published needs the `kernels` and `accelerate` packages and Triton kernels that are fetched from the
Hub, ask the GPU what it is while the weights load, and are compiled on it at first use. On ZeroGPU
no GPU is attached while the app starts. So the expert layers are de-quantised to bfloat16 as they
load, which Transformers offers by name (`Mxfp4Config(dequantize=True)`) and would do unasked if a
kernel package were missing or the wrong version: plain tensors, plain PyTorch, nothing fetched or
compiled. The price is memory, about 42 GB of a 48 GB slice, so a request is run in pieces sized to
what is left, and halved again if the GPU still runs out.

Hardware. As for the encoder: the model is loaded for the GPU at start-up, and the one function
that runs it, `_generate`, is handed to the platform with `@spaces.GPU`. It takes a whole request in
one stay on the GPU; plain lists go in, and plain strings and whole numbers come back. Off ZeroGPU
the `spaces` package does nothing.

Nothing here keeps a prompt or a text after the call that asked for it returns, and nothing here
logs one: log lines carry counts and durations.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import spaces        # before torch is imported anywhere: on ZeroGPU this is what lets a model be placed on the GPU at start-up

log = logging.getLogger("ssr_engine")

VARIABLE = "SSR_GENERATOR_MODEL"
CONFIG_FILE = "engine_config.json"
ROOT = Path(__file__).resolve().parents[1]       # the application's root: the directory app.py is in
REASONING = "low"                                # the template's other levels are medium and high; short texts need little thought

# The model's markers all look like <|name|>. One in a prompt would be read as the marker itself, so prompts carrying one are refused.
MARKER = re.compile(r"<\|[^<>|\s]{1,40}\|>")
_CHANNEL, _MESSAGE = "<|channel|>", "<|message|>"
_STOP = re.compile(r"<\|(?:return|end|call|start|channel|message|endoftext)\|>")

# Seconds. None of these has been measured on the Space: each request's log line gives the figures to set them from.
COLD_START = 40      # a GPU process that is new loads every weight on the Space before the function starts: about 42 GB here
PREFILL = 3          # reading one piece's prompts
STEP = 0.06          # one new token for a piece of one row,
STEP_PER_ROW = 0.0015    # and this much more for each further row
RESERVE = 2          # kept back at the end, to decode and hand the texts over
LONGEST = 120        # never ask the platform for more than this

# Bytes on the GPU while a piece runs. The attention weights are built in full (this model has no fused attention), three copies deep at worst.
ATTENTION = 400      # per row, per prompt token squared
EXPERTS = 240_000    # per row, per prompt token, while the expert layers read the prompt
CACHE = 50_000       # per row, per token, kept until the piece is done: keys and values in every layer
HEADROOM = 4 * 10**9     # what a 48 GB slice has left beside the weights; on the GPU the real figure is measured instead
SHARE = 0.8          # of the memory measured free, how much a piece may plan to use


def configured(root=None, environ=None) -> str | None:
    """The text model this deployment enables, or None. Set by the Space's variable or its file, never by a request."""
    environ = os.environ if environ is None else environ
    named = str(environ.get(VARIABLE) or "").strip()
    if named:
        return named                             # the variable wins
    path = Path(ROOT if root is None else root) / CONFIG_FILE
    if not path.is_file():
        return None
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise ValueError(f"{CONFIG_FILE} is not valid JSON.") from None
    if not isinstance(settings, dict):
        raise ValueError(f"{CONFIG_FILE} must be a JSON object.")
    if "generator_model" not in settings:
        return None                              # a file that says nothing about a text model enables none
    named = settings["generator_model"]
    if not isinstance(named, str) or not named.strip():
        raise ValueError(f"generator_model in {CONFIG_FILE} must be a model id.")
    return named.strip()


def final_text(raw: str) -> tuple[str, bool]:
    """The answer in one row of the model's output, and whether the model finished it.

    A row is a run of messages, each `<|channel|>NAME<|message|>BODY` up to an end marker. The answer
    is the body of the last message on the `final` channel; reasoning (`analysis`) and tool calls
    (`commentary`) are dropped, and so is every marker. A row that stops inside its reasoning has no
    answer; one that stops inside its answer has an unfinished one, which is returned as such.
    """
    text, finished = "", False
    at = raw.find(_CHANNEL)
    while at != -1:
        body = raw.find(_MESSAGE, at)
        if body == -1:
            break                                # cut off inside a header
        following = raw.find(_CHANNEL, at + len(_CHANNEL))
        if following != -1 and following < body:
            at = following                       # a header that never reached its message; the next is read in its turn
            continue
        name = re.match(r"\s*([A-Za-z_]*)", raw[at + len(_CHANNEL):body]).group(1)
        start = body + len(_MESSAGE)
        stop = _STOP.search(raw, start)
        end = stop.start() if stop else len(raw)
        if name == "final":
            text, finished = raw[start:end], stop is not None
        at = raw.find(_CHANNEL, end)
    return MARKER.sub("", text).strip(), finished


def _cost(rows: int, longest: int, new_tokens: int) -> int:
    return rows * (max(ATTENTION * longest * longest, EXPERTS * longest) + CACHE * (longest + new_tokens))


def _plan(lengths, new_tokens: int, budget: int) -> list[list[int]]:
    """Row numbers, in order, in as few equal pieces as fit the memory. A row that fits nowhere still gets a piece of its own."""
    count = len(lengths)
    for pieces in range(1, count + 1):
        size = -(-count // pieces)
        plan = [list(range(first, min(first + size, count))) for first in range(0, count, size)]
        if size == 1 or all(_cost(len(piece), max(lengths[i] for i in piece), new_tokens) <= budget for piece in plan):
            return plan
    return []


def _seconds(rows, max_new_tokens, temperature=0.0, seed=None) -> int:
    """How long one call may hold the GPU: the start of a new process, then every piece run to its last token.
    Small on purpose and never above two minutes: the platform checks the request against the caller's remaining quota."""
    pieces = _plan([len(row) for row in rows], max_new_tokens, HEADROOM)
    work = sum(PREFILL + max_new_tokens * (STEP + STEP_PER_ROW * len(piece)) for piece in pieces)
    return int(min(LONGEST, COLD_START + RESERVE + math.ceil(work)))


def _limited() -> bool:
    return str(os.getenv("SPACES_ZERO_GPU") or "").strip().lower() in ("1", "t", "true")     # as the `spaces` package reads it


def _full(error, torch) -> bool:
    """Whether a failure is the GPU running out of memory. A library's message is read here to tell; it is never shown or logged."""
    if isinstance(error, MemoryError) or (torch is not None and isinstance(error, torch.OutOfMemoryError)):
        return True
    return isinstance(error, RuntimeError) and any(sign in str(error).lower() for sign in ("out of memory", "alloc_failed"))


def _piece(model, rows, pad: int, max_new_tokens: int, temperature: float, seconds, torch) -> list[list[int]]:
    """One batch through the model: the new tokens of each row, and nothing of its prompt."""
    width = max(len(row) for row in rows)
    # Padded on the left, with a mask, so that every row's prompt ends where its answer begins.
    ids = [[pad] * (width - len(row)) + list(row) for row in rows]
    mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
    if torch is not None:
        ids, mask = torch.tensor(ids, device=model.device), torch.tensor(mask, device=model.device)
    options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "pad_token_id": pad}
    if temperature > 0:
        options["temperature"] = float(temperature)      # at 0 the most likely token is taken each time, and no temperature is passed
    if seconds is not None:
        options["max_time"] = seconds                    # the model stops here by itself; the platform would discard the whole request
    out = model.generate(input_ids=ids, attention_mask=mask, **options)
    out = out.tolist() if hasattr(out, "tolist") else out
    return [list(row[width:]) for row in out]


def _run(tokenizer, model, rows, max_new_tokens: int, temperature: float, seed, seconds, torch=None) -> dict:
    """Every row of a request in one stay on the GPU. `rows` are the prompts as token ids; `seconds` is the time
    there is, or None for no limit; `torch` is None for a test's stand-in model, which takes and returns lists.

    What comes back is plain strings and whole numbers, because it crosses a process boundary. A failure
    comes back as the name of its type and nothing else: raised, the platform would print its traceback, and
    a library's own message may quote the text it failed on.
    """
    started = time.monotonic()
    out = {"texts": [""] * len(rows), "prompt_tokens": 0, "completion_tokens": 0,
           "pieces": 0, "full": 0, "late": 0, "cut": 0, "free_mb": 0, "ms": 0}
    try:
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        on_gpu = torch is not None and str(getattr(model, "device", "cpu")).startswith("cuda")
        budget = HEADROOM
        if on_gpu:
            free = int(torch.cuda.mem_get_info()[0])
            budget, out["free_mb"] = int(free * SHARE), free // 2**20
        if torch is not None:
            # No seed means fresh randomness from the system. Left alone, a newly forked GPU process starts from torch's
            # fixed default, so the same batch would come back the same across calls and runs.
            torch.manual_seed(seed) if seed is not None else torch.seed()
        waiting = _plan([len(row) for row in rows], max_new_tokens, budget)
        while waiting:
            piece = waiting.pop(0)
            left = None if seconds is None else seconds - RESERVE - (time.monotonic() - started)
            if left is not None and left < PREFILL:
                out["late"] = len(piece) + sum(len(other) for other in waiting)      # no time to start them: their texts stay empty
                break
            new = None
            try:
                new = _piece(model, [rows[i] for i in piece], pad, max_new_tokens, temperature, left, torch)
            except Exception as error:  # noqa: BLE001 - only running out of memory is handled; anything else ends the request below
                if not _full(error, torch):
                    raise
            if new is None:                      # out of memory. Handled outside the except, so the failed attempt's tensors are already let go
                out["full"] += 1
                if on_gpu:
                    torch.cuda.empty_cache()
                if len(piece) > 1:               # try again in halves; a row that does not fit alone keeps its empty text
                    waiting[:0] = [piece[:len(piece) // 2], piece[len(piece) // 2:]]
                continue
            out["pieces"] += 1
            for index, ids in zip(piece, new, strict=True):
                while ids and ids[-1] == pad:
                    ids.pop()                    # rows that finished early were padded out to the longest
                out["prompt_tokens"] += len(rows[index])
                out["completion_tokens"] += len(ids)
                text, finished = final_text(tokenizer.decode(ids, skip_special_tokens=False))
                if finished:
                    out["texts"][index] = text
                elif text:
                    out["cut"] += 1              # an answer the token limit cut short is not an answer: the caller asks again
    except Exception as error:  # noqa: BLE001 - the message is dropped on purpose; see the docstring
        return {"failed": type(error).__name__}
    out["ms"] = int((time.monotonic() - started) * 1000)
    return out


_TOKENIZER = None    # the loaded tokenizer and model, as module globals: the process that runs _generate on a GPU reaches them by name
_MODEL = None


@spaces.GPU(duration=_seconds)
def _generate(rows: list[list[int]], max_new_tokens: int, temperature: float, seed: int | None) -> dict:
    import torch
    # The time this function has is what was asked for, less the start of the process that runs it. Off ZeroGPU nothing is counting.
    seconds = _seconds(rows, max_new_tokens) - COLD_START if _limited() else None
    return _run(_TOKENIZER, _MODEL, rows, max_new_tokens, temperature, seed, seconds, torch)


class TextGenerator:
    """A harmony chat model in bfloat16, on the GPU where the Space has one. Its libraries are imported only here."""

    def __init__(self, model_id: str, factory=None):
        self.name = model_id
        self._lock = threading.Lock()            # one request in the model at a time: no batch ever mixes two callers' prompts
        self.device, self._own = "cpu", factory is not None
        if factory is not None:                  # tests inject a stand-in for the library's tokenizer and model
            self.tokenizer, self.model = factory()
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"     # true on a ZeroGPU Space once `spaces` is imported
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        known = [self.tokenizer.convert_tokens_to_ids(marker) for marker in (_CHANNEL, _MESSAGE)]
        if any(not isinstance(i, int) or i < 0 or i == self.tokenizer.unk_token_id for i in known):
            raise ValueError("The text model does not use the harmony format; its answers could not be told from its reasoning.")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is None:
            raise ValueError("The text model's tokenizer has no token to pad a batch with.")
        options = {"dtype": torch.bfloat16}      # by name, as for the encoder: float32 would be twice the size and would not fit
        quantised = getattr(AutoConfig.from_pretrained(model_id), "quantization_config", None)
        method = quantised.get("quant_method") if isinstance(quantised, dict) else getattr(quantised, "quant_method", None)
        if str(getattr(method, "value", method)).lower() == "mxfp4":
            from transformers import Mxfp4Config
            # De-quantised as it loads, and said so, instead of left to what happens to be installed. See the module docstring.
            options["quantization_config"] = Mxfp4Config(dequantize=True)
        # Loaded on the CPU and then moved, the way ZeroGPU expects: its start-up records tensors moved to "cuda" and attaches them inside _generate.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **options)
        self.model.eval()
        self.model.to(self.device)
        self.gigabytes = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 10**9
        global _TOKENIZER, _MODEL
        _TOKENIZER, _MODEL = self.tokenizer, self.model

    def _ids(self, system: str, user: str) -> list[int]:
        """One prompt as the model's own template lays it out. The template files `system` as its developer instructions."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort=REASONING)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts, max_new_tokens: int, temperature: float, seed=None) -> dict:
        """`prompts` are (system, user) pairs. Texts come back in the same order; a prompt with no finished answer gets ""."""
        started = time.monotonic()
        with self._lock:
            rows = [self._ids(system, user) for system, user in prompts]
            asked = _seconds(rows, max_new_tokens)
            try:
                if self._own:                    # a test's stand-in model, run here
                    out = _run(self.tokenizer, self.model, rows, max_new_tokens, temperature, seed, None)
                else:
                    out = _generate(rows, max_new_tokens, temperature, seed)
            except Exception as error:
                log.warning("generator failed (%s)", type(error).__name__)       # the platform's own refusals arrive here: no quota left, no GPU in time
                raise
        if "failed" in out:
            log.warning("generator failed (%s)", out["failed"])
            raise RuntimeError("The text model could not complete this request.")
        # Counts and durations only. ms less gpu_ms is the wait for a GPU and the start of its process; asked_s is what the platform was asked for.
        log.info("generator rows=%d empty=%d cut=%d late=%d pieces=%d full=%d tokens_in=%d tokens_out=%d free_mb=%d gpu_ms=%d ms=%d asked_s=%d",
                 len(rows), sum(not text for text in out["texts"]), out["cut"], out["late"], out["pieces"], out["full"],
                 out["prompt_tokens"], out["completion_tokens"], out["free_mb"], out["ms"], (time.monotonic() - started) * 1000, asked)
        return {"texts": out["texts"], "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"]}


_GENERATOR = None    # the loaded text model, or None
_FAILURE = None      # why there is none although the deployment asked for one: a sentence naming the exception's type, and nothing else
_LOAD = threading.Lock()


def load(root=None, environ=None):
    """Called once as the Space starts: loads the text model if the deployment enables one, and returns it or None.
    It never raises. A text model that cannot load must not stop the Space scoring, so the failure is logged once and reported by `engine`."""
    global _GENERATOR, _FAILURE
    with _LOAD:
        _GENERATOR, _FAILURE = None, None
        started = time.monotonic()
        try:
            model_id = configured(root, environ)
            if model_id is not None:
                _GENERATOR = TextGenerator(model_id)
                log.info("generator loaded device=%s gb=%.1f s=%d", _GENERATOR.device, getattr(_GENERATOR, "gigabytes", 0.0), time.monotonic() - started)
                if _GENERATOR.device == "cuda":
                    from . import encoder
                    encoder.WARM_UP = COLD_START         # the encoder's GPU process now loads these weights too before it can score
        except Exception as error:  # noqa: BLE001 - whatever stops the text model, scoring goes on
            _FAILURE = f"The text model did not load ({type(error).__name__})."
            # Nothing a caller sent exists yet, so the traceback is safe to print, and it is what tells the deployer why.
            log.error("generator not loaded (%s) after s=%d", type(error).__name__, time.monotonic() - started, exc_info=True)
        return _GENERATOR


def current():
    """The Space's text model, or None when the deployment enables none or it failed to load."""
    return _GENERATOR


def failure() -> str | None:
    return _FAILURE
