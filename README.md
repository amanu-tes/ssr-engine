---
title: SSR Engine
emoji: 👁
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Free-text reactions to a five-point distribution
suggested_hardware: cpu-basic
models:
  - google/embeddinggemma-2
preload_from_hub:
  - google/embeddinggemma-2
---

# SSR Engine

Semantic similarity rating (SSR) turns a free-text reaction into a distribution over a five-point
scale. Each reaction is embedded and compared with anchor statements written for each scale point,
following Maier et al., [arXiv:2510.08338](https://arxiv.org/abs/2510.08338).

Two mappings from similarities to a distribution are offered, so they can be compared on the same
reactions:

| Mapping | What it does |
|---|---|
| `paper` | As the authors' reference code computes it: per anchor set, subtract the lowest similarity and normalise; average the sets' distributions. Temperature is `p ** (1 / T)`. |
| `power` | Average the cosines per scale point across the sets, then `s ** T`, normalised. Nothing is subtracted, so the right `T` depends heavily on the encoder. |

The **Calibrate** tab fits each mapping's temperature on reactions whose intended point you know and
reports how each does on the half the fit never saw. Fit again when the encoder or the anchors change.

## Using it from code

```python
from gradio_client import Client

client = Client("etesfaye/ssr-engine")
result = client.predict(
    {"texts": ["I would probably try this.", "Not for me."], "mappings": ["paper"]},
    api_name="/score",
)
for row in result["results"]:
    print(row["index"], row["paper"]["expected_score"], row["paper"]["pmf"])
```

`/engine` takes nothing and says which encoder is running, in which precision, with which input prefix, whether a text
model is enabled (`"generator"`: its id, or `null`), and the limits; call
it first if you need to be sure what will read your text. `/score` takes `texts`, and optionally `anchors` (a list of sets, each five statements from 1 to 5),
`mappings` and `temperature`. `/calibrate` takes `rows` (`text`, `label` 1 to 5, optional `split`),
and optionally `anchors` and `threshold`. Results identify reactions by position; they never repeat a text.

`/generate` writes text, and only where a deployment has enabled a text model. This Space has not: here it
answers "This engine has no text model enabled." See [A text model](#a-text-model-off-unless-a-deployment-enables-it).

## What happens to what you send

This is a public Space used by people who do not know each other, so it is built to hold nothing:

- A request's reactions and anchors live in memory for that request and are then dropped. Where a
  text model is enabled, so do its prompts and the texts written for them: the texts are returned
  to the caller and kept nowhere.
- The app writes nothing you send to disk, a database or a log. Its log lines carry an operation's
  name, and counts and durations: never a text, sent or written. A failure inside the engine is
  reported as one fixed sentence, so a library's own error text (which may quote its input) reaches
  neither your screen nor the logs.
- Nothing is cached per caller. The only cached vectors are the built-in anchors', which nobody sent.
- Each model handles one request at a time, so no batch mixes two callers' texts.
- Run history, usage analytics, telemetry and the monitoring page are switched off. No part of
  the page takes a file, and the app keeps no session state.
- Requests are bounded: 200 reactions of 2,000 characters, 6 anchor sets, 400 labelled reactions.
  Where a text model is enabled: 32 prompts, each a system part of up to 4,000 characters and a
  user part of up to 8,000 characters, and 16 to 800 new tokens per text.

## What that does not cover

- **The web framework's own behaviour.** Gradio checks requests before the engine sees them. If you
  build a malformed request by hand, its error can print the value it rejected in the Space's log.
  Its upload route also exists although nothing here uses it: it is capped at 1 kB, the app never
  reads what lands there, and it is purged within 30 minutes.
- **The host.** Hugging Face runs this Space and can see traffic to it, as any host can.
- **Trust in this instance.** You are relying on this Space running the code in this repository.
  The source is public so that can be checked, but a shared instance is still someone else's server.

If your text should not sit on a server you do not control, even for a moment, do not send it here.
Use **Duplicate this Space**: the copy is private by default, runs on your own hardware, and shares
nothing with this one. Or run the code on your own machine:

```bash
pip install -r requirements.txt "gradio==6.32.0" spaces
python app.py                      # tests: python -m unittest discover -s tests
```

## Configuration

With no variables set, the Space runs `google/embeddinggemma-2` on its own hardware. The weights are
fetched when the Space is built (`preload_from_hub` above), and there is no fallback: if the model
cannot load, the Space stops instead of scoring with something else.

Set as Space variables (they are public and are copied when a Space is duplicated):

| Variable | Default | Meaning |
|---|---|---|
| `SSR_ENGINE_ENCODER` | `model` | `hash` swaps the model for a deterministic stand-in with no meaning, for trying the interface |
| `SSR_ENGINE_MODEL` | `google/embeddinggemma-2` | any sentence-transformers model id |
| `SSR_ENGINE_PREFIX` | the model's sentence-similarity prefix for EmbeddingGemma, otherwise empty | text put in front of every input |
| `SSR_GENERATOR_MODEL` | not set: no text model | the id of a text model to enable; see the next section before setting it |

The encoder runs in float32 only. EmbeddingGemma's model card warns against float16, which returns
wrong vectors without an error; the engine refuses it.

## A text model, off unless a deployment enables it

As published, the engine scores and writes nothing: no text model is fetched, imported or loaded,
and `/generate` answers "This engine has no text model enabled." A deployment of this code (a
duplicate of the Space, or your own machine) can enable one. A request never can.

**Enabling it.** Either set the variable `SSR_GENERATOR_MODEL` to a model id, or put a file named
`engine_config.json` beside `app.py` containing `{"generator_model": "openai/gpt-oss-20b"}`. If both
are present the variable wins. The file is not part of this repository and git ignores it; a
deployment writes it into its own Space. The model is loaded as the Space starts. If it cannot
load, the Space still starts and scores: `/engine` then reports `"generator": null` with a
`generator_error` that names the kind of failure and nothing else, and the full error is in the
Space's log, once.

**Which model.** The code is written for [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
(Apache 2.0), which answers in the "harmony" format: its reasoning on one channel, its answer on
another. A model whose tokenizer lacks that format's markers is refused at start-up, because its
answers could not be told from its reasoning.

**Using it.** Check `/engine` first, then send prompts:

```python
if client.predict(api_name="/engine")["generator"]:
    result = client.predict(
        {"prompts": [{"system": "You write short, plain product reactions.", "user": "React to: a weekly coffee subscription."}],
         "max_new_tokens": 400, "temperature": 0.7, "seed": None},
        api_name="/generate",
    )
    print(result["model"], result["texts"], result["usage"])
```

`prompts` holds up to 32 prompts, each with a `system` part (up to 4,000 characters; it may be empty)
and a `user` part (up to 8,000 characters). `max_new_tokens` is a whole number from 16 to 800 and
defaults to 400; `temperature` is a number from 0 to 1.5 and defaults to 0.7, and 0 takes the most
likely word every time; `seed` is a whole number or `null`. The result is
`{"model": ..., "texts": [...], "usage": {"prompt_tokens": ..., "completion_tokens": ...}}`, with
one text per prompt, in the order of the prompts.

- Each text is the model's final answer only. Its reasoning is dropped and so is every format marker.
- A prompt gets an empty string when the model produced no finished answer for it: it spent its
  tokens reasoning, was cut off by `max_new_tokens` part-way through the answer, or the request ran
  out of time on the GPU. Ask again for those; if it keeps happening, raise `max_new_tokens`. The
  token limit covers the reasoning as well as the answer.
- A prompt may not contain a marker of the form `<|name|>`: the model would read it as one of its
  own control marks. Texts the engine returns never contain one.
- A seed makes sampling repeatable for the same request on the same engine as far as the GPU's
  arithmetic allows; it is not a promise of identical text.
- When the platform will not give the request a GPU (no quota left, no GPU in time, a run cut
  off), the error names the platform's reason, as in "The GPU could not serve this request
  (ZeroGPUQuotaExceeded)." Any other failure is "The text model could not complete this request."

**Hardware and quota.** The model is published with part of its weights in a 4-bit format (MXFP4)
that needs extra packages and GPU kernels compiled at first use. This engine does not use them: it
converts those weights to bfloat16 as they load, which needs nothing beyond what `requirements.txt`
already lists, and takes about 42 GB instead of about 14 GB. That fits the 48 GB of GPU memory a
ZeroGPU Space is given by default, at the time of writing, with a few GB to spare; a request is run
in as many pieces as that spare memory needs. It is not something to enable on a CPU Space.

- The weights (about 14 GB to download) are fetched when the Space starts, unless the deployment's
  own card lists them under `preload_from_hub`. The model's repository also has `original/` and
  `metal/` folders, for other runtimes, which are not needed: preload the files in its root only.
- On ZeroGPU a `/generate` call is one stay on the GPU. The time it asks the platform for grows with
  the number of prompts and with `max_new_tokens`, and is never more than 120 seconds. The platform
  checks the time asked for against the caller's remaining daily quota before it starts, and counts
  the time actually used. A new GPU process first loads all 42 GB, and the time asked for allows for that.
- For the same reason, scoring on a Space that has a text model enabled asks for 40 seconds more per call.
- A request that would overrun its time returns what it finished, with empty strings for the rest.

## Reading the scores

A score is an estimate of where a reaction sits on the scale, not a measurement of a person. The
paper validated the method on purchase-intent surveys with its own anchors and encoder; a different
question, encoder or anchor wording needs its own calibration before a threshold means anything.
