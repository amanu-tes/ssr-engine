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

`/engine` takes nothing and says which encoder is running, in which precision, with which input prefix, and the limits; call
it first if you need to be sure what will read your text. `/score` takes `texts`, and optionally `anchors` (a list of sets, each five statements from 1 to 5),
`mappings` and `temperature`. `/calibrate` takes `rows` (`text`, `label` 1 to 5, optional `split`),
and optionally `anchors` and `threshold`. Results identify reactions by position; they never repeat a text.

## What happens to what you send

This is a public Space used by people who do not know each other, so it is built to hold nothing:

- A request's reactions and anchors live in memory for that request and are then dropped.
- The app writes nothing you send to disk, a database or a log. Its log lines carry an operation's
  name and duration only. A failure inside the engine is reported as one fixed sentence, so a
  library's own error text (which may quote its input) reaches neither your screen nor the logs.
- Nothing is cached per caller. The only cached vectors are the built-in anchors', which nobody sent.
- The model handles one request at a time, so no batch mixes two callers' texts.
- Run history, usage analytics, telemetry and the monitoring page are switched off. No part of
  the page takes a file, and the app keeps no session state.
- Requests are bounded: 200 reactions of 2,000 characters, 6 anchor sets, 400 labelled reactions.

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
pip install -r requirements.txt "gradio==6.32.0"
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

The encoder runs in float32 only. EmbeddingGemma's model card warns against float16, which returns
wrong vectors without an error; the engine refuses it.

## Reading the scores

A score is an estimate of where a reaction sits on the scale, not a measurement of a person. The
paper validated the method on purchase-intent surveys with its own anchors and encoder; a different
question, encoder or anchor wording needs its own calibration before a threshold means anything.
