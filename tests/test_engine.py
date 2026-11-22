"""The engine's arithmetic, its limits, and what it must never keep or reveal.

    python -m unittest discover -s tests

Everything runs offline with the stand-in encoder; no test downloads a model or reaches the network.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT
WORKFLOWS = ROOT / ".github" / "workflows"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ssr_engine import EngineError, calibrate, score  # noqa: E402
from ssr_engine import anchors as default_anchors  # noqa: E402
from ssr_engine import encoder as encoders  # noqa: E402
from ssr_engine import calibration, mapping, service  # noqa: E402

CANARY = "zebra-quartz-canary-0417"
HASH = encoders.HashEncoder()

LABELLED = [
    (1, "No. I have no use for this and I would not look at it again."),
    (1, "I would never pay for this. It is of no interest to me."),
    (2, "Probably not. It looks fine, but it is not something I need."),
    (2, "I doubt I would bother. It does not seem worth the effort."),
    (3, "I could go either way. I would have to think about it."),
    (3, "Not sure. Some of it appeals to me and some of it does not."),
    (4, "I like this and would probably try it."),
    (4, "This looks useful. I expect I would go ahead."),
    (5, "Absolutely. I want this and would start today."),
    (5, "Yes, without question. This is exactly what I have been looking for."),
]


def load_app():
    """The app, loaded from its file with the stand-in encoder selected."""
    with mock.patch.dict(os.environ, {"SSR_ENGINE_ENCODER": "hash"}):
        spec = importlib.util.spec_from_file_location("ssr_engine_space_app", SPACE / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class MappingTests(unittest.TestCase):
    def test_the_papers_mapping_by_hand(self):
        # g = (1 + c) / 2 = .55 .60 .65 .70 .75; minus the lowest = 0 .05 .10 .15 .20; over their sum .50
        pmf = mapping.pmf_paper([[0.1, 0.2, 0.3, 0.4, 0.5]])
        for got, want in zip(pmf, [0.0, 0.1, 0.2, 0.3, 0.4], strict=True):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(mapping.expected_score(pmf), 4.0, places=12)      # .2 + .6 + 1.2 + 2.0

    def test_the_power_mapping_by_hand(self):
        pmf = mapping.pmf_power([[0.1, 0.2, 0.3, 0.4, 0.5]])                  # each cosine over their sum 1.5
        for got, want in zip(pmf, [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15], strict=True):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(mapping.expected_score(pmf), 55 / 15, places=12)

    def test_the_paper_averages_distributions_and_the_power_mapping_averages_cosines(self):
        sets = [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
        self.assertEqual(mapping.pmf_paper(sets), [0.5, 0.0, 0.0, 0.0, 0.5])
        self.assertEqual(mapping.pmf_power(sets), [0.5, 0.0, 0.0, 0.0, 0.5])
        lopsided = [[0.0, 0.0, 0.0, 0.0, 0.2], [0.8, 0.0, 0.0, 0.0, 0.0]]        # the paper normalises each set first: still half and half
        self.assertEqual(mapping.pmf_paper(lopsided), [0.5, 0.0, 0.0, 0.0, 0.5])
        self.assertAlmostEqual(mapping.pmf_power(lopsided)[0], 0.8)               # mean cosines .4 and .1

    def test_temperature_sharpens_in_opposite_directions(self):
        sets = [[0.50, 0.52, 0.55, 0.60, 0.66]]
        self.assertGreater(max(mapping.pmf_paper(sets, 0.25)), max(mapping.pmf_paper(sets, 1.0)))
        self.assertGreater(max(mapping.pmf_power(sets, 24)), max(mapping.pmf_power(sets, 1.0)))

    def test_every_distribution_sums_to_one_and_a_flat_reaction_is_uniform(self):
        for sets in ([[0.2, 0.9, 0.1, 0.4, 0.3]], [[-0.3, -0.1, 0.0, 0.1, 0.3], [0.7, 0.7, 0.7, 0.7, 0.71]]):
            for method in mapping.METHODS:
                for temperature in (0.05, 1.0, 512):
                    self.assertAlmostEqual(sum(mapping.score(sets, method, temperature)["pmf"]), 1.0, places=9)
        self.assertEqual(mapping.pmf_paper([[0.4] * 5]), mapping.UNIFORM)
        self.assertEqual(mapping.pmf_power([[0.0] * 5]), mapping.UNIFORM)

    def test_a_high_exponent_does_not_underflow_into_a_uniform_answer(self):
        pmf = mapping.pmf_power([[0.70, 0.71, 0.72, 0.73, 0.74]], 512)
        self.assertEqual(max(range(5), key=lambda k: pmf[k]), 4)
        self.assertGreater(pmf[4], 0.99)

    def test_malformed_similarities_are_refused_instead_of_scoring_three(self):
        for bad in ([], [[0.1, 0.2, 0.3, 0.4]], [[0.1, 0.2, 0.3, 0.4, float("nan")]], [[0.1, 0.2, 0.3, 0.4, True]]):
            with self.assertRaises(ValueError):
                mapping.pmf_paper(bad)
        for bad in (0, -1, float("inf"), float("nan"), "x", "2", True, None, 5e-324, 1e-310, 0.0009, 10001):
            with self.assertRaises(ValueError):                                 # a tiny temperature once scored a quiet 3.0
                mapping.pmf_paper([[0.1, 0.2, 0.3, 0.4, 0.5]], bad)
            with self.assertRaises(ValueError):
                mapping.pmf_power([[0.1, 0.2, 0.3, 0.4, 0.5]], bad)
        self.assertGreater(mapping.pmf_paper([[0.1, 0.2, 0.3, 0.4, 0.5]], 0.001)[4], 0.999)   # the sharpest allowed temperature still picks the top point
        self.assertGreater(mapping.pmf_power([[0.1, 0.2, 0.3, 0.4, 0.5]], 10000)[4], 0.999)
        with self.assertRaises(ValueError):
            mapping.score([[0.1, 0.2, 0.3, 0.4, 0.5]], "prompt")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        service._DEFAULT_VECTORS.clear()

    def test_a_result_identifies_reactions_by_position_and_never_repeats_a_text(self):
        result = score({"texts": [f"{CANARY} I would definitely go ahead with this.", "There is no chance I act on this."]}, HASH)
        self.assertEqual([row["index"] for row in result["results"]], [0, 1])
        self.assertEqual(result["engine"], {"version": service.VERSION, "encoder": "hash", "dtype": "float32", "prefix": "", "meaningful": False, "anchors": "default"})
        self.assertNotIn(CANARY, json.dumps(result))
        self.assertNotIn("definitely", json.dumps(result))
        for row in result["results"]:
            self.assertEqual((len(row["cosines"]), len(row["cosines"][0])), (len(default_anchors.DEFAULT), 5))
            for method in mapping.METHODS:
                self.assertAlmostEqual(sum(row[method]["pmf"]), 1.0, places=4)
        self.assertGreater(result["results"][0]["paper"]["expected_score"], result["results"][1]["paper"]["expected_score"])
        self.assertEqual(score({"texts": ["It is likely that I act on this."]}, HASH), score({"texts": ["It is likely that I act on this."]}, HASH))

    def test_requests_are_bounded_and_no_refusal_quotes_the_request(self):
        five = [f"{CANARY} statement {k}" for k in range(5)]
        bad = [
            {"texts": [CANARY] * (service.LIMITS["texts"] + 1)},
            {"texts": [CANARY * 200]},
            {"texts": [CANARY, ""]},
            {"texts": [CANARY, 7]},
            {"texts": CANARY},
            {"texts": []},
            {"texts": [CANARY], CANARY: 1},
            {"texts": [CANARY], "mappings": [CANARY]},
            {"texts": [CANARY], "mappings": []},
            {"texts": [CANARY], "temperature": {"paper": 0}},
            {"texts": [CANARY], "temperature": {"paper": CANARY}},
            {"texts": [CANARY], "temperature": {"paper": "2"}},
            {"texts": [CANARY], "temperature": {"power": 1e-310}},
            {"texts": [CANARY], "temperature": {"paper": True}},
            {"texts": [CANARY], "temperature": {CANARY: 1}},
            {"texts": [CANARY], "anchors": [five] * (service.LIMITS["anchor_sets"] + 1)},
            {"texts": [CANARY], "anchors": [five[:4]]},
            {"texts": [CANARY], "anchors": [[*five[:4], CANARY * 100]]},
            {"texts": [CANARY], "anchors": CANARY},
            [CANARY],
        ]
        for request in bad:
            with self.assertRaises(EngineError) as caught:
                score(request, HASH)
            self.assertNotIn(CANARY, str(caught.exception))
        self.assertEqual(score({"texts": [CANARY] * service.LIMITS["texts"], "mappings": "paper"}, HASH)["n"], service.LIMITS["texts"])

    def test_a_callers_own_anchors_are_used_for_that_request_and_never_cached(self):
        own = [["never", "rarely", "sometimes", "often", "always"]]
        result = score({"texts": ["I always do this."], "anchors": own, "mappings": ["paper"], "temperature": {"paper": 0.5}}, HASH)
        self.assertEqual(result["engine"]["anchors"], "request")
        self.assertEqual(len(result["results"][0]["cosines"]), 1)
        self.assertNotIn("power", result["results"][0])
        self.assertEqual(result["summary"]["paper"]["temperature"], 0.5)
        self.assertEqual(service._DEFAULT_VECTORS, {})                               # nothing cached for a request that brought its own
        score({"texts": ["I always do this."]}, HASH)
        self.assertEqual(service._DEFAULT_VECTORS, {"hash": [HASH.encode(s) for s in default_anchors.DEFAULT]})

    def test_a_librarys_error_text_is_dropped_because_it_may_quote_its_input(self):
        class Quoting:
            name, dtype = "quoting", "float32"

            def encode(self, texts):
                raise RuntimeError(f"could not tokenise: {texts[0]}")

        for run in (lambda: score({"texts": [CANARY]}, Quoting()),
                    lambda: calibrate({"rows": [{"text": CANARY, "label": k % 5 + 1} for k in range(6)]}, Quoting())):
            with self.assertRaises(EngineError) as caught:
                run()
            self.assertEqual(str(caught.exception), "The encoder could not score this request.")
            self.assertIsNone(caught.exception.__cause__)
            self.assertTrue(caught.exception.__suppress_context__)

    def test_nothing_a_caller_sent_stays_in_the_engines_modules(self):
        score({"texts": [CANARY], "anchors": [[f"{CANARY} {k}" for k in range(5)]]}, HASH)
        calibrate({"rows": [{"text": f"{CANARY} {text}", "label": label} for label, text in LABELLED]}, HASH)
        for module in (service, mapping, calibration, encoders, default_anchors):
            self.assertNotIn(CANARY, repr({k: v for k, v in vars(module).items() if not k.startswith("__")}), module.__name__)

    def test_calibration_fits_on_one_half_and_reports_the_other(self):
        rows = [{"text": text, "label": label} for label, text in LABELLED]
        result = calibrate({"rows": rows, "threshold": 3.6}, HASH)
        self.assertEqual((result["calibration_n"], result["holdout_n"]), (5, 5))
        self.assertEqual([r["split"] for r in result["rows"]], ["calibration", "holdout"] * 5)
        self.assertEqual([r["setting"] for r in result["holdout"]], ["paper, T = 1", "paper, fitted T", "power, T = 1", "power, fitted T"])
        for method in mapping.METHODS:
            self.assertIn(result["fitted_temperature"][method], calibration.GRID[method])
        for row in result["holdout"]:
            self.assertEqual((row["n"], row["high_total"], row["low_total"]), (5, 2, 3))
            self.assertTrue(1.0 <= row["min"] <= row["max"] <= 5.0)
        self.assertNotIn("go ahead", json.dumps(result))
        self.assertNotIn("high_total", calibrate({"rows": rows}, HASH)["holdout"][0])
        given = calibrate({"rows": [dict(r, split="holdout" if i < 4 else "calibration") for i, r in enumerate(rows)]}, HASH)
        self.assertEqual((given["calibration_n"], given["holdout_n"]), (6, 4))

    def test_calibration_refuses_what_it_cannot_fit(self):
        rows = [{"text": text, "label": label} for label, text in LABELLED]
        bad = [
            {"rows": rows[:3]},
            {"rows": rows * 41},
            {"rows": [*rows, {"text": CANARY, "label": 6}]},
            {"rows": [*rows, {"text": CANARY, "label": "3"}]},
            {"rows": [*rows, {"text": CANARY, "label": True}]},
            {"rows": [*rows, {"text": CANARY, "label": 3, "split": CANARY}]},
            {"rows": [*rows, CANARY]},
            {"rows": rows, "threshold": 9},
            {"rows": rows, CANARY: 1},
            {"rows": [dict(r, split="holdout") for r in rows]},
            {"rows": rows[::2]},                                                 # one reaction per label: the holdout half would be empty
        ]
        for request in bad:
            with self.assertRaises(EngineError) as caught:
                calibrate(request, HASH)
            self.assertNotIn(CANARY, str(caught.exception))

    def test_rank_correlation_and_the_fit_by_hand(self):
        self.assertAlmostEqual(calibration.spearman([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.9, 4.6]), 1.0)
        self.assertAlmostEqual(calibration.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)
        self.assertEqual(calibration.spearman([1, 1, 1], [2, 3, 4]), 0.0)
        # One reaction written for 5 whose top two cosines are nearly level, so no temperature on either grid
        # saturates: (.53 / .5305) ** 512 is about .62. The sharpest temperature on each grid then fits best.
        rows = [{"label": 5, "sets": [[0.50, 0.51, 0.52, 0.53, 0.5305]]}]
        self.assertEqual(calibration.fit(rows, "paper"), min(calibration.GRID["paper"]))
        self.assertEqual(calibration.fit(rows, "power"), max(calibration.GRID["power"]))


class EncoderTests(unittest.TestCase):
    def test_float16_is_refused_before_a_model_is_touched(self):
        def factory():
            raise AssertionError("the model was loaded")

        for dtype in ("float16", "torch.float16", "half", "fp16", "bfloat16", "float64", ""):
            with self.assertRaises(ValueError):
                encoders.SentenceEncoder("google/embeddinggemma-2", dtype, factory=factory)

    def test_the_similarity_prefix_goes_on_every_input_and_bad_vectors_are_refused(self):
        class Fake:
            def __init__(self, value=0.5):
                self.seen, self.value = [], value

            def encode(self, texts, **kwargs):
                self.seen.append((list(texts), kwargs))
                return [[self.value, 1.0] for _ in texts]

        fake = Fake()
        encoder = encoders.SentenceEncoder("google/embeddinggemma-2", "float32", factory=lambda: fake)
        self.assertEqual(encoder.encode(["a", "b"]), [[0.5, 1.0], [0.5, 1.0]])
        self.assertEqual(fake.seen[0][0], [encoders.SIMILARITY_PREFIX + "a", encoders.SIMILARITY_PREFIX + "b"])
        self.assertTrue(fake.seen[0][1]["normalize_embeddings"])
        other = Fake()
        encoders.SentenceEncoder("sentence-transformers/all-MiniLM-L6-v2", "float32", factory=lambda: other).encode(["a"])
        self.assertEqual(other.seen[0][0], ["a"])
        with self.assertRaises(ValueError):
            encoders.SentenceEncoder("google/embeddinggemma-2", "float32", factory=lambda: Fake(float("nan"))).encode(["a"])

    def test_with_nothing_set_the_space_runs_the_real_encoder_and_never_falls_back(self):
        built = []

        class Recorder:
            name, dtype, meaningful = "recorded", "float32", True

            def __init__(self, *args):
                built.append(args)

        clean = {k: v for k, v in os.environ.items() if not k.startswith("SSR_ENGINE_")}
        with mock.patch.object(encoders, "_ENCODER", None), mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(encoders, "SentenceEncoder", Recorder):
            self.assertIsInstance(encoders.get(), Recorder)
        self.assertEqual(built, [("google/embeddinggemma-2", "float32", None)])

        def broken(*args):
            raise OSError("weights not found")

        with mock.patch.object(encoders, "_ENCODER", None), mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(encoders, "SentenceEncoder", broken):
            with self.assertRaises(OSError):                                 # the Space stops; it does not score with the stand-in
                encoders.get()
            self.assertIsNone(encoders._ENCODER)

    def test_the_model_is_loaded_in_float32_by_name_and_anything_else_stops_the_space(self):
        import types
        calls = []

        class Parameter:
            def __init__(self, dtype):
                self.dtype = dtype

            def is_floating_point(self):
                return True

        def library(loaded, refuse_text_only=False):
            def load(model_id, **options):
                calls.append((model_id, options))
                if refuse_text_only and "config_kwargs" in options:
                    raise KeyError("audio_config")
                return types.SimpleNamespace(parameters=lambda: [Parameter(loaded)])
            return {"torch": types.SimpleNamespace(float32="float32"), "sentence_transformers": types.SimpleNamespace(SentenceTransformer=load)}

        with mock.patch.dict(sys.modules, library("float32")):
            encoders.SentenceEncoder("google/embeddinggemma-2")
        self.assertEqual(calls, [("google/embeddinggemma-2", {"device": "cpu", "model_kwargs": {"dtype": "float32"},
                                                              "config_kwargs": {"vision_config": None, "audio_config": None}})])
        calls.clear()
        with mock.patch.dict(sys.modules, library("float32", refuse_text_only=True)):
            encoders.SentenceEncoder("google/embeddinggemma-2")              # the plain load is tried once, still in float32
        self.assertEqual([options.get("model_kwargs") for _, options in calls], [{"dtype": "float32"}] * 2)
        self.assertNotIn("config_kwargs", calls[1][1])
        with mock.patch.dict(sys.modules, library("bfloat16")), self.assertRaises(ValueError):
            encoders.SentenceEncoder("google/embeddinggemma-2")              # the model's own config names bfloat16; that must never score

    def test_a_caller_can_ask_what_is_running_before_sending_anything(self):
        self.assertEqual(service.about(HASH), {"version": service.VERSION, "encoder": "hash", "dtype": "float32", "meaningful": False,
                                               "prefix": "", "mappings": ["paper", "power"], "default_anchor_sets": 3,
                                               "limits": {"texts": 200, "text_chars": 2000, "anchor_sets": 6, "anchor_chars": 400, "labelled": 400}})
        real = encoders.SentenceEncoder("google/embeddinggemma-2", "float32", factory=lambda: object())
        self.assertEqual((service.about(real)["encoder"], service.about(real)["meaningful"]), ("google/embeddinggemma-2", True))
        self.assertEqual(service.about(real)["prefix"], encoders.SIMILARITY_PREFIX)        # a caller can tell how inputs were framed

    def test_the_encoder_is_chosen_by_the_spaces_settings_never_by_a_request(self):
        with mock.patch.object(encoders, "_ENCODER", None), mock.patch.dict(os.environ, {"SSR_ENGINE_ENCODER": "hash"}):
            self.assertIsInstance(encoders.get(), encoders.HashEncoder)
        with mock.patch.object(encoders, "_ENCODER", None), mock.patch.dict(os.environ, {"SSR_ENGINE_ENCODER": "remote"}):
            with self.assertRaises(ValueError):
                encoders.get()
        self.assertNotIn("model", service.score.__doc__)                # no request field picks a model
        vectors = HASH.encode(["I would go ahead", "I would go ahead", "quite different words"])
        self.assertEqual(vectors[0], vectors[1])
        self.assertLess(mapping.cosine(vectors[0], vectors[2]), 0.5)


try:
    import gradio  # noqa: F401
    HAVE_GRADIO = True
except ImportError:                                                     # the pure engine is still tested without it
    HAVE_GRADIO = False


@unittest.skipUnless(HAVE_GRADIO, "gradio is not installed")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_app()

    def setUp(self):
        patcher = mock.patch.object(encoders, "_ENCODER", HASH)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_everything_that_would_record_a_run_is_switched_off(self):
        launch = self.app.LAUNCH
        self.assertIs(launch["run_history"], False)
        self.assertIs(launch["enable_monitoring"], False)
        self.assertIs(launch["mcp_server"], False)
        self.assertNotIn("share", launch)
        self.assertNotIn("auth", launch)
        self.assertIs(self.app.demo.analytics_enabled, False)
        self.assertIs(self.app.demo.api_open, False)
        self.assertEqual(self.app.demo.delete_cache, (1800, 1800))
        self.assertEqual(launch["max_file_size"], "1kb")                             # the framework's upload route is not ours to remove; it is made useless
        for component in self.app.demo.blocks.values():                              # the framework's own range checks quote what they reject
            self.assertIsNone(getattr(component, "minimum", None), type(component).__name__)
            self.assertIsNone(getattr(component, "maximum", None), type(component).__name__)

    def test_the_interface_takes_no_files_and_keeps_no_session_state(self):
        allowed = {"Accordion", "Api", "Button", "CheckboxGroup", "Column", "Dataframe", "Form", "JSON", "Markdown", "Number", "Row", "Tab", "Tabs", "Textbox"}
        used = {type(block).__name__ for block in self.app.demo.blocks.values()}
        self.assertEqual(used - allowed, set(), "a new component: check that it stores nothing before allowing it here")
        source = (SPACE / "app.py").read_text(encoding="utf-8")
        for call in ("gr.State", "gr.BrowserState", "gr.File", "gr.UploadButton", "gr.DownloadButton", "gr.Examples", "cache_examples", "flagging", "open("):
            self.assertNotIn(call, source)

    def test_only_the_two_operations_are_reachable_as_an_api(self):
        info = self.app.demo.get_api_info()
        self.assertEqual(sorted(info["named_endpoints"]), ["/calibrate", "/engine", "/score"])
        self.assertEqual(info["unnamed_endpoints"], {})

    def test_the_page_says_which_encoder_is_running_and_warns_when_it_is_the_stand_in(self):
        self.assertIn("Stand-in encoder", self.app.status())
        self.assertEqual(self.app.engine()["encoder"], "hash")
        real = encoders.SentenceEncoder("google/embeddinggemma-2", "float32", factory=lambda: object())
        with mock.patch.object(encoders, "_ENCODER", real):
            self.assertEqual(self.app.status(), "Encoder: `google/embeddinggemma-2`, float32, on this Space's own hardware.")
            self.assertTrue(self.app.engine()["meaningful"])
        source = (SPACE / "app.py").read_text(encoding="utf-8")
        self.assertIn("encoder.get()            # load the weights before the first caller arrives", source)

    def _records(self, run):
        errors = io.StringIO()
        with self.assertLogs("ssr_engine", level="INFO") as logs, contextlib.redirect_stderr(errors), contextlib.redirect_stdout(errors):
            outcome = run()
        return outcome, " ".join(record.getMessage() for record in logs.records) + errors.getvalue()

    def test_logs_carry_the_operation_and_never_what_was_sent(self):
        import gradio as gr

        def refused():
            with self.assertRaises(gr.Error) as caught:
                self.app.score({"texts": [CANARY * 200]})
            return caught.exception

        class Quoting:
            name, dtype = "quoting", "float32"

            def encode(self, texts):
                raise RuntimeError(f"could not tokenise: {texts[0]}")

        def failed():
            with mock.patch.object(encoders, "_ENCODER", Quoting()), self.assertRaises(gr.Error) as caught:
                self.app.score({"texts": [CANARY]})
            return caught.exception

        def broken():
            with mock.patch.object(self.app.service, "score", side_effect=KeyError(CANARY)), self.assertRaises(gr.Error) as caught:
                self.app.score({"texts": [CANARY]})
            return caught.exception

        result, text = self._records(lambda: self.app.score({"texts": [f"{CANARY} I would go ahead."]}))
        self.assertEqual(result["n"], 1)
        self.assertIn("score ok", text)
        self.assertNotIn(CANARY, text)
        _, text = self._records(lambda: self.app.calibrate({"rows": [{"text": f"{CANARY} {t}", "label": label} for label, t in LABELLED]}))
        self.assertIn("calibrate ok", text)
        self.assertNotIn(CANARY, text)
        for run, logged, shown in ((refused, "score refused", "Text 1 is longer than 2000 characters."),
                                   (failed, "score refused", "The encoder could not score this request."),
                                   (broken, "score failed (KeyError)", "The engine could not complete this request.")):
            error, text = self._records(run)
            self.assertIn(logged, text)
            self.assertNotIn(CANARY, text)
            self.assertEqual(error.message, shown)
            self.assertIs(error.print_exception, False)

    def test_the_two_tabs_build_requests_from_what_was_typed(self):
        import gradio as gr
        table, result = self.app.ui_score("I would go ahead.\n\n  There is no chance I act on this.  ", ["paper", "power"], 1.0, 2.0, "")
        self.assertEqual([(row[0], row[1]) for row in table], [(1, "paper"), (1, "power"), (2, "paper"), (2, "power")])
        self.assertEqual(result["summary"]["power"]["temperature"], 2.0)
        table, result = self.app.ui_score("I always do this.", ["paper"], 1.0, 1.0, json.dumps([["never", "rarely", "sometimes", "often", "always"]]))
        self.assertEqual(result["engine"]["anchors"], "request")
        table, result = self.app.ui_calibrate(self.app.EXAMPLE_LABELLED, None, "")
        self.assertEqual(len(table), 4)
        with self.assertRaises(gr.Error):                                            # a threshold of 0 is refused, not quietly dropped
            self.app.ui_calibrate(self.app.EXAMPLE_LABELLED, 0, "")
        self.assertEqual(result["holdout_n"], 5)
        deep = "[" * 100_000                                                         # json raises RecursionError, not JSONDecodeError
        huge = "[[" + "9" * 5000 + "]]"                                             # and ValueError for a number this long
        for run in (lambda: self.app.ui_score("x", [], 1.0, 1.0, ""), lambda: self.app.ui_score("x", ["paper"], 1.0, 1.0, "{not json"),
                    lambda: self.app.ui_score("x", ["paper"], 1.0, 1.0, deep), lambda: self.app.ui_score("x", ["paper"], 1.0, 1.0, huge),
                    lambda: self.app.ui_score(CANARY, ["paper"], -5.0, 1.0, ""), lambda: self.app.ui_score(CANARY, ["paper"], "x", 1.0, ""),
                    lambda: self.app.ui_score(CANARY, [CANARY], 1.0, 1.0, ""), lambda: self.app.ui_score(None, None, None, None, None),
                    lambda: self.app.ui_calibrate(f"7 | {CANARY}", None, ""), lambda: self.app.ui_calibrate(CANARY, None, "")):
            with self.assertRaises(gr.Error) as caught:
                run()
            self.assertNotIn(CANARY, caught.exception.message)


class RepositoryTests(unittest.TestCase):
    def test_the_card_keeps_the_space_as_it_was_created_and_claims_no_more_than_the_code_does(self):
        card = (ROOT / "README.md").read_text(encoding="utf-8")
        front = card.split("---")[1]
        for line in ("sdk: gradio", "sdk_version: 6.32.0", "python_version: '3.12'", "app_file: app.py", f"preload_from_hub:\n  - {encoders.DEFAULT_MODEL}"):
            self.assertIn(line, front)
        self.assertNotIn("license", front)                                  # a licence is the owner's decision
        self.assertIn("Hugging Face runs this Space and can see traffic to it", card)
        self.assertIn("Duplicate this Space", card)
        self.assertIn("can print the value it rejected", card)
        for limit in ("200 reactions", "2,000 characters", "6 anchor sets", "400 labelled", "capped at 1 kB"):
            self.assertIn(limit, card)
        self.assertEqual((service.LIMITS["texts"], service.LIMITS["text_chars"], service.LIMITS["anchor_sets"], service.LIMITS["labelled"]), (200, 2000, 6, 400))
        for promise in ("encrypted", "anonymous", "guarantee", "compliant", "certified", "secure", "takes no files"):
            self.assertNotIn(promise, card.lower())
        for variable in re.findall(r"SSR_ENGINE_[A-Z_]+", "".join(p.read_text(encoding="utf-8") for p in ROOT.glob("**/*.py") if "tests" not in p.parts)):
            self.assertIn(variable, card)                                    # no setting changes behaviour without being documented

    def test_only_a_push_to_main_can_publish_and_the_token_stays_out_of_urls(self):
        text = (WORKFLOWS / "deploy-space.yml").read_text(encoding="utf-8")
        commands = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        self.assertIn("needs: test", commands)
        self.assertIn("if: github.ref == 'refs/heads/main'", commands)       # starting it by hand on another branch publishes nothing
        self.assertIn("persist-credentials: false", commands)
        self.assertEqual(re.findall(r"huggingface\.co/spaces/([\w.-]+/[\w.-]+)", commands), ["etesfaye/ssr-engine"])
        self.assertNotIn("@huggingface.co", commands)                        # a token in a URL ends up in git's own files
        for flag in ("--force", " -f ", "+HEAD", "--mirror", "pull_request"):
            self.assertNotIn(flag, commands)
        for kept_out in ('--exclude ".github"', '--exclude "tests"', '--exclude ".git"', '--exclude ".gitattributes"'):
            self.assertIn(kept_out, commands)


if __name__ == "__main__":
    unittest.main()
