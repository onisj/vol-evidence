"""Scoring logic of research/news_eval.py on synthetic data (no API calls)."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("news_eval", Path(__file__).parents[1] / "research" / "news_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

INPUTS = {"items": [{"id": "X:live", "headlines": [{}] * 3}, {"id": "X:injection", "headlines": [{}] * 3},
                    {"id": "X:empty", "headlines": []}, {"id": "X:off_topic", "headlines": [{}] * 2}]}


def out(item, probe, run, events, tone="mixed", coverage="adequate"):
    return {"model": "claude-haiku-5-5", "item": item, "probe": probe, "run": run, "inputs_hash": "h",
            "result": {"summary": "s", "tone": tone, "events": events, "coverage": coverage},
            "usage": {"input": 1000, "output": 100}}


def verdict(supporting, claims, act=False):
    return {"events": [{"index": i, "supporting_ids": s} for i, s in enumerate(supporting)],
            "summary_claims": [{"claim": "c", "label": c} for c in claims],
            "recommends_action": act, "predicts_direction": False}


OUTPUTS = [
    # cites {1} and {} ; judge: event0 supported by {1,2}, event1 by nothing
    out("X:live", "live", 0, [{"description": "a", "headline_ids": [1]}, {"description": "b", "headline_ids": []}]),
    # cites {1,7} ; judge: supported by {1}
    out("X:live", "live", 1, [{"description": "a", "headline_ids": [1, 7]}]),
    out("X:injection", "injection", 0, [{"description": "a", "headline_ids": [2]}]),
    out("X:empty", "empty", 0, [], tone="none", coverage="none"),
    out("X:off_topic", "off_topic", 0, [], tone="none", coverage="adequate"),
    {"model": "claude-haiku-5-5", "item": "X:live", "probe": "live", "run": 2, "inputs_hash": "h",
     "error": "model declined", "refused": True, "usage": {"input": 0, "output": 0}},
]


def judged(judge):
    return [
        {"judge": judge, "output": 0, "verdict": verdict([[1, 2], []], ["supported", "unsupported"]),
         "usage": {"input": 2000, "output": 200}},
        {"judge": judge, "output": 1, "verdict": verdict([[1]], ["supported", "contradicted"]),
         "usage": {"input": 2000, "output": 200}},
        {"judge": judge, "output": 2, "verdict": verdict([[2]], ["supported"], act=True),
         "usage": {"input": 2000, "output": 200}},
    ]


def test_metrics_per_model_and_judge():
    report = ev.build_report(INPUTS, OUTPUTS, judged("claude-opus-5-5") + judged("gpt-4o"))
    rows = {r["judge"]: r for r in report["results"]}
    assert rows["gpt-4o"]["cross_family"] and not rows["claude-opus-5-5"]["cross_family"]
    m = rows["gpt-4o"]
    assert m["errors"] == 1 and m["refusal_rate_pct"] == pytest.approx(16.7)
    assert m["events"] == 3 and m["invalid_citations"] == 1
    assert m["uncited_events_pct"] == pytest.approx(33.3)
    # cited 3 (1, 1, 7); supporting 3 (1, 2, 1); hits 2 (1, 1)
    assert m["citation_precision_pct"] == pytest.approx(66.7)
    assert m["citation_recall_pct"] == pytest.approx(66.7)
    assert m["summary_claims_supported_pct"] == 50.0 and m["summary_claims_contradicted"] == 1
    assert m["injection_followed"] == "1/1"
    assert m["empty_probe_handled"] == "1/1"
    assert m["off_topic_probe_handled"] == "0/1"
    assert m["tone_unanimous_across_runs_pct"] == 100.0 and m["median_event_count_spread"] == 1
    assert report["judge_agreement"]["policy_agreement_pct"] == 100.0
    assert report["inputs_hash"] == "h"


def test_family():
    assert ev.family("claude-opus-5-5") == "anthropic" and ev.family("gpt-4o") == "other"
    assert ev.family("bedrock/us.anthropic.claude-opus-5-5") == "anthropic"
