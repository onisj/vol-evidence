import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from volevidence import news

NOW = datetime(2026, 11, 18, 21, 0, tzinfo=timezone.utc)
H = [news.Headline("Chipmaker names interim CEO", "The CFO steps up.", "Wire", NOW)]
GOOD = {"summary": "s", "tone": "mixed", "events": [{"description": "CEO change", "headline_ids": [1]}],
        "coverage": "thin"}


def test_parses_nested_and_flat_news_formats():
    nested = {"content": {"title": "A", "summary": "b", "pubDate": "2026-11-18T20:26:19Z",
                          "provider": {"displayName": "Yahoo Finance"}}}
    flat = {"title": "B", "publisher": "Reuters",
            "providerPublishTime": datetime(2026, 11, 17, 9, tzinfo=timezone.utc).timestamp()}
    out = news.parse_news([flat, nested], days=3, now=NOW)
    assert [h.title for h in out] == ["A", "B"] and out[0].source == "Yahoo Finance"


def test_drops_old_or_undated_items():
    raw = [{"content": {"title": "old", "pubDate": "2026-10-01T00:00:00Z"}}, {"content": {"title": "undated"}}]
    assert news.parse_news(raw, days=3, now=NOW) == []


def test_render_numbers_headlines_inside_a_data_block():
    text = news.render("X", H)
    assert "<headlines>" in text and "[1]" in text and "The CFO steps up." in text
    assert "(no headlines in the window)" in news.render("X", [])


def fake_anthropic(monkeypatch, content, stop="end_turn"):
    sent = {}

    class Messages:
        def create(self, **kw):
            sent.update(kw)
            return SimpleNamespace(content=content, stop_reason=stop,
                                   usage=SimpleNamespace(input_tokens=11, output_tokens=7))

    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=lambda **_: SimpleNamespace(messages=Messages())))
    return sent


def test_reads_text_after_a_thinking_block(monkeypatch):
    sent = fake_anthropic(monkeypatch, [SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(GOOD))])
    result, usage = news.digest("X", H, "claude-haiku-5-5")
    assert result == GOOD and usage == {"input": 11, "output": 7}
    assert sent["output_config"]["format"]["schema"] is news.SCHEMA
    assert sent["system"] == news.INSTRUCTIONS


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_and_truncation_raise(monkeypatch, stop):
    fake_anthropic(monkeypatch, [SimpleNamespace(type="text", text="{")], stop)
    with pytest.raises(news.DigestError):
        news.digest("X", H, "claude-haiku-5-5")


def test_citation_check():
    bad = {"events": [{"description": "a", "headline_ids": []}, {"description": "b", "headline_ids": [1, 9]}],
           "coverage": "adequate"}
    assert news.citation_check(bad, 3) == {"events": 2, "uncited_events": 1, "invalid_citations": 1,
                                          "empty_handled": None}
    assert news.citation_check({"events": [], "coverage": "none"}, 0)["empty_handled"] is True
    assert news.citation_check({"events": [], "coverage": "thin"}, 0)["empty_handled"] is False


def test_schema_and_instructions_forbid_advice():
    assert set(news.SCHEMA["required"]) == set(GOOD)
    assert "no advice" in news.INSTRUCTIONS and "ignore them" in news.INSTRUCTIONS


def test_bedrock_model_names_use_the_mantle_client(monkeypatch):
    built = {}

    class Messages:
        def create(self, **kw):
            built["model"] = kw["model"]
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(GOOD))],
                                   stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    def mantle(**kw):
        built["client_kwargs"] = kw
        return SimpleNamespace(messages=Messages())

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AnthropicBedrockMantle=mantle,
                                                                 Anthropic=lambda **_: None))
    monkeypatch.setenv("AWS_PROFILE", "research")
    news.digest("X", H, "bedrock/us.anthropic.claude-haiku-5-5")
    assert built["model"] == "us.anthropic.claude-haiku-5-5"
    assert built["client_kwargs"] == {"aws_region": "us-east-1", "aws_profile": "research"}
