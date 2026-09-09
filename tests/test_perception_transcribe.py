"""transcribe_audio 单测（fake funasr 返回，不打模型）。"""
from __future__ import annotations

from src.perception.transcribe_audio import (extract_tags, parse_sensevoice_result,
                                              transcribe_with_model)


def fake_raw():
    return {
        "text": "<|BGM|><|zh|><|NEUTRAL|>家人们看好了，<|LAUGHTER|>今天教你们水上飞人。"
                "<|APPLAUSE|>记得点赞关注。",
        "sentence_info": [
            {"start": 0, "end": 1520, "text": "家人们看好了，"},
            {"start": 1520, "end": 4210, "text": "今天教你们水上飞人。<|LAUGHTER|>"},
            {"start": 4210, "end": 5900, "text": "记得点赞关注。<|APPLAUSE|>"},
        ],
    }


def test_extract_tags():
    events, emotions = extract_tags(fake_raw()["text"])
    assert events == ["APPLAUSE", "BGM", "LAUGHTER"]
    assert emotions == ["NEUTRAL"]


def test_parse_full_with_postprocess():
    out = parse_sensevoice_result(fake_raw(), postprocess=lambda s: s.replace("<|", "").replace("|>", ""))
    assert "家人们看好了" in out["full_text"]
    assert out["audio_events"] == ["APPLAUSE", "BGM", "LAUGHTER"]
    assert out["segments_available"] is True
    assert len(out["segments"]) == 3
    assert out["segments"][0] == {"start_ms": 0, "end_ms": 1520, "text": "家人们看好了，"}
    # 段内标签也要清干净
    assert "LAUGHTER" not in out["segments"][1]["text"]
    assert out["segments"][2]["end_ms"] == 5900


def test_parse_without_postprocess_strips_tags():
    out = parse_sensevoice_result(fake_raw(), postprocess=None)
    assert "<|" not in out["full_text"]
    assert "水上飞人" in out["full_text"]


def test_parse_no_sentence_info_degrades():
    raw = {"text": "<|BGM|><|zh|>纯音乐视频"}
    out = parse_sensevoice_result(raw)
    assert out["segments"] == [] and out["segments_available"] is False
    assert out["audio_events"] == ["BGM"]
    assert out["full_text"] == "纯音乐视频"


def test_parse_empty_text():
    out = parse_sensevoice_result({"text": ""})
    assert out["full_text"] == "" and out["audio_events"] == []


def test_withitn_filtered_and_emoji_stripped():
    # 真实踩坑：WITHITN（ITN 处理标记）混入事件；postprocess 会把标签转成 emoji 留在文本
    raw = {"text": "<|BGM|><|zh|><|WITHITN|>🎼我就是鸡蛋。😊"}
    out = parse_sensevoice_result(raw, postprocess=lambda s: s)  # postprocess 原样回显
    assert out["audio_events"] == ["BGM"]          # WITHITN 被过滤
    assert out["full_text"] == "我就是鸡蛋。"        # 🎼😊 被剥离


def test_transcribe_requests_sentence_timestamps(tmp_path):
    class FakeModel:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [{"text": "<|ja|>こんにちは", "sentence_info": [
                {"start": 100, "end": 900, "text": "こんにちは"},
            ]}]

    model = FakeModel()
    output = transcribe_with_model(model, tmp_path / "reference.mp4", language="ja")

    assert model.kwargs["sentence_timestamp"] is True
    assert output["segments"] == [
        {"start_ms": 100, "end_ms": 900, "text": "こんにちは"},
    ]
