"""parser 单测：fixture 来自 2026-09-06 真实响应裁剪（1 视频 + 1 图集 + 1 残缺 + 1 无ID）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trend.parser import parse_file, parse_payload

FIXTURE = Path(__file__).parent / "fixtures" / "2026-09-06_1001.json"


@pytest.fixture()
def records():
    return parse_file(FIXTURE)


def test_fixture_shape(records):
    # 4 条原始 obj：视频 + 图集 + 残缺 + 无ID → 有效记录 3 条
    assert len(records) == 3


def test_video_full_mapping(records):
    video = records[0]
    assert video.source == "wellbyte"
    assert video.platform == "douyin"
    assert video.aweme_id == "7681813021265751398"
    assert video.title == "洗个奶瓶一转头…. #五个月的宝宝"
    assert video.author == "嘻嘻的小时候"
    # 响应中不存在的字段必须是 None，绝不能编造
    assert video.author_id is None
    assert video.share_url is None
    assert video.stats.comments is None
    assert video.stats.shares is None
    assert video.music.id is None
    assert video.music.title is None
    assert video.music.url is None
    # 真实存在的统计字段
    assert video.stats.likes == 931464
    assert video.stats.views == 21292308
    # 直链
    assert video.download_url and video.download_url.startswith("https://")
    # extras 真实字段
    assert video.extras["duration_ms"] == 6467
    assert video.extras["media_type"] == 4
    assert video.extras["image_cnt"] == 0
    assert video.extras["publish_time"] == 1788561473
    assert video.extras["score"] == 1432514
    # trend：榜单号 + 数组顺序即 rank
    assert video.trend.lists == ["1001"]
    assert video.trend.rank == {"1001": 1}
    assert video.trend.date_window == 24
    # raw 完整保留
    assert video.raw["item_id"] == video.aweme_id
    assert video.raw["nick_name"] == video.author


def test_image_post_mapping(records):
    image = records[1]
    assert image.aweme_id == "7681645586118682619"
    assert image.extras["media_type"] == 2
    assert image.extras["image_cnt"] and image.extras["image_cnt"] > 0
    assert image.extras["duration_ms"] == 0
    assert image.trend.rank == {"1001": 2}


def test_partial_item_missing_fields_become_none(records):
    partial = records[2]
    assert partial.aweme_id == "999partial0001"
    assert partial.title is None
    assert partial.author is None
    assert partial.stats.likes is None
    assert partial.stats.views is None
    assert partial.trend.rank == {"1001": 3}


def test_item_without_id_dropped(records):
    ids = [r.aweme_id for r in records]
    assert all(ids)


def test_rank_is_array_order():
    payload = {
        "code": 0,
        "data": {"objs": [{"item_id": str(i)} for i in range(1, 4)], "page": {}},
        "meta": {},
    }
    recs = parse_payload(payload, sub_type=1005)
    assert [r.trend.rank["1005"] for r in recs] == [1, 2, 3]


def test_empty_objs_returns_empty():
    assert parse_payload({"code": 0, "data": {"objs": [], "page": {}}}, sub_type=1001) == []
    assert parse_payload({"code": 0}, sub_type=1001) == []


def test_to_dict_roundtrip(records):
    d = records[0].to_dict()
    assert d["stats"]["likes"] == 931464
    assert d["trend"]["rank"] == {"1001": 1}
    json.dumps(d, ensure_ascii=False)  # 可序列化


def test_parse_file_sub_type_from_filename(tmp_path):
    p = tmp_path / "2026-09-07_1003.json"
    p.write_text(json.dumps({"code": 0, "data": {"objs": [{"item_id": "42"}], "page": {}}}), encoding="utf-8")
    recs = parse_file(p)
    assert recs[0].trend.lists == ["1003"]
    with pytest.raises(ValueError):
        parse_file(tmp_path / "badname.json")
