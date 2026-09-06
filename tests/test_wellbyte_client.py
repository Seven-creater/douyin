"""wellbyte_client 单测：全部 mock HTTP，不打真实 API、不花 credits。"""
from __future__ import annotations

import json

import requests

from src.trend.wellbyte_client import FetchOutcome, WellbyteClient, WellbyteApiError

API_KEY = "dg_live_test_key_000"
BASE = "https://api.example.test"
ENDPOINT = "/v1/douyin/billboard/fetch_hot_total_video_list"
PARAMS = {"date_window": 24, "page": 1, "page_size": 20, "keyword": "", "tags": []}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        if payload is not None:
            self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        else:
            self.content = text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, script):
        self.script = list(script)  # 每项: FakeResponse 或 Exception
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(session, max_attempts=3, backoff=0.0):
    return WellbyteClient(
        API_KEY, base_url=BASE, endpoint=ENDPOINT,
        timeout=5, max_attempts=max_attempts, backoff_seconds=backoff, session=session,
    )


def ok_payload(n=2):
    return {
        "code": 0, "message": "ok",
        "data": {"objs": [{"id": str(i)} for i in range(n)], "page": {"total": n}},
        "meta": {"request_id": "req_1", "credits_charged": 2, "cache_hit": False},
    }


def test_request_assembly(tmp_path):
    s = FakeSession([FakeResponse(payload=ok_payload())])
    c = make_client(s)
    c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert len(s.calls) == 1
    call = s.calls[0]
    assert call["url"] == BASE + ENDPOINT
    assert call["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert call["headers"]["Content-Type"] == "application/json"
    body = call["json"]
    assert body["sub_type"] == 1001
    for k, v in PARAMS.items():
        assert body[k] == v


def test_success_writes_raw_bytes_verbatim(tmp_path):
    payload = ok_payload(n=20)
    s = FakeSession([FakeResponse(payload=payload)])
    c = make_client(s)
    out = tmp_path / "2026-09-06_1001.json"
    oc = c.fetch_and_save(1001, PARAMS, out)
    assert oc.status == "fetched"
    assert oc.video_count == 20
    assert oc.api_code == 0
    assert oc.meta["credits_charged"] == 2
    # 字节级一致：落盘内容 = 响应原始字节
    assert out.read_bytes() == json.dumps(payload, ensure_ascii=False).encode("utf-8")


def test_existing_file_skips_request(tmp_path):
    out = tmp_path / "2026-09-06_1001.json"
    out.write_text(json.dumps(ok_payload(3), ensure_ascii=False), encoding="utf-8")
    s = FakeSession([])
    c = make_client(s)
    oc = c.fetch_and_save(1001, PARAMS, out)
    assert oc.status == "skipped_existing"
    assert oc.video_count == 3
    assert s.calls == []  # 完全没有发请求


def test_auth_error_fails_fast_no_retry(tmp_path):
    bad = {"code": 40101, "message": "Invalid or revoked API Key", "data": None, "meta": {}}
    s = FakeSession([FakeResponse(status_code=401, payload=bad)])
    c = make_client(s)
    oc = c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert oc.status == "failed"
    assert oc.api_code == 40101
    assert oc.kind == "auth"
    assert len(s.calls) == 1  # 非重试类只打一次


def test_rate_limit_retries_then_succeeds(tmp_path):
    limited = {"code": 42901, "message": "rate limit", "data": None, "meta": {}}
    s = FakeSession([FakeResponse(status_code=429, payload=limited), FakeResponse(payload=ok_payload())])
    c = make_client(s, backoff=0.0)
    oc = c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert oc.status == "fetched"
    assert len(s.calls) == 2


def test_network_errors_exhaust_retries(tmp_path):
    s = FakeSession([requests.ConnectionError("boom")] * 3)
    c = make_client(s, backoff=0.0)
    oc = c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert oc.status == "failed"
    assert "ConnectionError" in (oc.error or "")
    assert len(s.calls) == 3


def test_non_json_200_fails(tmp_path):
    # 网关返回垃圾 HTML 属瞬态错误：可重试；脚本给足 3 次的量
    s = FakeSession([FakeResponse(status_code=200, text="<html>gateway</html>")] * 3)
    c = make_client(s, backoff=0.0)
    oc = c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert oc.status == "failed"
    assert not (tmp_path / "a.json").exists()  # 失败不落盘


def test_error_never_contains_api_key(tmp_path):
    # 网关把请求回显的极端场景：错误信息里也不能带出 key
    s = FakeSession([FakeResponse(status_code=502, text=f"bad gateway saw {API_KEY}")])
    c = make_client(s, backoff=0.0)
    oc = c.fetch_and_save(1001, PARAMS, tmp_path / "a.json")
    assert oc.status == "failed"
    assert API_KEY not in (oc.error or "")


def test_wellbyte_api_error_kind_mapping():
    e = WellbyteApiError(40201, "insufficient balance")
    assert e.kind == "quota"
    e2 = WellbyteApiError(99999, "weird")
    assert e2.kind == "unknown"
