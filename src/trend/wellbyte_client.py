"""Wellbyte 抖音榜单客户端：只负责发请求 + 响应字节原样落盘，不理解业务字段。

官方信封格式（https://www.wellbyte.net/zh/docs）：
    {code, message, data: {objs: [...], page: {...}}, meta: {...}}
    - code==0 且 HTTP 200 才算成功；失败不计费
    - meta 含 request_id / credits_charged / cache_hit / elapsed_ms

本模块不做字段映射（parser 的职责），不打日志输出密钥，
当日 raw 文件已存在时默认跳过请求（省 credits）。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import requests

logger = logging.getLogger(__name__)

# 官方错误码 → 语义分类（https://www.wellbyte.net/zh/docs/errors，
# 接口页与全局表在 40101/40102、40401/40402 上略有出入，两类都收录）
_KIND_BY_CODE = {
    40001: "param",
    40101: "auth", 40102: "auth", 40301: "auth",
    40201: "quota",
    40401: "not_found", 40402: "not_found",
    42201: "quality",
    42901: "rate_limit",
    50001: "server", 50301: "server", 50302: "server",
    50201: "upstream", 50202: "upstream", 50401: "upstream",
}

# 只有这些分类值得重试；auth/quota/param/not_found/quality 立即失败
_RETRYABLE_KINDS = {"rate_limit", "upstream", "server"}


class WellbyteApiError(RuntimeError):
    """Wellbyte 返回非成功信封。message 中绝不含 API key。"""

    def __init__(self, code: int, message: str, http_status: int | None = None):
        self.code = code
        self.message = message
        self.http_status = http_status
        self.kind = _KIND_BY_CODE.get(code, "unknown")
        super().__init__(f"[wellbyte] code={code} http={http_status} kind={self.kind} message={message}")


@dataclass
class FetchOutcome:
    """单榜单一次抓取的结果（成功/跳过/失败三态）。"""

    sub_type: int
    status: str  # "fetched" | "skipped_existing" | "failed"
    http_status: int | None = None
    api_code: int | None = None
    message: str = ""
    raw_path: Path | None = None
    video_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def kind(self) -> str | None:
        return _KIND_BY_CODE.get(self.api_code, "unknown") if self.api_code is not None else None


def _envelope_error(resp: requests.Response) -> WellbyteApiError:
    """从失败响应（非 200 或 code!=0）构造 WellbyteApiError，尽力解析信封。"""
    code, message = 0, resp.text[:200] if resp.text else ""
    try:
        payload = resp.json()
        code = int(payload.get("code") or 0)
        message = str(payload.get("message") or message)
    except ValueError:
        pass
    return WellbyteApiError(code or resp.status_code * 100, message, resp.status_code)


def _count_objs(payload: Mapping[str, Any]) -> int:
    data = payload.get("data")
    if isinstance(data, Mapping):
        objs = data.get("objs")
        if isinstance(objs, list):
            return len(objs)
    return 0


class WellbyteClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        endpoint: str,
        timeout: float = 30.0,
        max_attempts: int = 3,
        backoff_seconds: float = 5.0,
        session: requests.Session | None = None,
    ):
        self._api_key = api_key
        self._url = base_url.rstrip("/") + endpoint
        self._timeout = timeout
        self._max_attempts = max(1, int(max_attempts))
        self._backoff = float(backoff_seconds)
        self._session = session or requests.Session()

    def _redact(self, text: str) -> str:
        """把密钥从任意错误文本中抹掉（防网关/异常信息回显）。"""
        return text.replace(self._api_key, "***REDACTED***") if self._api_key else text

    def fetch_billboard_once(self, sub_type: int, params: Mapping[str, Any]) -> requests.Response:
        """发一次请求（不重试不落盘）。sub_type 注入 body，其余参数原样透传。"""
        body = dict(params)
        body["sub_type"] = int(sub_type)
        return self._session.post(
            self._url,
            json=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )

    def fetch_and_save(
        self,
        sub_type: int,
        params: Mapping[str, Any],
        out_path: Path,
        *,
        overwrite: bool = False,
    ) -> FetchOutcome:
        """抓一个榜单并把成功响应字节原样写入 out_path。

        - out_path 已存在且非 overwrite → skipped_existing（不花 credits）
        - 成功 = HTTP 200 且 code==0，写 resp.content 原始字节
        - 网络错误 / 限流 42901 / 上游 5xx → 退避重试；auth/quota 等立即失败
        - 任何失败返回 status="failed"（由上层汇总，不抛异常中断其它榜单）
        """
        if out_path.exists() and not overwrite:
            count = 0
            try:
                count = _count_objs(json.loads(out_path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass
            logger.info("[wellbyte %s] 当日 raw 已存在，跳过请求（省 credits）: %s", sub_type, out_path.name)
            return FetchOutcome(
                sub_type=sub_type, status="skipped_existing", raw_path=out_path,
                video_count=count, message="raw exists",
            )

        last_error: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self.fetch_billboard_once(sub_type, params)
            except requests.RequestException as exc:
                last_error = self._redact(f"network: {type(exc).__name__}: {exc}")
                logger.warning("[wellbyte %s] 第 %d/%d 次尝试网络错误: %s", sub_type, attempt, self._max_attempts, last_error)
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    last_error = self._redact(f"invalid json: {exc}")
                    logger.warning("[wellbyte %s] 第 %d/%d 次尝试响应非 JSON: %s", sub_type, attempt, self._max_attempts, last_error)
                    self._sleep_backoff(attempt)
                    continue
                if payload.get("code") == 0:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(resp.content)  # 字节原样，不做任何 reshape
                    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                    count = _count_objs(payload)
                    logger.info(
                        "[wellbyte %s] 成功: %d 条视频, credits=%s, cache_hit=%s, request_id=%s",
                        sub_type, count, meta.get("credits_charged"), meta.get("cache_hit"), meta.get("request_id"),
                    )
                    return FetchOutcome(
                        sub_type=sub_type, status="fetched", http_status=200, api_code=0,
                        message="ok", raw_path=out_path, video_count=count, meta=dict(meta),
                    )
                err = WellbyteApiError(int(payload.get("code") or 0), str(payload.get("message") or ""), 200)
            else:
                err = _envelope_error(resp)

            last_error = self._redact(str(err))
            if err.kind in _RETRYABLE_KINDS and attempt < self._max_attempts:
                logger.warning("[wellbyte %s] 第 %d/%d 次尝试失败（%s），退避后重试", sub_type, attempt, self._max_attempts, last_error)
                self._sleep_backoff(attempt)
                continue
            logger.error("[wellbyte %s] 失败: %s", sub_type, last_error)
            return FetchOutcome(
                sub_type=sub_type, status="failed", http_status=err.http_status,
                api_code=err.code, message=self._redact(err.message), error=last_error,
            )

        logger.error("[wellbyte %s] 重试耗尽: %s", sub_type, last_error)
        return FetchOutcome(sub_type=sub_type, status="failed", error=last_error)

    def _sleep_backoff(self, attempt: int) -> None:
        time.sleep(self._backoff * attempt)
