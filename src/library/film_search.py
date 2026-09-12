"""Film Knowledge Bootstrap 的外部检索层（P1.5，可插拔，默认禁用）。

SearchProvider 协议 + HTTP 实现（照 wellbyte_client 先例：requests + Bearer +
超时）。无 provider/无 key 时 get_provider 返回 None——三档策略安全降级为
纯感知（search 档降 model_prior_unverified：title/实体都不进任何 prompt）。
产品模式有 provider 时应主动使用（coding agent 不会明知 README 写着 PyTorch
还硬看代码猜项目）；benchmark 模式用 --mask-metadata 控制泄漏源。
"""
from __future__ import annotations

import os


class HttpSearchProvider:
    """通用搜索客户端：endpoint + Bearer key，返回归一化的 {title,url,snippet}。"""

    def __init__(self, endpoint: str, api_key: str, *, timeout_s: float = 10.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = timeout_s

    def search(self, query: str) -> list[dict]:
        import requests

        response = requests.get(
            self.endpoint, params={"q": query, "count": 5},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        rows = data.get("results") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return []
        return [{"title": str(row.get("title") or "")[:120],
                 "url": str(row.get("url") or "")[:300],
                 "snippet": str(row.get("snippet") or "")[:300]}
                for row in rows if isinstance(row, dict)][:5]


def get_provider(cfg) -> HttpSearchProvider | None:
    """配置驱动：library.film_bootstrap.search。provider=none 或缺 key/endpoint
    → None（调用方按降级链处理，绝不因搜索缺失崩掉 bootstrap）。"""
    bootstrap_cfg = (cfg.library.get("film_bootstrap") or {}) \
        if hasattr(cfg, "library") else {}
    search_cfg = bootstrap_cfg.get("search") or {}
    if not search_cfg.get("enabled") \
            or str(search_cfg.get("provider") or "none") == "none":
        return None
    key = os.environ.get(str(search_cfg.get("key_env") or "FILM_SEARCH_API_KEY"), "")
    endpoint = str(search_cfg.get("endpoint") or "")
    if not (key and endpoint):
        return None
    return HttpSearchProvider(endpoint, key,
                              timeout_s=float(search_cfg.get("timeout_s", 10)))
