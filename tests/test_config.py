"""src.config 单测：路径绝对化、密钥校验、日志脱敏。不打任何真实 API。"""
from __future__ import annotations

import logging

import pytest

from src.config import (
    ConfigError,
    SecretMaskFilter,
    load_config,
    repo_root,
    require_api_key,
)


def test_repo_root_contains_readme():
    assert (repo_root() / "README.md").exists()
    assert (repo_root() / "config" / "default.yaml").exists()


def test_load_config_paths_absolute_and_under_repo():
    cfg = load_config()
    for p in (cfg.paths.raw_dir, cfg.paths.processed_dir, cfg.paths.videos_dir, cfg.paths.logs_dir):
        assert p.is_absolute()
        assert repo_root() in p.parents


def test_load_config_wellbyte_fields():
    cfg = load_config()
    assert cfg.wellbyte.sub_types == [1001, 1002, 1003, 1004, 1005]
    assert cfg.wellbyte.request_params["date_window"] == 24
    assert cfg.wellbyte.request_params["page_size"] == 20
    assert cfg.ranking["top_n"] == 20


def test_require_api_key_missing_raises():
    with pytest.raises(ConfigError):
        require_api_key({})


def test_require_api_key_placeholder_raises():
    with pytest.raises(ConfigError):
        require_api_key({"WELLBYTE_API_KEY": "your_key_here"})


def test_require_api_key_ok():
    assert require_api_key({"WELLBYTE_API_KEY": "sk-test-123"}) == "sk-test-123"


def test_secret_mask_filter_redacts():
    filt = SecretMaskFilter(["dg_live_SECRET"])
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="request failed with key dg_live_SECRET in header", args=None, exc_info=None,
    )
    assert filt.filter(record) is True
    assert "dg_live_SECRET" not in record.getMessage()
    assert "***REDACTED***" in record.getMessage()


def test_secret_mask_filter_no_secrets_passthrough():
    filt = SecretMaskFilter([])
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="plain message", args=None, exc_info=None,
    )
    assert filt.filter(record) is True
    assert record.getMessage() == "plain message"
