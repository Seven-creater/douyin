"""配置加载、路径解析、密钥管理与日志初始化。

- 所有相对路径基于 repo_root()（仓库根）解析为绝对路径，保证本地 Windows 与服务器 Linux 通用
- 密钥只从环境变量（.env）读取，绝不写入 git/配置文件/日志
- SecretMaskFilter 挂到所有日志 handler，密钥即使意外进入日志也会被脱敏
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REDACTED = "***REDACTED***"


class ConfigError(RuntimeError):
    """配置缺失或非法。"""


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_utf8_stdio() -> None:
    """Windows 控制台默认 GBK，中文/emoji 会炸；统一改为 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - 非终端环境（如重定向）可能不支持
                pass


def _abs_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else repo_root() / pp


@dataclass(frozen=True)
class WellbyteCfg:
    base_url: str
    endpoint: str
    timeout_seconds: float
    sub_types: list[int]
    sub_type_names: dict[int, str]
    request_params: dict[str, Any]
    retry_max_attempts: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class PathsCfg:
    raw_dir: Path
    processed_dir: Path
    videos_dir: Path
    logs_dir: Path
    perception_dir: Path


@dataclass(frozen=True)
class AppConfig:
    wellbyte: WellbyteCfg
    paths: PathsCfg
    ranking: dict[str, Any]
    download: dict[str, Any]
    perception: dict[str, Any]
    template: dict[str, Any]
    logging_level: str


def load_config(path: Path | None = None) -> AppConfig:
    """加载 config/default.yaml（或指定路径），相对路径绝对化为仓库根下。"""
    cfg_path = _abs_path(path) if path else repo_root() / "config" / "default.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"配置文件不存在: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    wb = raw.get("wellbyte") or {}
    retry = wb.get("retry") or {}
    paths = raw.get("paths") or {}
    cfg = AppConfig(
        wellbyte=WellbyteCfg(
            base_url=wb.get("base_url", "https://api.wellbyte.net"),
            endpoint=wb["endpoint"],
            timeout_seconds=float(wb.get("timeout_seconds", 30)),
            sub_types=list(wb.get("sub_types", [])),
            sub_type_names={int(k): str(v) for k, v in (wb.get("sub_type_names") or {}).items()},
            request_params=dict(wb.get("request_params") or {}),
            retry_max_attempts=int(retry.get("max_attempts", 3)),
            retry_backoff_seconds=float(retry.get("backoff_seconds", 5)),
        ),
        paths=PathsCfg(
            raw_dir=_abs_path(paths["raw_dir"]),
            processed_dir=_abs_path(paths["processed_dir"]),
            videos_dir=_abs_path(paths["videos_dir"]),
            logs_dir=_abs_path(paths["logs_dir"]),
            perception_dir=_abs_path(paths.get("perception_dir", "data/perception")),
        ),
        ranking=dict(raw.get("ranking") or {}),
        download=dict(raw.get("download") or {}),
        perception=dict(raw.get("perception") or {}),
        template=dict(raw.get("template") or {}),
        logging_level=str((raw.get("logging") or {}).get("level", "INFO")).upper(),
    )
    for p in cfg.paths.__dict__.values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


def require_api_key(env: Mapping[str, str] | None = None) -> str:
    """从环境变量读取 WELLBYTE_API_KEY；缺失/占位符 → 人话报错。同时顺手加载 .env。"""
    if env is None:
        load_dotenv(repo_root() / ".env")
        env = os.environ
    key = (env.get("WELLBYTE_API_KEY") or "").strip()
    if not key or key == "your_key_here":
        raise ConfigError(
            "WELLBYTE_API_KEY 未设置：请复制 .env.example 为 .env 并填写真实密钥（.env 已 gitignore）"
        )
    return key


class SecretMaskFilter(logging.Filter):
    """把密钥字符串从日志消息中替换为 ***REDACTED***，挂到所有 handler。"""

    def __init__(self, secrets: list[str] | tuple[str, ...] = ()):
        super().__init__()
        self._secrets = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        msg = record.getMessage()
        redacted = msg
        for s in self._secrets:
            redacted = redacted.replace(s, _REDACTED)
        if redacted != msg:
            record.msg = redacted
            record.args = None
        return True


def setup_logging(logs_dir: Path, level: str = "INFO", secrets: list[str] | None = None,
                  filename_prefix: str = "collect_trends") -> Path:
    """控制台 + 按次轮转文件双 handler，全部挂密钥脱敏。返回日志文件路径。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
    mask = SecretMaskFilter(secrets or [])

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(mask)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(mask)
    root.addHandler(file_handler)
    return log_path
