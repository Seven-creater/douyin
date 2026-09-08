"""fetch_context：外部上下文采集（标题/作者/相关推荐/热评）——识别热梗真实核心的信号源。

CLI：python -m src.perception.fetch_context --aweme-id <id> [--share-url URL] [--force]

产物：data/perception/<id>/context/result.json

背景（2026-09-08 实测）：抖音分享页/桌面版匿名都不出评论（老 API 参数不合法、
桌面版无内容、分享页不调评论接口）；分享页在移动 UA + 匿名 ttwid 下可正常渲染，
滚动后底部有「相关推荐」列表（同款系列视频的标题/作者/点赞）——标题与相关列表
是识别「模仿对象/IP」（如 王者荣耀·安琪拉）的关键证据，评论区需登录 Cookie
（.env 里配 DOUYIN_COOKIE 后自动尝试桌面版热评）。

只在本地跑（服务器防火墙屏蔽抖音全域），产物 scp 到服务器同路径即可参与模板抽取。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

from src.config import ensure_utf8_stdio, load_config, setup_logging
from src.perception import common

logger = logging.getLogger(__name__)

UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1")
UA_DESK = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

_MAX_RELATED = 20
_MAX_COMMENTS = 30


def register_ttwid() -> str | None:
    """匿名注册 ttwid（绕过分享页'抱歉出错了'风控的关键，2026-09-08 实测）。"""
    body = {"region": "cn", "aid": 6383, "needFid": False, "service": "www.ixigua.com",
            "migrate_info": {"ticket": "", "source": "node"},
            "cbUrlProtocol": "https", "union": True}
    try:
        r = requests.post("https://ttwid.bytedance.com/ttwid/union/register/",
                          json=body, timeout=15)
        return r.cookies.get("ttwid")
    except requests.RequestException as exc:
        logger.warning("[context] ttwid 注册失败：%s", exc)
        return None


# 相关条目：分享页相关推荐不是 <a> 链接（2026-09-08 实测），是「标题(#话题) → 作者 → 点赞数」
# 三行一组的文本块，直接对页面行做状态机解析
_JS_MAIN = """
() => {
  const t = document.body.innerText.split('\\n').map(s => s.trim()).filter(Boolean);
  const at = t.find(s => s.startsWith('@')) || '';
  return {lines: t.slice(0, 160), author: at};
}
"""

_LIKE_RE = re.compile(r"^(\d+(?:\.\d+)?[万亿]?\+?)$")


def _parse_related(lines: list[str], own_title_hint: str) -> list[dict]:
    """「含#的长行=标题，后随短行=作者，再后数字行=点赞」三行一组；主视频标题（含'展开'）跳过。"""
    related: list[dict] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines) and len(related) < _MAX_RELATED:
        line = lines[i]
        if "#" in line and len(line) >= 6 and "展开" not in line \
                and (not own_title_hint or own_title_hint[:12] not in line):
            author, likes = "", ""
            for nxt in lines[i + 1:i + 4]:
                if _LIKE_RE.match(nxt) and not author:
                    likes = nxt
                elif 1 < len(nxt) <= 20 and "#" not in nxt and not nxt.isdigit() and not author:
                    author = nxt
                if author and likes:
                    break
            key = line[:30]
            if key not in seen:
                seen.add(key)
                related.append({"aweme_id": "", "title": line[:80], "author": author,
                                "likes": likes, "raw": ""})
            i += 3
            continue
        i += 1
    return related


async def _scrape(aweme_id: str, share_url: str, cookie: str | None) -> dict:
    from playwright.async_api import async_playwright

    comment_hits: list[dict] = []
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel="msedge")
        except Exception:  # 系统 Edge 缺失 → 回退自带 chromium
            browser = await p.chromium.launch(headless=True)

        ttwid = register_ttwid()

        # --- 1) 移动分享页：主信息 + 相关推荐 ---
        mob = await browser.new_context(
            user_agent=UA_MOBILE, viewport={"width": 390, "height": 844},
            is_mobile=True, has_touch=True, device_scale_factor=3, locale="zh-CN")
        if ttwid:
            for dom in (".iesdouyin.com", ".douyin.com"):
                await mob.add_cookies([{"name": "ttwid", "value": ttwid,
                                        "domain": dom, "path": "/"}])

        async def _on_resp(resp):
            if "comment/list" in resp.url:
                try:
                    j = await resp.json()
                    comment_hits.extend((j.get("data") or {}).get("comments") or [])
                except Exception:  # noqa: BLE001
                    pass

        mob.on("response", _on_resp)
        pg = await mob.new_page()
        await pg.goto(share_url, wait_until="domcontentloaded", timeout=60000)
        try:
            await pg.wait_for_selector("video", timeout=20000)
        except Exception:  # noqa: BLE001
            logger.warning("[context] 分享页未渲染出 video（风控？），尽力抓取")
        await pg.wait_for_timeout(3000)
        for _ in range(4):  # 滚动加载相关推荐
            await pg.mouse.wheel(0, 1400)
            await pg.wait_for_timeout(1200)
        main_raw = await pg.evaluate(_JS_MAIN)
        await mob.close()

        # --- 2) 桌面版热评（需要登录 Cookie）---
        comments: list[dict] = list(comment_hits)
        if cookie and not comments:
            desk = await browser.new_context(user_agent=UA_DESK, locale="zh-CN",
                                             viewport={"width": 1380, "height": 900})
            if ttwid:
                await desk.add_cookies([{"name": "ttwid", "value": ttwid,
                                         "domain": ".douyin.com", "path": "/"}])
            for kv in [c.strip() for c in cookie.split(";") if "=" in c][:40]:
                k, _, v = kv.partition("=")
                await desk.add_cookies([{"name": k.strip(), "value": v.strip(),
                                         "domain": ".douyin.com", "path": "/"}])
            desk.on("response", _on_resp)
            pg2 = await desk.new_page()
            try:
                await pg2.goto(f"https://www.douyin.com/video/{aweme_id}",
                               wait_until="domcontentloaded", timeout=60000)
                await pg2.wait_for_timeout(5000)
                for _ in range(3):
                    await pg2.mouse.wheel(0, 1600)
                    await pg2.wait_for_timeout(1500)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[context] 桌面版热评抓取失败：%s", exc)
            await desk.close()
            comments = comment_hits[:_MAX_COMMENTS]
        await browser.close()

    # --- 解析 ---
    lines = main_raw.get("lines") or []
    own_hint = ""
    for ln in lines[:12]:  # 主视频标题（含'展开'截断标记）仅用于排除
        if "#" in ln and "展开" in ln:
            own_hint = ln
            break
    related = _parse_related(lines, own_hint)

    return {
        "author": (main_raw.get("author") or "").lstrip("@"),
        "own_title_hint": own_hint[:80],
        "page_excerpt": " / ".join(lines[:30])[:1500],
        "related": related,
        "comments": [{"text": str(c.get("text", ""))[:160],
                      "digg_count": c.get("digg_count", 0)} for c in comments],
        "comment_source": ("douyin_web_cookie" if cookie else None),
    }


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    ap = argparse.ArgumentParser(description="外部上下文采集（标题/相关推荐/热评）")
    ap.add_argument("--aweme-id", required=True)
    ap.add_argument("--share-url", default=None, help="默认由 aweme_id 构造分享页")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)
    setup_logging(cfg.paths.logs_dir, cfg.logging_level, filename_prefix="fetch_context")

    tdir = common.tool_dir_for(cfg.paths.perception_dir, args.aweme_id, "context")
    if common.done_or_skip(tdir, params={"aweme_id": args.aweme_id}, force=args.force) is not None:
        common.emit_status_line("ok", cached=True, output=str(tdir / "result.json"))
        return 0

    share_url = args.share_url or f"https://www.iesdouyin.com/share/video/{args.aweme_id}/"
    cookie = os.environ.get("DOUYIN_COOKIE") or ""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        cookie = os.environ.get("DOUYIN_COOKIE") or cookie
    except ImportError:
        pass

    try:
        output = asyncio.run(_scrape(args.aweme_id, share_url, cookie or None))
    except ImportError:
        logger.error("缺 playwright：本地 pip install playwright 并保证系统 Edge 可用")
        common.emit_status_line("error", error="playwright not installed")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_context 失败")
        common.emit_status_line("error", error=str(exc)[:200])
        return 1

    common.write_result_json(tdir, tool="fetch_context", aweme_id=args.aweme_id,
                             params={"share_url": share_url,
                                     "with_cookie": bool(cookie)},
                             output=output)
    common.append_metric(cfg.paths.perception_dir / "metrics.jsonl",
                         tool="fetch_context", aweme_id=args.aweme_id, status="ok",
                         elapsed_s=0.0,
                         extra={"related": len(output["related"]),
                                "comments": len(output["comments"])})
    logger.info("[context] 相关视频 %d 条，热评 %d 条 → %s",
                len(output["related"]), len(output["comments"]), tdir / "result.json")
    common.emit_status_line("ok", related=len(output["related"]),
                            comments=len(output["comments"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
