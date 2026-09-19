# -*- coding: utf-8 -*-
"""ReGen 生成阶段：故事外壳缺段（S1/S3）的 H3 take 生产。

复用生产链底盘（serve 生命周期/官方 wire 契约/预算纪律）。S2=take_001
不重生成。人物卡=故事实例自身（无参考特定约束——身体条件等只是实例）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.manifest import json_hash
from src.agentic_video.ragen_director import ReGenBlocked

SECTION_TAKE_PROMPT_TMPL = """subject_definitions:
<Subject 1> is the {role_label}: {protagonist_desc}.
Other people present may react to <Subject 1>; keep them secondary.

summary:
[reference generation] Generate one coherent {seconds}-second clip of the
{role_label} for a short video. Preserve <Subject 1> throughout.

retention_analysis:
<Subject 1> (throughout): fully_preserved - identity, wardrobe, and
consistency across the clip.
Setting (throughout): fully_preserved - one coherent location.

detailed_description:
[Shot 1] At 00:00.000, {event_desc}
{extra}

overall_soundscape:
Natural ambient sound consistent with the depicted activity.

non_diegetic_music:
None."""


def build_section_take_request(story: dict[str, Any], role: str,
                               contract: dict[str, Any], *,
                               seconds: float,
                               seed: int) -> dict[str, Any]:
    """故事段 → 六段 wire 请求（纯 CPU，人物/事件全来自故事实例）。"""
    from src.agentic_video.generation_v9g import build_h3_request
    if role == "situation_setup":
        role_label = "underestimated protagonist"
        belief = story.get("initial_belief") or {}
        protagonist_desc = f"person who others underestimate due to " \
                           f"{belief.get('source_of_underestimation', 'circumstances')}"
        event_desc = (f"<Subject 1> is seen in a way that makes others form "
                      f"the belief: {belief.get('claim', 'they are not capable')}. "
                      f"The underestimation must be visually readable without "
                      f"narration.")
        extra = "Keep the pace slow enough for the premise to land."
    elif role == "evidence_expansion":
        role_label = "protagonist after the surprise"
        protagonist_desc = "the same protagonist, now shown beyond a single label"
        events = [row.get("event") for row in story.get("reinforcement") or []
                  if row.get("event")]
        ending = story.get("ending") or {}
        event_desc = "; ".join(str(e) for e in events[:2]) or \
            "additional visible evidence that the corrected belief holds"
        extra = (f"End with a light humanizing beat: {ending.get('statement', "
                 f"'a small relatable quirk')}. Do not reverse the corrected "
                 f"belief.")
    else:
        raise ReGenBlocked("ragen_generate", "role_not_generatable", role)
    prompt = SECTION_TAKE_PROMPT_TMPL.format(
        role_label=role_label, protagonist_desc=protagonist_desc,
        seconds=f"{seconds:g}", event_desc=event_desc, extra=extra)
    request = build_h3_request(
        prompt=prompt, duration_s=float(seconds),
        references=None, first_frame=None, last_frame=None,
        capabilities=None, seed=int(seed), aspect_ratio="3:4")
    request["_ragen"] = {"role": role, "story_id": story.get("story_id"),
                         "seconds": float(seconds)}
    return request


def run_section_takes(cfg: Any, output_dir: Path, *, story: dict[str, Any],
                      contract: dict[str, Any], endpoints: str,
                      gpu_set: str, h3_python_bin: str | None = None,
                      seconds: float = 10.0, seeds: tuple[int, ...] = (2001, 2003),
                      manage_server: bool = True) -> dict[str, Any]:
    """S1/S3 两条 take 并行（每端点一实例）。S2 复用 take_001 不在此生成。"""
    import queue
    import threading

    from src.agentic_video.generation_long_take import (
        _ensure_h3_server, _ffprobe_bin_of, _stop_owned_server)
    from src.agentic_video.generation_v9g import SGLangH3Client, V9GBlocked
    from src.perception import common

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = ["situation_setup", "evidence_expansion"]
    jobs = [{"role": role, "seed": seed,
             "request": build_section_take_request(
                 story, role, contract, seconds=seconds, seed=seed)}
            for role, seed in zip(roles, seeds)]
    _write = output_dir / "section_takes.json"
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    eps = [e.strip().rstrip("/") for e in endpoints.split(",") if e.strip()]
    indices = [i for i in gpu_set.split(",") if i.strip()]
    if len(indices) != 2 * len(eps):
        raise ReGenBlocked("ragen_generate", "gpu_endpoint_mismatch")
    pairs = [",".join(indices[i:i + 2]) for i in range(0, len(indices), 2)]
    repo_root = Path(__file__).resolve().parents[2]
    servers = []
    try:
        for endpoint, pair in zip(eps, pairs):
            port = endpoint.rsplit(":", 1)[-1]
            servers.append(_ensure_h3_server(
                endpoint, backend="diffusers", variant="unified",
                manage_server=manage_server, port=int(port),
                repo_root=repo_root,
                log_path=output_dir / f"h3_server_{port}.log",
                gpu_set=pair, python_bin=h3_python_bin))
        work: "queue.Queue[dict[str, Any]]" = queue.Queue()
        for job in jobs:
            work.put(job)

        def lane(endpoint: str) -> None:
            client = SGLangH3Client(endpoint)
            while True:
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                take_id = f"{job['role']}_s{job['seed']}"
                take_dir = output_dir / take_id
                take_dir.mkdir(parents=True, exist_ok=True)
                state = {"take_id": take_id, "role": job["role"],
                         "state": "planned"}
                try:
                    (take_dir / "request.json").write_text(
                        json.dumps(job["request"], ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    payload = client.serialize_payload(job["request"])
                    created = client.submit(payload)
                    state.update({"state": "submitted",
                                  "job_id": created.get("id")})
                    client.poll(str(created["id"]))
                    media = client.download(str(created["id"]),
                                            take_dir / "take.mp4")
                    state.update({"state": "transport_validated", **media})
                except V9GBlocked as exc:
                    state.update({"state": "failed",
                                  "reason_code": exc.reason_code,
                                  "detail": str(exc)[:300]})
                with lock:
                    results.append(state)

        threads = [threading.Thread(target=lane, args=(e,)) for e in eps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for server in servers:
            _stop_owned_server(server)
    result = {"schema_version": "ragen_section_takes_v1",
              "story_id": story.get("story_id"), "takes": results}
    _write.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return result
