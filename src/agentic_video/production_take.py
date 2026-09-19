# -*- coding: utf-8 -*-
"""生产化单条素材工作流（p0521 用户拍板：替代消融实验）。

同一份人物约束 → 生成前控制（证据质量门 + 传递链 + 主体作用域冲突检查）
→ H3 生成一条 12s → 两段 Omni 审核（盲观察 + 逐属性核对）→ 机器判定与
人工 override 分离 → 有 conditioning delta 才允许一次定向修复 → 预算内
未解决就停止（不自动烧第三次）。

文献机制轻量版：VideoMemory（固定人物卡+参考复用）、Gloria（职责单一的
紧凑角色锚 + 证据质量门）、VideoRepair（逐属性细粒度核对）、VISTA（压缩
版反馈链：单候选+最多一修）。不搬论文代码，不引第三模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agentic_video.generation_long_take import (
    EVIDENCE_QUALITY_REQUIRED, LT0Blocked, RECIPES, SGLangH3Client,
    V9GBlocked, _ensure_h3_server, _ffprobe_bin_of, _load_p04e, _stop_owned_server,
    build_canonical_world_pack, build_long_take_request,
    check_take_attributes, choose_canonical_pack, compare_take_to_brief,
    compile_long_take_brief, default_subject_constraints,
    observe_long_take, prepare_visual_reference, probe_media_geometry,
    validate_evidence_quality, validate_lt0_wire_prompt,
    validate_subject_constraints, validate_subject_scoped_actions,
    _write_json)
from src.perception import common
from src.perception.omni_pool import OmniProcessPool

PRODUCTION_VERSION = "production_take_v1"
# 预算（用户 §十：上限不是最佳参数；第二次是修复额度不是必选项）
PRODUCTION_BUDGET = {
    "max_h3_requests": 2,
    "outputs_per_request": 1,
    "duration_seconds": 12.0,
    "allow_seed_sweep": False,
    "allow_automatic_third_attempt": False,
    "max_repairs": 1,
    "require_specific_condition_change": True,
}


class ProductionBlocked(RuntimeError):
    def __init__(self, stage: str, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{stage}:{reason_code}:{detail}")
        self.stage = stage
        self.reason_code = reason_code
        self.detail = detail


def validate_attribute_transmission(constraints: dict[str, Any],
                                    brief: dict[str, Any],
                                    request: dict[str, Any],
                                    pack: dict[str, Any]) -> list[dict[str, Any]]:
    """四环传递链（p0521）：卡属性 → prompt 子句 → 参考资产 → 验收项。

    缺任一环不发送 H3 请求。返回映射表（落档审计用）。
    """
    prompt = str(request.get("prompt") or "")
    by_role = {str(entry.get("role")): entry
               for entry in pack.get("entries") or []}
    # morphology 锚实际标签（与 wire 生成同规则：Picture 1=identity，其后
    # body 按条目顺序，C1 最后）
    body_entries = [entry for entry in pack.get("entries") or []
                    if str(entry.get("role")).startswith("c0_body")]
    body_list = " and ".join(f"<Picture {2 + index}>"
                             for index in range(len(body_entries)))
    mapping = []
    for row in constraints.get("critical_attributes") or []:
        attribute_id = str(row["id"])
        # 环 1：卡（结构校验已过）
        # 环 2：prompt 中的明确约束——**正面层必须存在**（只有负面 guard
        # 的 prompt 正是修正 3 要消灭的弱约束形态；semantic/negative 不能
        # 替代指向参考的正面句）
        wire = row.get("wire") or {}
        positive = (str(wire.get("positive") or "")
                    .replace("{BODY_N}", "2")
                    .replace("{BODY_LIST}", body_list))
        if positive:
            in_prompt = positive in prompt
        elif wire.get("semantic") or wire.get("negative"):
            in_prompt = str(wire.get("semantic") or
                            wire.get("negative") or "") in prompt
        else:
            # 无 wire 字段的属性（如 target_character）：以 Subject 定义句为准
            in_prompt = "<Subject 1>" in prompt
        # 环 3：参考资产存在且过质量门
        assets_ok = bool(row.get("evidence_asset_ids"))
        if assets_ok:
            for asset_id in row["evidence_asset_ids"]:
                quality = (by_role.get(str(asset_id)) or {}
                           ).get("evidence_quality") or {}
                if any(quality.get(key) != expected
                       for key, expected in EVIDENCE_QUALITY_REQUIRED.items()):
                    assets_ok = False
                    break
        # 环 4：验收项 = 属性核对覆盖每个 attribute（check_take_attributes
        # 的 expected 集合来自同一张卡，结构性保证；此处登记）
        mapping.append({"attribute_id": attribute_id,
                        "in_prompt": in_prompt,
                        "evidence_asset_ids": row.get("evidence_asset_ids") or [],
                        "evidence_quality_ok": assets_ok,
                        "acceptance_item": f"attribute:{attribute_id}"})
        if not in_prompt:
            raise ProductionBlocked(
                "precheck", "attribute_missing_in_prompt", attribute_id)
        if not assets_ok:
            raise ProductionBlocked(
                "precheck", "attribute_evidence_not_approved", attribute_id)
    return mapping


def evaluate_take_against_card(observation_result: dict[str, Any],
                               attribute_result: dict[str, Any]) -> dict[str, Any]:
    """机器判定（p0521 修正 4）：一致性（稳）与目标人物（对）分开；
    unknown 不折算 pass；缺项不假通过。人工 override 单独记录，不覆盖。"""
    verdicts = attribute_result.get("attribute_verdicts") or {}
    rows = {key: {"verdict": row.get("verdict"),
                  "ownership_confirmed": row.get("ownership_confirmed"),
                  "evidence_interval": row.get("evidence_interval"),
                  "notes": row.get("notes")}
            for key, row in verdicts.items()}
    values = [row["verdict"] for row in rows.values()] or ["unknown"]
    if "fail" in values:
        machine = "fail"
    elif "unknown" in values:
        machine = "unknown"
    else:
        machine = "pass"
    return {
        "target_person": {"attribute_verdicts": rows,
                          "scope_note": attribute_result.get("scope_note")},
        "machine_verdict": machine,
        "human_override": None,
        "final_acceptance": machine == "pass",
    }


def apply_human_override(evaluation: dict[str, Any], decision: str,
                         reason: str) -> dict[str, Any]:
    """人工 override 单独落档：machine_verdict 永不改动（审计链不失真）。"""
    if decision not in {"pass", "fail"}:
        raise ProductionBlocked("acceptance", "override_decision_invalid",
                                decision)
    evaluation["human_override"] = {"decision": decision,
                                    "reason": reason}
    evaluation["final_acceptance"] = decision == "pass"
    return evaluation


def validate_repair_plan(repair_plan: dict[str, Any],
                         previous_request: dict[str, Any]) -> None:
    """修复门（p0521 修正 5）：必须有真实 conditioning delta。

    delta 为空 / 仅换 seed / 条件没变 → STOP（第二次钱不许白花：参考已
    清楚而 H3 不听，是 conditioning capacity 问题，不是抽卡能解决的）。
    """
    delta = repair_plan.get("condition_delta") or {}
    new_conditions = delta.get("new") or []
    old_conditions = delta.get("old") or []
    if not new_conditions or new_conditions == old_conditions:
        raise ProductionBlocked(
            "repair", "repair_without_condition_delta",
            "second generation requires a real conditioning change "
            "(e.g. a complementary body reference); STOP and archive "
            "evidence instead")
    if repair_plan.get("seed_only"):
        raise ProductionBlocked("repair", "repair_seed_only_forbidden",
                                "seed reroll is not a repair")
    if not str(repair_plan.get("why_this_change_addresses_failure") or
               "").strip():
        raise ProductionBlocked("repair", "repair_rationale_missing", "")
    if json.dumps(previous_request.get("conditions") or [],
                  ensure_ascii=False, sort_keys=True) == json.dumps(
            new_conditions, ensure_ascii=False, sort_keys=True):
        raise ProductionBlocked("repair", "repair_without_condition_delta",
                                "new conditions identical to previous")


def run_production_take(cfg: Any, p04e_dir: Path, output_dir: Path, *,
                        reference: Path,
                        constraints: dict[str, Any] | None = None,
                        pack_picks: dict[str, Any] | None = None,
                        repair_plan: dict[str, Any] | None = None,
                        plan_only: bool = False, execute: bool = False,
                        endpoint: str = "http://127.0.0.1:30011",
                        gpu_set: str = "0,1",
                        gpu_pairs: str = "0,1",
                        h3_backend: str = "diffusers",
                        h3_python_bin: str | None = None,
                        manage_server: bool = True,
                        ffmpeg_bin: str = "ffmpeg") -> dict[str, Any]:
    """一条生产素材：预检（拦）→ 1×H3 → 两段审核 → 判定/决策表。

    预算：max_h3_requests=2（第二次仅凭 repair_plan 里的真实 conditioning
    delta）；unknown → 补看不生成；fail 无 delta → 停止并存档证据。
    """
    from src.agentic_video.generation_p51 import probe_gpu_preflight

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints = constraints or default_subject_constraints()
    validate_subject_constraints(constraints)
    _write_json(output_dir / "subject_constraints.json", constraints)
    p04e = _load_p04e(Path(p04e_dir).resolve())
    section_clip = Path(p04e_dir) / "review_assets" / "section_02.mp4"

    # ---- 计划阶段（CPU）：pack（含人工批准的 c0_body 证据）→ 三道门 ----
    candidates = build_canonical_world_pack(
        Path(reference), output_dir, ffmpeg_bin=ffmpeg_bin)
    pack = choose_canonical_pack(candidates, output_dir,
                                 picks=(pack_picks or {}).get("picks"))
    # morphology 锚选择（p0522）：单帧够则一锚；否则双互补锚——由人工在
    # picks.morphology_refs 指定（默认双区域各一）。约束卡的 evidence_asset_ids
    # 同步为所选锚（门与 wire 均按实际锚集合工作）。
    morphology_refs = ((pack_picks or {}).get("morphology_refs") or
                       ["c0_body_front", "c0_body_aftermath"])
    pack["entries"] = [entry for entry in pack.get("entries") or []
                       if not str(entry["role"]).startswith("c0_body") or
                       str(entry["role"]) in morphology_refs]
    for row in constraints.get("critical_attributes") or []:
        if "hand_morphology" in str(row.get("id") or ""):
            row["evidence_asset_ids"] = list(morphology_refs)
    # 人工批准信息随 picks 文件写入（evidence_quality 各字段）
    approvals = (pack_picks or {}).get("evidence_quality") or {}
    for entry in pack.get("entries") or []:
        if entry["role"] in approvals:
            entry["evidence_quality"] = approvals[entry["role"]]
    _write_json(output_dir / "canonical_world_pack.json", pack)
    visual_reference = prepare_visual_reference(section_clip, output_dir,
                                                ffmpeg_bin=ffmpeg_bin)
    brief = compile_long_take_brief(
        p04e["content_program"], p04e["edit_program"], p04e["requirement"],
        p04e["section_observations"], constraints=constraints,
        output_dir=output_dir)
    geometry = probe_media_geometry(_ffprobe_bin_of(ffmpeg_bin), section_clip)
    active = (visual_reference or {}).get("active_crop")
    from src.agentic_video.generation_long_take import \
        resolve_reference_aspect_ratio
    aspect = resolve_reference_aspect_ratio(
        int(active["w"]) if active else geometry["width"],
        int(active["h"]) if active else geometry["height"])
    request = build_long_take_request(
        brief, pack=pack, visual_reference=visual_reference,
        recipe="canonical_plus_video", seed=1001,
        seconds=PRODUCTION_BUDGET["duration_seconds"], aspect_ratio=aspect)
    validate_lt0_wire_prompt(request)
    # 门 1：证据质量（错图不能一路"完整传递"到 H3）
    validate_evidence_quality(constraints, pack)
    # 门 2：主体作用域动作冲突（不全局扫词；禁止句/C1 动作不误触发）
    validate_subject_scoped_actions(brief, constraints)
    # 门 3：四环传递链
    transmission = validate_attribute_transmission(
        constraints, brief, request, pack)
    plan = {"schema_version": PRODUCTION_VERSION,
            "budget": PRODUCTION_BUDGET,
            "aspect_ratio": aspect,
            "canonical_pack": pack, "visual_reference": visual_reference,
            "attribute_transmission": transmission,
            "request": request}
    _write_json(output_dir / "production_plan.json", plan)
    if plan_only or not execute:
        return {"phase": "planned",
                "plan_path": str(output_dir / "production_plan.json"),
                "gates_passed": ["wire_audit", "evidence_quality",
                                 "subject_scoped_actions",
                                 "attribute_transmission"],
                "contact_sheets": [row["contact_sheet"]
                                   for row in candidates["roles"]]}

    # ---- 生成阶段：单请求；预算=1（repair 时=2）----
    take_dir = output_dir / ("take_r1" if repair_plan else "take_001")
    take_dir.mkdir(parents=True, exist_ok=True)
    if repair_plan is not None:
        previous = json.loads(
            (output_dir / "production_plan.json").read_text(encoding="utf-8")
        ) if (output_dir / "production_plan.json").is_file() else plan
        validate_repair_plan(repair_plan, previous.get("request") or request)
        # 真实 delta 应用：替换 conditions（保持 references 顺序规则）
        delta = repair_plan["condition_delta"]
        for row in delta.get("new"):
            if row not in (request.get("conditions") or []):
                request["conditions"].append(row)
        request["prompt"] += "\n" + str(
            repair_plan.get("prompt_addendum") or "")
        validate_lt0_wire_prompt(request)
    preflight = probe_gpu_preflight(
        [int(item) for item in gpu_set.split(",") if item.strip()])
    if preflight["status"] != "READY":
        raise ProductionBlocked("h3", "BLOCKED_GPU_BUSY", "")
    port = endpoint.rsplit(":", 1)[-1].split("/")[0]
    repo_root = Path(__file__).resolve().parents[2]
    server = _ensure_h3_server(
        endpoint, backend=h3_backend, variant="unified",
        manage_server=manage_server, port=int(port), repo_root=repo_root,
        log_path=output_dir / "h3_server.log", gpu_set=gpu_set,
        python_bin=h3_python_bin)
    client = SGLangH3Client(endpoint)
    job_state: dict[str, Any] = {"state": "planned"}
    try:
        _write_json(take_dir / "request.json", request)
        try:
            payload = client.serialize_payload(request)
            created = client.submit(payload)
            job_state.update({"state": "submitted", "job_id": created.get("id")})
            completed = client.poll(str(created["id"]))
            media = client.download(str(created["id"]), take_dir / "take.mp4")
            job_state.update({"state": "downloaded", **media})
        except V9GBlocked as exc:
            # 轮询超时/失败：不自动重发（先查服务端任务状态）
            job_state.update({"state": "failed", "reason_code": exc.reason_code,
                              "detail": str(exc)[:500], "resubmitted": False})
        _write_json(take_dir / "job_state.json", job_state)
    finally:
        _stop_owned_server(server)
    if job_state.get("state") != "downloaded":
        return {"phase": "generation_failed",
                "job_state": job_state,
                "note": "no automatic resubmission; inspect server job "
                        "status before any retry"}

    # ---- 审核阶段：盲观察 + 属性核对（Omni 池；H3 已停）----
    omni_cfg = (getattr(cfg, "perception", None) or {}).get("omni") or {}
    with OmniProcessPool(gpu_pairs, omni_cfg, ffmpeg_bin=ffmpeg_bin,
                         response_timeout_s=3600.0) as runner:
        observation = observe_long_take(
            Path(job_state["output_path"]), output_dir, runner=runner,
            take_id=take_dir.name, pack=pack, ffmpeg_bin=ffmpeg_bin)
        attribute = check_take_attributes(
            Path(job_state["output_path"]), output_dir, runner=runner,
            take_id=take_dir.name, constraints=constraints, pack=pack,
            ffmpeg_bin=ffmpeg_bin)
    comparison = compare_take_to_brief(observation, brief)
    evaluation = evaluate_take_against_card(observation, attribute)
    report = {
        "schema_version": PRODUCTION_VERSION,
        "take_dir": str(take_dir), "budget": PRODUCTION_BUDGET,
        "h3_requests_used": 2 if repair_plan else 1,
        "comparison": comparison, "evaluation": evaluation,
        "decision": None,
    }
    # 决策表（p0521 §八）：pass→待人工快速确认；unknown→补看不生成；
    # fail→有真实 delta 才修一次；无 delta→停止存档。
    machine = evaluation["machine_verdict"]
    if machine == "pass":
        report["decision"] = {
            "action": "human_quick_review",
            "regenerate": False,
            "note": "机器全 pass；人工看一遍 12s+关键帧后 apply_human_override "
                    "或直接采纳"}
    elif machine == "unknown":
        report["decision"] = {
            "action": "inspect_existing_output",
            "regenerate": False,
            "note": "关键区域不可见不是失败：补抽清楚帧/复看局部/人工确认；"
                    "不得据此重新生成"}
    else:
        report["decision"] = {
            "action": "repair_gate" if repair_plan is None else "stop_budget",
            "regenerate": repair_plan is None,
            "note": ("fail 需真实 conditioning delta 才允许第二次"
                     if repair_plan is None else
                     "修复额度已用或无新 delta：存档证据，本轮结束")}
    _write_json(output_dir / "production_report.json", report)
    return {"phase": "complete",
            "report_path": str(output_dir / "production_report.json"),
            "machine_verdict": machine,
            "decision": report["decision"]["action"]}
