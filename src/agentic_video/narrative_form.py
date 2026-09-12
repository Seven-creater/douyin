"""Narrative Form（V3 P1）：参考叙事 → 抽象表达形态 → 目标需求 的映射层。

lxh_p4_C2 badcase 的两个结构性病灶在此修复：
1. _expand_thin_arc 给薄参考弧（hook/consequence）强补 conflict——把
   "断言→反证→扩展→收尾" 型视频错套成"冲突"故事语法。Form 表按参考片的
   实际表达结构提供声明式槽序列，薄弧如实映射，绝不合成不存在的冲突。
2. 参考事实泄漏 target need（跆拳道服女性）。Form 的 need 按 form_function
   编译，零参考事实——Reference Narrative 不直接成为 Target Narrative。

Entity Role Schema（外审二轮核心补充）：主角持续性是 global 硬约束——
protagonist 槽只允许锁定实体，禁止 relaxed；配角/环境/反应槽显式声明可选。
身份约束 > 语义相似度（identity constraint > semantic similarity）。
"""
from __future__ import annotations

from copy import deepcopy

# 角色级主角要求（厚参考弧路径与 Form 路径共用）：
# consequence 允许"主角或明确受影响对象"（结果可以是他人反应镜头）；
# 其余主角槽一律 required——reaction/antagonist 类内容用 context 槽承载。
ROLE_ENTITY_REQUIREMENTS = {
    "hook": {"protagonist": "required"},
    "context": {"protagonist": "optional"},
    "conflict": {"protagonist": "required"},
    "choice": {"protagonist": "required"},
    "climax": {"protagonist": "required"},
    "consequence": {"protagonist": "optional"},
    "resolution": {"protagonist": "required"},
}

_FORM_FUNCTION_NEEDS = {
    "premise": {
        "need": "开场呈现主角的处境或一个关于主角的待检验断言，让观众想看后续",
        "must_have": ["主角在场的可见处境"],
        "must_not": ["与主角无关的纯环境空镜"],
        "evidence_mode": "visual",
    },
    "counter_evidence": {
        "need": "用主角的可见行动直接反驳开场断言——观众看到的是'做得到'",
        "must_have": ["主角的关键行动可见", "主角在场"],
        "must_not": ["主角不在场的纯环境镜头", "平淡过场"],
        "evidence_mode": "visual",
    },
    "evidence_expansion": {
        "need": "进一步展示主角更多能力/成绩/生活事实，扩展反驳的证据面",
        "must_have": ["主角的新增可见事实（能力/成绩/状态）"],
        "must_not": ["重复前槽已呈现的同一动作"],
        "evidence_mode": "both",
    },
    "payoff": {
        "need": "用一个具体的收束动作或细节完成情绪落地（和解/启程/轻细节）",
        "must_have": ["收束性画面（和解/启程/定格/轻细节特写）", "主角在场"],
        "must_not": ["悬而未决的新冲突", "与前槽行动无关的新场景"],
        "evidence_mode": "both",
    },
}

NARRATIVE_FORMS = {
    # 7682 型：断言（刻板印象）→ 视觉实证反驳 → 成就/事实扩展 → 轻细节收束。
    # 文字轨强（OCR/ASR 密集）且参考弧无 conflict/climax 时命中。
    "assertion_visual_payoff": {
        "description": "提出断言 → 可见行动反驳 → 事实扩展 → 收束细节",
        "entity_schema": {
            "protagonist": {"cardinality": 1, "persistence": "global"},
            "supporting": {"cardinality": "0..N", "persistence": "local"},
        },
        "slots": [
            {"role": "hook", "form_function": "premise", "required": True},
            {"role": "context", "form_function": "counter_evidence", "required": True},
            {"role": "consequence", "form_function": "evidence_expansion",
             "required": True},
            {"role": "resolution", "form_function": "payoff", "required": True},
        ],
        "copy_formula": "hook=断言句；cards=可见实证连发；punchline=轻细节收束",
    },
    # 传统叙事弧：conflict/climax 在场或因果链非空时命中。
    "classic_arc": {
        "description": "处境 → 冲突 → 关键行动 → 收束",
        "entity_schema": {
            "protagonist": {"cardinality": 1, "persistence": "global"},
            "supporting": {"cardinality": "0..N", "persistence": "local"},
            "antagonist": {"cardinality": "0..N", "persistence": "local"},
        },
        "slots": [
            {"role": "hook", "form_function": "premise", "required": True},
            {"role": "context", "form_function": "evidence_expansion",
             "required": False},
            {"role": "conflict", "form_function": "confrontation", "required": True},
            {"role": "climax", "form_function": "decisive_action", "required": True},
            {"role": "resolution", "form_function": "payoff", "required": True},
        ],
        "copy_formula": "hook=悬念；cards=冲突升级；punchline=结果反转",
    },
}

_FORM_FUNCTION_NEEDS.update({
    "confrontation": {
        "need": "呈现主角面对的问题具象化为可见的冲突或危险",
        "must_have": ["冲突/危险可见", "主角在场"],
        "must_not": ["主角不在场的纯环境镜头"],
        "evidence_mode": "visual",
    },
    "decisive_action": {
        "need": "呈现主角的关键行动与情绪峰值",
        "must_have": ["高潮动作或情绪峰值画面", "主角在场"],
        "must_not": ["平淡过场"],
        "evidence_mode": "visual",
    },
})

THIN_ARC_SLOTS = 3      # 参考弧段数低于此 → Form 模板槽序列接管


def infer_narrative_form(narrative: dict) -> str:
    """确定性形态推断（不调模型）：参考弧有 conflict/climax/choice 或因果链
    非空 → classic_arc；无冲突型角色且文字轨密/含 consequence →
    assertion_visual_payoff。"""
    roles = {segment.get("role") for segment in narrative.get("arc") or []}
    arc_len = len(narrative.get("arc") or [])
    if roles & {"conflict", "climax", "choice"}:
        return "classic_arc"
    # 因果链只在厚弧时才指向 classic——薄弧+少量链接不该被强套经典弧
    # （V3 P1：绝不给断言型视频合成冲突语法）。
    if narrative.get("causal_links") and arc_len >= THIN_ARC_SLOTS:
        return "classic_arc"
    text_track = len(narrative.get("utterances") or []) >= 3
    if text_track or "consequence" in roles or roles <= {"hook", "context",
                                                         "resolution"}:
        return "assertion_visual_payoff"
    return "classic_arc"


def resolve_slot_sequence(narrative: dict, *, form_name: str | None = None
                          ) -> tuple[list[dict], str]:
    """槽序列解析。厚参考弧（≥3 段）直接用参考弧段（保留事件引用与既有
    行为）；薄弧由 Form 模板提供声明式槽序列——替代已删除的 _expand_thin_arc
    自动补弧。返回 (slots, form_name)；每槽带 role/required/
    entity_requirements/form_function（+厚弧段的 event_ids）。"""
    form_name = form_name or infer_narrative_form(narrative)
    arc = [segment for segment in narrative.get("arc") or []]
    events = {event.get("id"): event for event in narrative.get("events") or []}
    slots: list[dict] = []
    if len(arc) >= THIN_ARC_SLOTS:
        for segment in arc:
            role = segment["role"]
            requirements = deepcopy(ROLE_ENTITY_REQUIREMENTS.get(role))
            # 模板决定哪些位置必须同主体：参考该槽事件零参与者（环境/反应/
            # 收束镜头位）→ 主角要求降为可选，不由角色名一刀切。
            participants = set()
            for event_id in segment.get("event_ids") or []:
                participants |= set((events.get(event_id) or {}).get("participants") or [])
            if requirements and requirements.get("protagonist") == "required" \
                    and not participants:
                requirements["protagonist"] = "optional"
            # V4 C：form_function 消费顺序 = canonical enum（narrative_agent
            # 双字段输出）→ legacy 关键词归一 → 角色名回退。自由中文整句
            # 直接当 form_function 是 V3 转换丢失的根因。
            raw_enum = str(segment.get("form_function") or "").strip()
            if raw_enum in _FORM_FUNCTION_NEEDS:
                form_function = raw_enum
            else:
                form_function = (normalize_form_function(
                    str(segment.get("function") or "")) or role)
            slots.append({
                "role": role,
                "required": role in {"hook", "conflict", "climax", "resolution"},
                "entity_requirements": requirements,
                "form_function": form_function,
                "event_ids": list(segment.get("event_ids") or []),
            })
    else:
        form = NARRATIVE_FORMS.get(form_name) or NARRATIVE_FORMS["classic_arc"]
        for template in form["slots"]:
            slots.append({
                "role": template["role"],
                "required": bool(template.get("required")),
                "entity_requirements": deepcopy(
                    ROLE_ENTITY_REQUIREMENTS.get(template["role"])),
                "form_function": template["form_function"],
                "event_ids": [],
            })
    return slots, form_name


def compile_form_need(form_function: str) -> dict:
    """按 form_function 编译零参考事实的证据需求（need/must_have/must_not）。

    V4 C（外审三轮）：消费顺序 = 固定 enum 精确命中 → legacy 关键词归一器
    （旧数据/模型没给 enum 时的迁移补丁）→ 空（上层回退角色模板）。
    红线不变：need 文本永远来自模板，绝不把输入句原文拼进 need——参考片
    的自由中文 function 可能携带参考事实（跆拳道/比赛），拼进去就是 C2
    式跨库荒谬检索需求。"""
    value = str(form_function or "")
    template = _FORM_FUNCTION_NEEDS.get(value)
    if template is not None:
        return deepcopy(template)
    normalized = normalize_form_function(value)
    if normalized is not None:
        return deepcopy(_FORM_FUNCTION_NEEDS[normalized])
    return {"need": "", "must_have": [], "must_not": [], "evidence_mode": "visual"}


# legacy 关键词归一器（迁移补丁，长期架构是 narrative_agent 输出 canonical
# enum）。词表按外审十四节裁定：列举/成就/能力 等是**强扩展信号**——同句
# 出现时压过 对比/局限（"列举生活技能与局限形成对比"在后段承担继续扩展
# 证据+人物化收束，非首段反证）；其余按首条命中。
_EXPANSION_STRONG = ("扩展", "更多", "进一步", "成绩", "成就", "能力", "实力",
                     "证明", "认可", "列举")
_FORM_KEYWORD_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("反驳", "打破", "证伪", "做得到", "反差", "对比"), "counter_evidence"),
    (("冲突", "对抗", "危险", "威胁"), "confrontation"),
    (("高潮", "关键行动", "峰值", "爆发"), "decisive_action"),
    (("收束", "落地", "和解", "启程", "收尾", "局限", "反转"), "payoff"),
    (("断言", "处境", "开场", "悬念"), "premise"),
    (_EXPANSION_STRONG, "evidence_expansion"),
)


def normalize_form_function(text: str) -> str | None:
    """自由中文 function → 形态 enum（确定性关键词匹配，无模型）。

    多类同句时的裁定（外审十四节）：强扩展信号（列举/成就/能力…）优先于
    对比/局限——"列举生活技能与局限形成对比" → evidence_expansion。"""
    value = str(text or "")
    if not value:
        return None
    if any(keyword in value for keyword in _EXPANSION_STRONG):
        return "evidence_expansion"
    for keywords, enum in _FORM_KEYWORD_RULES:
        if any(keyword in value for keyword in keywords):
            return enum
    return None


def entity_schema(form_name: str) -> dict:
    return deepcopy((NARRATIVE_FORMS.get(form_name) or NARRATIVE_FORMS["classic_arc"])
                    .get("entity_schema") or {})


def protagonist_required(slot: dict) -> bool:
    return ((slot.get("entity_requirements") or {})
            .get("protagonist") == "required")
