# -*- coding: utf-8 -*-
"""P0.4 叙事功能边界判定（Narrative Boundary）。

用户拍板（P0.4 转向）：Section 边界不再判"是否同一 event"（粒度错配根源），
只判**叙事功能是否变化**。Section 是服务下游生成的中间表示，不是标注
benchmark（GEBD：同一视频存在多个合理粒度；边界差 1-2 秒不影响生成）。

分工：LLM 只从**有限功能表**里为边界两侧各选一个功能枚举（描述层，枚举值
不可能"编造"），keep/move 由代码做字符串不等比较（决策层，确定性）。
无外部代码可复用（VISTA 未放码；GEBD 为检测器训练管线），本模块按用户
伪代码实现。
"""
from __future__ import annotations

from typing import Any

# 有限叙事功能表（通用论证角色，不含任何参考特定词）
NARRATIVE_FUNCTIONS: tuple[str, ...] = (
    "situation_setup",     # 建立情境/人物印象/前提
    "problem_statement",   # 提出待检验命题、悬念或偏见
    "counter_evidence",    # 用行动/事实直接反驳该命题
    "evidence_expansion",  # 并列展开更多独立证据
    "punchline_payoff",    # 收尾打趣、点题或情绪收束
)

NARRATIVE_BOUNDARY_PROMPT = """判断候选边界两侧的叙事功能是否变化。输入是整条
视频的全局表达目的（供参照，不含任何边界信息）与边界前后各三帧（before_1
约-2.0s、before_2 约-1.0s、before_3 约-0.4s、after_1 约+0.4s、after_2 约
+1.0s、after_3 约+2.0s）。

背景（GEBD 粒度宽容原则）：边界不需要精确到帧，本判定只回答"两侧是否承担
不同的叙事功能"。

功能取值（只能从这五个里选，逐项独立判断，禁止发明新值）：
- situation_setup：建立情境、人物印象或前提条件
- problem_statement：提出待检验的命题、悬念或成见
- counter_evidence：用行动或事实直接反驳上述命题
- evidence_expansion：并列展开更多互相独立的新证据
- punchline_payoff：收尾打趣、点题或情绪收束

判定规则：
1. 事件的完整闭环（准备→对抗→结果→赛后反应）属于同一功能；一个事件的
   直接后果与反应仍是原功能的收尾，不是新功能。**若两侧仍属同一完整事件
   （same_event=true），边界必不成立**——闭环内的庆祝/收束是 phase。
2. 新功能必须开启新的表达目的（换论证角色），不是给旧目的换情绪色彩。
3. punchline_payoff 只用于**片尾**的打趣/点题/收束段；事件刚结束时的
   情绪反应（微笑、庆祝、面对镜头）仍是该事件功能的收尾，不是
   punchline_payoff。
4. 只输出一个 JSON 对象：
{"before_function":"从功能表中选一个","after_function":"从功能表中选一个",
 "same_event":true,"reason":"具体描述两侧画面与功能归属的依据"}
same_event 记录两侧是否同一独立发生（供下游剪辑模式派生用，不影响本判定）。
reason 必须具体到画面内容，不得输出占位文字。输入："""


def decide_narrative_boundary(before_function: Any, after_function: Any,
                              same_event: Any = False) -> bool:
    """确定性终判：边界成立 ⟺ 功能枚举不同 **且** 两侧不是同一完整事件。

    p04b 实测：模型可把事件收尾庆祝选成 punchline_payoff（功能看似变化），
    但它同时如实报告 same_event=true（完整闭环未中断）。闭环内的收尾是
    phase 不是 Section，因此 same_event=true 一票否决（用户判据：完整事件
    setup/attack→result→reaction 必须同段）。
    """
    if same_event is True:
        return False
    return str(before_function) != str(after_function)


def validate_narrative_verdict(verdict: dict[str, Any]) -> None:
    """校验判定输出形状：功能必须取自有限表，reason 必须具体。"""
    from src.agentic_video.reference_program_v9 import V9Blocked

    for key in ("before_function", "after_function"):
        if str(verdict.get(key) or "").strip() not in NARRATIVE_FUNCTIONS:
            raise V9Blocked("boundary_reconciliation",
                            "narrative_verdict_invalid_function", str(key))
    if not isinstance(verdict.get("same_event"), bool):
        raise V9Blocked("boundary_reconciliation",
                        "narrative_verdict_invalid", "same_event")
    reason = str(verdict.get("reason") or "").strip()
    if (len(reason) < 6 or reason in {
            "具体描述两侧画面与功能归属的依据", "画面依据",
            "具体描述边界前后画面内容与叙事功能的差异"}):
        raise V9Blocked("boundary_reconciliation",
                        "narrative_verdict_echoed_example", "reason")
