"""生成层 prompts（纯文本模块）。"""
from __future__ import annotations

from src.generation.models import GenerationPlan, GenerationUnit, VariantSpec

T2VA_STYLE_HEADER = (
    "Vertical 9:16 short-form video, handheld comedic energy, "
    "natural lighting, crisp subject in frame. Audio: "
)


def dedupe_array_repetition(text: str, first_key: str) -> str:
    """数组复读保险：第二个以 first_key 开头的元素起点（`[{"key"`）即截断整个复读数组。

    注意不能用对象版的键级去重——数组里键在各元素中合法重复。
    """
    marker = '[{' + first_key
    first = text.find(marker)
    if first < 0:
        return text
    second = text.find(marker, first + len(marker))
    return text[:second].rstrip().rstrip(",") if second > 0 else text

REWRITE_PROMPT_HEADER = """你是短视频生成 prompt 工程师。下面给你一个「热门模板」的生成单元列表（每个单元有中文视觉描述、
台词、以及全模板的 BGM 描述）和一个「变体替换要求」。请为每个单元写一条给文生视频模型（生成画面+同步音频）的英文 prompt。

要求：
1. 只输出一个 JSON 数组，无围栏无解释：[{"unit": 0, "prompt": "..."}, {"unit": 1, "prompt": "..."}]
2. 每条 prompt 用英文描述：画面主体与动作 / 场景 / 光线与运镜 / 环境音与 BGM 氛围；
   人物要说出的台词用原语言（中文）原样保留，写成 The person says: "<台词>"。
3. 严格执行变体替换：把可替换元素换成新值；固定元素（结构节拍/钩子/反转）必须保留。
4. 同一单元内若含多个先后动作，用 "first ... then ..." 时序措辞连接。
5. 每条 prompt 以 "Vertical 9:16 short-form video" 开头，60~120 词。
6. 若单元视觉描述含「模仿某角色/IP」（如游戏/影视角色的台词与动作、名人模仿），每条 prompt 必须把
   人物写成该角色的可辨识模仿演绎（标志性造型、姿势、台词语气），模仿行为是梗核心，不得只写道具场景。
"""


def unit_brief(u: GenerationUnit) -> str:
    speech = "；台词：" + "／".join(f"「{s}」" for s in u.speech_lines) if u.speech_lines else ""
    return (f"单元 u{u.unit_id:02d}（{u.duration_s:.1f}s，roles={'+'.join(u.roles)}）："
            f"{u.visual_brief}{speech}")


def build_rewrite_prompt(plan: GenerationPlan, variant: VariantSpec) -> str:
    briefs = "\n".join(unit_brief(u) for u in plan.units)
    subs = "\n".join(f"  - {k} → {v}" for k, v in variant.substitutions.items())
    return (
        REWRITE_PROMPT_HEADER
        + f"\n【模板 BGM】{plan.audio_brief or '未提供'}\n"
        + f"\n【生成单元】\n{briefs}\n"
        + f"\n【变体：{variant.label}】替换要求：\n{subs}\n"
    )


VARIANT_PROPOSAL_PROMPT_HEADER = """你是短视频翻拍策划。基于下面的热门模板核心梗、固定元素和可替换元素，
提出 {n} 个不同变体。要求：
1. 只输出 JSON 数组，无围栏无解释：
   [{{"variant_id": "snake_case_英文", "label": "中文标签", "substitutions": {{"<可替换元素原文>": "新值", ...}}}}]
2. 每个变体只替换「可替换元素」里列出的项（键必须用给出的原文），固定元素不得改动；
   变体之间风格差异要大（如换道具/换场景/换服装主题各一个）。
3. 若模板是对某角色/IP 的模仿（游戏/影视角色等），不得替换被模仿的角色/出处本身
   （那是梗核心），只换场景/服装皮肤/道具等外围元素。
"""


def build_variant_proposal_prompt(core_meme: str, fixed: list[str], replaceable: list[str], n: int) -> str:
    return (
        VARIANT_PROPOSAL_PROMPT_HEADER.format(n=n)
        + f"\n【核心梗】{core_meme}\n"
        + f"\n【固定元素（必须保留）】\n" + "\n".join(f"- {f}" for f in fixed)
        + f"\n\n【可替换元素（只能替换这些）】\n" + "\n".join(f"- {r}" for r in replaceable)
        + "\n"
    )


# 预设池（ask() 不可用时的降级；键与常见 replaceable 元素措辞模糊匹配）
PRESET_SUBSTITUTION_POOL = [
    ("giftbox", "礼盒变体", {"道具": "一个系着红丝带的大礼盒", "场景": "明亮的客厅", "服装": "居家休闲装"}),
    ("subway", "地铁变体", {"道具": "一个银色行李箱", "场景": "地铁车厢里", "服装": "通勤西装"}),
    ("snowfield", "雪地变体", {"道具": "一个彩色礼物盒", "场景": "雪地里", "服装": "鲜艳的羽绒服"}),
]


def build_t2va_prompt(visual_en: str, ambience_en: str | None = None,
                      speech: str | None = None, duration_s: float = 5.2) -> str:
    parts = [T2VA_STYLE_HEADER + (ambience_en or "ambient room tone")]
    if speech:
        parts.append(f'The person says: "{speech}"')
    parts.append(visual_en)
    return " ".join(parts)
