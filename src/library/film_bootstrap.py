"""Film Knowledge Bootstrap（P1.5）：素材电影看片前先建世界模型。

外审二轮修订的核心设计：
1. **两层 Identity 永不合并**——标注层输出 local vis_ id + 旁挂 canonical
   binding（binding_status/confidence/evidence）；pack 认错只改 binding，
   视觉观察层数据无损可审计。
2. **Identity Gate 二次独立投票**——代表帧分 A/B 两组各自识别，agreement +
   平均置信 + 元数据佐证决定 tier；自报 confidence 不可信（可能认错作品还
   0.96）。模型认 A + 元数据说 B → identity_conflict，不注入任何实体表。
3. **annotation_view 只放 Recognition Prior**（roster/aliases/appearance）——
   关系/剧情/高潮/结局是 Narrative Prior，绝不进标注器（防"知道后面会发生
   什么就假装这段表现了它"）。
4. 红线：知识=prior（提假设缩搜索空间），视频=evidence（定事实）。

成功标准：pack-on 后同一真实角色跨窗口绑同一 canonical（pack-off 时
boy_1/girl_3/sword_person 漂移）——直接修复 V3 成片"前一个镜头还是这个人，
下一个镜头主角突然换掉"的底层根因。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import unicodedata
from pathlib import Path

from src.config import AppConfig
from src.library.entity_registry import load_entity_registry
from src.perception import common
from src.template.schema import extract_json_block

logger = logging.getLogger(__name__)

KNOWLEDGE_PACK_SCHEMA = "knowledge_pack_v1"
BOOTSTRAP_PROMPT_VERSION = "film_bootstrap_v1"

IDENTITY_PROMPT = """你是影视作品识别助手。这是同一部长片抽取的 {n_frames} 个代表帧，
来自影片的不同时间位置（帧间不连续，禁止从相邻帧推断事件顺序或因果），
帧角标注了 T=秒数。只输出 JSON：
{{"recognized": true, "title": "作品名（不确定留空）",
"installment": "系列第几部/剧场版名", "confidence": 0.0,
"basis": "判断依据：画风/角色设计/标志性场景/画面文字，而非凭空记忆",
"franchise_slug": "系列 ascii 短名如 luoxiaohei（无则空）",
"alt_candidates": [{{"title": "", "confidence": 0.0}}]}}
判定纪律：只有画面证据支持才给高置信；仅"画风像某公司"不算认识；
完全不认识就 recognized=false 且 confidence<0.4，禁止猜。
"""

ROSTER_PROMPT = """你已确认认识《{title}》（{installment}）。凭你对这部作品的知识
输出主要角色候选表。这是**待验证先验**，不是事实，后续会由视频证据逐窗核对。
只输出 JSON：
{{"franchise_slug": "{franchise}",
"entities": [{{"canonical_id": "char:{franchise}:拼音小写",
"name": "常用名", "aliases": ["别名"], "type": "human|creature|deity|group|object",
"appearance": "外观速写≤20字（发色/服装/体型/标志物）"}}],
"relations": [{{"a": "char:..", "b": "char:..", "relation": "师徒|敌对|同伴|亲情|救助|其他"}}],
"plot_arcs": [{{"segment": "A", "summary": "剧情段≤40字"}}]}}
只列主要角色（≤{max_entities} 个）；canonical_id 必须是
char:{franchise}:slug 三段式（人物）/ item:{franchise}:slug / loc:{franchise}:slug。
"""

# ---------------------------------------------------------------- 归一与工具


def _norm_title(text: str) -> str:
    out = []
    for char in unicodedata.normalize("NFKC", str(text or "")):
        if char.isalnum():
            out.append(char.lower())
    return "".join(out)


def _titles_agree(left: str, right: str) -> bool:
    a, b = _norm_title(left), _norm_title(right)
    return bool(a and b and (a == b or a in b or b in a))


def _stable_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()[:16]


def sanitize_canonical_id(canonical: str, franchise: str) -> str | None:
    """canonical 三段式规范：char:<franchise>:<slug>。模型给的单段 id 自动补
    franchise；非法字符清成 [a-z0-9-]。跨 installment 共用 franchise 名。"""
    value = str(canonical or "").strip().lower()
    value = re.sub(r"[^a-z0-9:-]", "-", value)
    parts = [part for part in value.split(":") if part]
    if not parts:
        return None
    kind = parts[0] if parts[0] in {"char", "item", "loc"} else "char"
    slug_parts = parts[1:] if parts[0] in {"char", "item", "loc"} else parts
    franchise = re.sub(r"[^a-z0-9-]", "-", str(franchise or "").lower()).strip("-")
    if not franchise:
        return None
    if len(slug_parts) >= 2 and slug_parts[0] == franchise:
        slug_parts = slug_parts[1:]                     # 模型把 franchise 重复写进 slug
    slug = "-".join(slug_parts).strip("-")
    if not slug:
        return None
    return f"{kind}:{franchise}:{slug}"


# ---------------------------------------------------------------- 代表帧

def pick_representative_timestamps(duration_s: float, *, n_uniform: int = 6,
                                   head_s: float = 90.0, tail_s: float = 360.0
                                   ) -> list[float]:
    """6 张均匀采样（复用 source_zones 排 OP/ED）。"""
    lo, hi = float(head_s), max(float(head_s) + 1, duration_s - float(tail_s))
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n_uniform
    return [round(lo + step * (idx + 0.5), 1) for idx in range(n_uniform)]


_PERSON_MARKERS = ("男", "女", "少年", "少女", "孩", "角色", "人物", "猫", "妖",
                  "师", "者", "人")


def pick_informative_frames(shots: list[dict], *, n: int = 4,
                            captions: dict | None = None) -> list[Path]:
    """4 张高信息帧：kfs 按 dialogue+action+**含人物 caption 加权**挑——
    冒烟实锤：纯分数挑出的全是风景（高空飞行/自然扫镜），Omni 诚实拒认
    （门控正确，采样背锅）；识别作品靠的是角色脸，不是山水。"""
    captions = captions or {}

    def _score(row: dict) -> float:
        caption = str(captions.get(str(row.get("shot_idx"))) or "")
        person_bonus = 2.0 if any(m in caption for m in _PERSON_MARKERS) else 0.0
        return (float(row.get("dialogue_score") or 0)
                + float(row.get("action_score") or 0) + person_bonus)

    ranked = sorted((row for row in shots if row.get("kfs")),
                    key=_score, reverse=True)
    frames: list[Path] = []
    for row in ranked:
        for kf in row.get("kfs") or []:
            if Path(kf).exists():
                frames.append(Path(kf))
                break
        if len(frames) >= n:
            break
    return frames


def pick_person_frames(shots: list[dict], captions: dict, *, n: int = 10
                       ) -> list[Path]:
    """含人物镜头 kf 的均匀采样（三轮冒烟实锤的最终结论：识别作品靠角色脸。

    纯时间均匀 → 山水风景；分数加权 → 高分风景；片头字幕区 → 出品方名单。
    captions 已判过主体，直接按"主体含人物"过滤后全片均匀取 n 张。"""
    person_shots = [row for row in shots if row.get("kfs")
                    and any(marker in str(captions.get(str(row.get("shot_idx")))
                                          or "") for marker in _PERSON_MARKERS)]
    if not person_shots:
        return []
    frames: list[Path] = []
    if len(person_shots) <= n:
        for row in person_shots:
            kf = next((Path(p) for p in row["kfs"] if Path(p).exists()), None)
            if kf:
                frames.append(kf)
        return frames
    step = len(person_shots) / n
    for idx in range(n):
        row = person_shots[min(int(idx * step), len(person_shots) - 1)]
        kf = next((Path(p) for p in row["kfs"] if Path(p).exists()), None)
        if kf:
            frames.append(kf)
    return frames


def build_slideshow(video: Path, work_dir: Path, *, timestamps: list[float],
                    informative_frames: list[Path], ffmpeg_bin: str = "ffmpeg"
                    ) -> tuple[Path, Path, list[Path]]:
    """→ (审计拼图 slideshow.jpg, slideshow.mp4, 归一帧列表)。帧统一 640 宽，
    mp4 走 -framerate 1（OmniRunner.watch 已实测的视频路径）。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = work_dir / "frames"
    norm_dir.mkdir(exist_ok=True)
    normalized: list[Path] = []
    for idx, ts in enumerate(timestamps):
        out = norm_dir / f"u{idx:02d}.jpg"
        common.run_ffmpeg(ffmpeg_bin, [
            "-y", "-loglevel", "error", "-ss", f"{ts:g}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=640:-2", "-qscale:v", "2", str(out)],
            timeout_s=120)
        normalized.append(out)
    for idx, kf in enumerate(informative_frames):
        out = norm_dir / f"i{idx:02d}.jpg"
        shutil.copy2(kf, out)
        normalized.append(out)
    names = sorted(p.name for p in norm_dir.glob("*.jpg"))
    pattern = names[0].replace("0", "%d", 1) if names else None
    count = len(names)
    grid = "5x2" if count <= 10 else "5x3"
    audit = work_dir / "slideshow.jpg"
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-pattern_type", "glob", "-i",
        str(norm_dir / "*.jpg"), "-vf", f"tile={grid}", str(audit)], timeout_s=120)
    # 交错排序（冒烟二轮实锤：A/B 按顺序切半把均匀风景帧全分给 A、人物帧全给
    # B——A 拒认拉低整门控）。交错后每组都是 均匀+人物 混合。
    uniform = [p for p in normalized if p.name.startswith("u")]
    informative = [p for p in normalized if p.name.startswith("i")]
    shuffled: list[Path] = []
    for idx in range(max(len(uniform), len(informative))):
        if idx < len(uniform):
            shuffled.append(uniform[idx])
        if idx < len(informative):
            shuffled.append(informative[idx])
    normalized = shuffled or normalized
    slideshow = work_dir / "slideshow.mp4"
    # 静音轨必带：OmniRunner.watch 的 use_audio_in_video=True 断言视频有音轨
    # （冒烟实锤），与 renderer 无声槽同款 anullsrc 方案
    common.run_ffmpeg(ffmpeg_bin, [
        "-y", "-loglevel", "error", "-framerate", "1", "-pattern_type", "glob",
        "-i", str(norm_dir / "*.jpg"),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-shortest", str(slideshow)], timeout_s=120)
    return audit, slideshow, normalized


# ---------------------------------------------------------------- 识别与门控

def parse_identity_answer(raw: str) -> dict:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "recognized": bool(payload.get("recognized")),
        "title": str(payload.get("title") or "").strip(),
        "installment": str(payload.get("installment") or "").strip(),
        "confidence": confidence,
        "basis": str(payload.get("basis") or "")[:200],
        "franchise_slug": str(payload.get("franchise_slug") or "").strip(),
    }


def title_from_metadata(metadata_title: str) -> str:
    """文件名→作品名：剥离 技术噪声（4K/H265/2160p/WEB-DL/…），取最长 CJK 段
    （含尾缀数字）。'罗小黑1 2019.4K.H265...' → '罗小黑1'。"""
    import re as _re

    text = str(metadata_title or "")
    cjk = _re.findall(r"[一-鿿]{2,}[0-9ⅠⅡⅢ]?", text)
    if not cjk:
        return ""
    return max(cjk, key=len)


def resolve_tier(vote_a: dict, vote_b: dict, metadata_title: str = "", *,
                 tier_high: float = 0.85, tier_low: float = 0.4) -> dict:
    """Identity Gate（外审二轮）：tier = f(双投一致性, 平均置信, 元数据佐证)。

    两投一致 + 平均≥high → model_prior；任一投票被元数据佐证 → model_prior；
    视觉全盲但文件名含明确作品名 → metadata_prior（四轮冒烟实锤：Qwen3-Omni
    对罗小黑视觉拒认——文件名是产品模式的合法便宜信号；roster 照生成但全部
    proposed，绑定层仍要求画面特征吻合才生效，红线"视频=evidence"不破）；
    模型与元数据互相矛盾 → identity_conflict（不注入任何实体表）；
    其余中间地带 → search；低置信 → unknown。"""
    same = _titles_agree(vote_a.get("title"), vote_b.get("title"))
    avg = (float(vote_a.get("confidence") or 0) + float(vote_b.get("confidence") or 0)) / 2
    meta = str(metadata_title or "").strip()
    votes = [vote for vote in (vote_a, vote_b)
             if vote.get("recognized") and vote.get("title")]
    meta_match = any(_titles_agree(vote.get("title"), meta) for vote in votes) if meta else False
    meta_conflict = bool(meta and votes
                         and not any(_titles_agree(vote.get("title"), meta)
                                     for vote in votes)
                         and all(vote.get("confidence", 0) >= tier_low for vote in votes))
    if meta_conflict:
        tier = "identity_conflict"
    elif same and avg >= tier_high:
        tier = "model_prior"
    elif meta_match:
        tier = "model_prior"
    elif not votes and title_from_metadata(meta):
        tier = "metadata_prior"
    elif (same and avg >= tier_low) or max(float(vote_a.get("confidence") or 0),
                                           float(vote_b.get("confidence") or 0)) >= 0.7:
        tier = "search"
    else:
        tier = "unknown"
    return {"tier": tier, "votes_agree": same, "avg_confidence": round(avg, 3),
            "metadata_match": meta_match, "metadata_conflict": meta_conflict,
            "metadata_title_extracted": title_from_metadata(meta)}


def parse_roster_answer(raw: str, *, franchise: str, max_entities: int) -> dict:
    block = extract_json_block(raw)
    try:
        payload = json.loads(block) if block else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    franchise_slug = re.sub(r"[^a-z0-9-]", "-",
                            str(payload.get("franchise_slug") or franchise
                                or "").lower()).strip("-") or None
    entities, seen = [], set()
    for item in payload.get("entities") or []:
        if not isinstance(item, dict):
            continue
        canonical = sanitize_canonical_id(str(item.get("canonical_id") or ""),
                                          franchise_slug or "unknown")
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        entities.append({
            "canonical_id": canonical,
            "name": str(item.get("name") or "").strip()[:24],
            "aliases": [str(a).strip()[:24] for a in item.get("aliases") or []][:6],
            "type": str(item.get("type") or "unknown"),
            "appearance": str(item.get("appearance") or "").strip()[:40],
            "source": "model_prior", "status": "proposed",
        })
        if len(entities) >= max_entities:
            break
    relations = []
    for item in payload.get("relations") or []:
        if not (isinstance(item, dict) and item.get("a") and item.get("b")):
            continue
        # 自环过滤（外审三轮实锤：roster 出现"某角色对自己亲情/敌对"）——
        # 同 narrative_index.parse_type_facets 已有同款
        if str(item.get("a")) == str(item.get("b")):
            continue
        relations.append({"a": str(item.get("a")), "b": str(item.get("b")),
                          "relation": str(item.get("relation") or "")[:16],
                          "source": "model_prior", "status": "proposed"})
    plot_arcs = []
    for item in payload.get("plot_arcs") or []:
        if isinstance(item, dict) and item.get("summary"):
            plot_arcs.append({"segment": str(item.get("segment") or "")[:4],
                              "summary": str(item.get("summary"))[:60],
                              "source": "model_prior", "status": "proposed"})
    return {"franchise_slug": franchise_slug, "entities": entities,
            "relations": relations, "plot_hypotheses": plot_arcs}


def _ground_roster_canonicals(entities: list[dict], franchise: str,
                              registry: dict) -> list[dict]:
    """canonical 真值归并（3 窗门控实锤的幻觉 slug 修复）。

    模型给的 slug 不可信：film1 pack 实况——「无限」的 slug 被写成 zhongli
    （钟离的拼音，纯记忆幻觉），「罗小黑」→luoxiaohai，且 franchise 断链后
    全落 char:unknown:* 命名空间。规则：
    ①roster name/alias 精确命中手写 registry 的身份别名 → 用手写 canonical
      （手写表是 canonical 真值来源；跨片/跨窗合并、row_identity_keys 的
      别名路径都认它，char:unknown:* 它视而不见）；
    ②未命中的新角色 → 代码确定性生成 char:{franchise}:e{N}，模型原始 slug
      记档 model_slug 供审计。"""
    from src.library.entity_registry import _norm, build_alias_maps

    aliases, _source_map, _windowed = build_alias_maps(registry)
    grounded, id_map = [], {}
    for idx, entity in enumerate(entities):
        canonical = None
        for name in [entity.get("name"), *(entity.get("aliases") or [])]:
            hit = aliases.get(_norm(str(name or "")))
            if hit:
                canonical = hit
                break
        if canonical is None:
            canonical = sanitize_canonical_id(
                f"char:{franchise or 'unknown'}:e{idx + 1}", franchise)
        id_map[str(entity.get("canonical_id") or "")] = canonical
        grounded.append({**entity, "canonical_id": canonical,
                         "model_slug": str(entity.get("canonical_id") or ""),
                         # grounded=命中手写真值；未命中的是模型记忆候选
                         # （metadata_prior 档幻觉高发：film1 roster 12 实体
                         # 9 个是编造角色），annotation_view 据此过滤注入。
                         "grounded": canonical in set(aliases.values())})
    return grounded, id_map


def _remap_relations(relations: list[dict], id_map: dict[str, str]) -> list[dict]:
    """roster relations 的 a/b 换成归并后的 canonical（防悬空引用）。"""
    return [{**relation,
             "a": id_map.get(str(relation.get("a")), str(relation.get("a"))),
             "b": id_map.get(str(relation.get("b")), str(relation.get("b")))}
            for relation in relations]


# ---------------------------------------------------------------- pack 视图

def annotation_view(pack: dict | None, *, max_entities: int = 12) -> list[dict]:
    """标注器唯一注入物（外审红线）：Recognition Prior——roster/aliases/
    appearance。关系与剧情是 Narrative Prior，绝不进标注器。tier 未达
    model_prior/metadata_prior 时返回空（含 model_prior_unverified：错 title
    会诱导幻觉；metadata_prior 的 roster 全 proposed，绑定层要求画面特征
    吻合才生效）。pack_validity 失效（塌缩采样/tier 缺失/roster 空）同样
    返回空——外审三轮必改①：坏 pack 不重跑也不能继续污染身份。"""
    if not isinstance(pack, dict):
        return []
    validity = pack_validity(pack)
    if not validity["valid"]:
        logger.warning("[bootstrap] pack rejected (%s): %s",
                       pack.get("source") or "?", "; ".join(validity["reasons"]))
        return []
    identity = pack.get("work_identity") or {}
    tier = identity.get("tier")
    if tier not in {"model_prior", "metadata_prior"}:
        return []
    view = []
    for entity in (pack.get("entities") or [])[:max_entities]:
        # 门控 v2 实锤：metadata_prior 档 roster 幻觉高发（film1 12 实体
        # 9 个是编造角色——阿根/小茶/谛听/李云祥…），标注器拿幻觉条目兜底
        # 乱绑（w19 三实体全绑幻觉角色 supported@0.9）。只注入归并到手工
        # 真值的条目；幻觉条目留在 pack 归档，不进 binding 候选表。
        # model_prior（视觉已认出作品）roster 可信度高，全量注入。
        if tier == "metadata_prior" and "grounded" in entity \
                and not entity.get("grounded"):
            continue
        view.append({
            "canonical_id": str(entity.get("canonical_id") or ""),
            "name": str(entity.get("name") or ""),
            "aliases": [str(a) for a in entity.get("aliases") or []],
            "appearance": str(entity.get("appearance") or ""),
        })
    return [row for row in view if row["canonical_id"]]


def merge_pack_registry(saved_shots: dict, view: list[dict],
                        *, budget: int = 8000) -> str:
    """pack 候选表排前（canonical 为 id，可见名=名+别名+外观速写），窗口
    累积注册表排后去重——标注 prompt 的 registry 注入物。"""
    from src.agentic_video.narrative_index import build_entity_registry

    rows = []
    seen = set()
    for entry in view:
        names = [entry["name"], *entry["aliases"]]
        if entry["appearance"]:
            names.append(entry["appearance"])
        names = [n for n in dict.fromkeys(names) if n]
        if not names:
            continue
        rows.append({"entity_id": entry["canonical_id"], "visible_names": names,
                     "shot_idxs": []})
        seen.add(entry["canonical_id"])
    for entry in build_entity_registry(saved_shots or {}):
        if entry.get("entity_id") in seen:
            continue
        rows.append(entry)
        seen.add(entry.get("entity_id"))
    return json.dumps(rows, ensure_ascii=False)[:budget]


# ---------------------------------------------------------------- 主流程

def _load_scan(shots_dir: Path) -> tuple[Path, dict, dict]:
    """→ (video, envelope, output)。envelope 整体返回——aweme_id 在信封顶层
    （common.write_result_json），只回 output 会把备用读取链全部读空。"""
    result_path = shots_dir / "result.json"
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    return Path(envelope["output"]["video"]), envelope, envelope["output"]


def _source_duration_s(cfg: AppConfig, video: Path, envelope: dict,
                       scan_output: dict) -> tuple[float, str]:
    """片长三级读取链（外审三轮实锤的塌缩根因修复）。

    旧代码 duration 恒 0 的双层错位：索引 result.json 的 output 既无
    duration_s 也无 aweme_id（aweme_id 在信封顶层被读 output 层）；inspect
    的 duration_s 在 envelope.output 却被读信封根层。duration=0 时
    pick_representative_timestamps 的 hi=max(head+1, 0-360)=91 → 六帧全部
    塌缩 [90.1, 90.9]s（film2 实测复现）。主读链改 ffprobe 探视频文件本身，
    全部失败显式 raise——静默塌缩采样比崩溃恶劣。"""
    try:
        probed = float(common.video_duration_s(
            cfg.perception.get("ffprobe_bin", "ffprobe"), video))
    except Exception:                                          # noqa: BLE001 - 备用链兜底
        probed = 0.0
    if probed > 0:
        return probed, "ffprobe"
    if float(scan_output.get("duration_s") or 0) > 0:
        return float(scan_output["duration_s"]), "scan_result"
    aweme_id = str(envelope.get("aweme_id") or "")
    if aweme_id:
        inspect = common.read_result_json(
            cfg.paths.perception_dir / aweme_id / "inspect")
        output = (inspect or {}).get("output") if isinstance(inspect, dict) else {}
        if float((output or {}).get("duration_s") or 0) > 0:
            return float(output["duration_s"]), "inspect"
    raise RuntimeError(
        f"bootstrap: cannot determine duration of {video.name} "
        "(ffprobe/scan/inspect 全失败)——拒绝采样，静默塌缩 [90,91] 比崩溃恶劣")


def pack_validity(pack: dict | None) -> dict:
    """旧坏 pack 的失效门（外审三轮必改①：本轮不重跑 bootstrap，但坏 pack
    禁止注入标注器）。判据：代表帧采样跨度（provenance.timestamps 非空时
    max-min ≥ 300s——film2 塌缩 pack span<1s 直接 invalid）；tier 缺失；
    roster 空。timestamps 为空 = 走了人物帧路径（film1 实际生效路径，帧跨
    全片），无法判跨度，放行由 binding 层守。"""
    if not isinstance(pack, dict):
        return {"valid": False, "reasons": ["pack_missing"]}
    reasons: list[str] = []
    timestamps = []
    for value in (pack.get("provenance") or {}).get("timestamps") or []:
        try:
            timestamps.append(float(value))
        except (TypeError, ValueError):
            continue
    if timestamps:
        span = max(timestamps) - min(timestamps)
        if span < 300.0:
            reasons.append(f"sampling_span={span:.1f}s<300s (collapsed)")
    if not (pack.get("work_identity") or {}).get("tier"):
        reasons.append("tier_missing")
    if not (pack.get("entities") or []):
        reasons.append("roster_empty")
    return {"valid": not reasons, "reasons": reasons}


def bootstrap_film_knowledge(cfg: AppConfig, source: str, *, force: bool = False,
                             runner=None, mask_metadata: bool = False) -> Path:
    """编排入口。产物：library_dir/shots/{source}__narrative/knowledge_pack.json。

    mask_metadata=True（benchmark 模式）：Identity Gate 不看文件名/元数据，
    只用双投视觉识别——控制泄漏源测纯视觉贡献（产品模式默认全用便宜元数据）。"""
    bootstrap_cfg = cfg.library.get("film_bootstrap") or {}
    if not bootstrap_cfg.get("enabled", False):
        raise RuntimeError("library.film_bootstrap.enabled=false（默认），"
                           "开启后本模块才可用——不改变 V3 默认行为")
    shots_dir = cfg.paths.library_dir / "shots" / f"{source}__narrative"
    pack_path = shots_dir / "knowledge_pack.json"
    video, envelope, scan_output = _load_scan(shots_dir)
    duration, duration_source = _source_duration_s(cfg, video, envelope, scan_output)
    zones = cfg.library.get("source_zones") or {}
    timestamps = pick_representative_timestamps(
        duration, n_uniform=int(bootstrap_cfg.get("n_uniform", 6)),
        head_s=float(zones.get("head_s", 90)), tail_s=float(zones.get("tail_s", 360)))
    captions: dict = {}
    captions_path = shots_dir / "captions.json"
    if captions_path.exists():
        try:
            captions = json.loads(captions_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            captions = {}
    n_frames = int(bootstrap_cfg.get("n_uniform", 6)) \
        + int(bootstrap_cfg.get("n_informative", 4))
    # 人物帧优先（三轮冒烟实锤：时间均匀=风景、片头区=出品方名单——识别靠
    # 角色脸）；无 captions/无人物镜头才回退 时间均匀+分数加权。
    person_frames = pick_person_frames(scan_output.get("shots") or [], captions,
                                        n=n_frames)
    if person_frames:
        timestamps, informative = [], person_frames
    else:
        informative = pick_informative_frames(
            scan_output.get("shots") or [],
            n=int(bootstrap_cfg.get("n_informative", 4)), captions=captions)
    work_dir = shots_dir / "bootstrap"
    audit, slideshow, frames = build_slideshow(
        video, work_dir, timestamps=timestamps, informative_frames=informative,
        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    logger.info("[bootstrap %s] slideshow=%d 帧 审计图=%s", source, len(frames), audit)

    if runner is None:
        from src.perception.omni_runner import OmniRunner
        runner = OmniRunner(cfg.perception.get("omni") or {})
    prompt = IDENTITY_PROMPT.format(n_frames=len(frames))
    # 二次独立投票：A=前半帧，B=后半帧（各自独立识别，防单次高置信误认）
    half = max(1, len(frames) // 2)
    vote_a = parse_identity_answer(runner.watch(slideshow, prompt).text) \
        if len(frames) <= 1 else _watch_frame_subset(
            video, work_dir, frames[:half], prompt, cfg, runner)
    vote_b = _watch_frame_subset(
        video, work_dir, frames[half:], prompt, cfg, runner) if len(frames) > 1 else vote_a
    metadata_title = "" if mask_metadata else str(
        scan_output.get("title") or Path(video).stem)
    gate = resolve_tier(
        vote_a, vote_b, metadata_title,
        tier_high=float(bootstrap_cfg.get("tier_high", 0.85)),
        tier_low=float(bootstrap_cfg.get("tier_low", 0.4)))
    logger.info("[bootstrap %s] gate=%s A=%s(%.2f) B=%s(%.2f)", source,
                gate["tier"], vote_a.get("title"), vote_a.get("confidence", 0),
                vote_b.get("title"), vote_b.get("confidence", 0))

    roster = {"franchise_slug": None, "entities": [], "relations": [],
              "plot_hypotheses": []}
    tier = gate["tier"]
    if tier == "search":
        from src.library.film_search import get_provider
        provider = get_provider(cfg)
        if provider is None:
            # 无搜索通道：降 model_prior_unverified——title/实体都不进 prompt
            tier = "model_prior_unverified"
    roster_title = vote_a.get("title") or vote_b.get("title") or ""
    if tier == "metadata_prior":
        roster_title = gate.get("metadata_title_extracted") or roster_title
    if tier in {"model_prior", "metadata_prior"}:
        franchise = vote_a.get("franchise_slug") or vote_b.get("franchise_slug") or ""
        if not franchise:
            # 3 窗门控实锤：视觉拒认（metadata_prior 档）时 franchise_slug 恒空
            # → roster 全部落 char:unknown:* 命名空间（film1 pack 实况）。
            # 兜底：source 名剥离开头数字段（luoxiaohei1 → luoxiaohei）。
            franchise = re.sub(r"\d+$", "", str(source or "").strip())
        roster_prompt = ROSTER_PROMPT.format(
            title=roster_title,
            installment=vote_a.get("installment") or "",
            franchise=franchise or "unknown",
            max_entities=int(bootstrap_cfg.get("max_entities", 12)))
        roster = parse_roster_answer(
            runner.ask(roster_prompt, max_new_tokens=2048).text,
            franchise=franchise,
            max_entities=int(bootstrap_cfg.get("max_entities", 12)))
        grounded_entities, id_map = _ground_roster_canonicals(
            roster.get("entities") or [], franchise,
            load_entity_registry(cfg) if hasattr(cfg, "library") else {})
        roster["entities"] = grounded_entities
        roster["relations"] = _remap_relations(
            roster.get("relations") or [], id_map)
        if tier == "metadata_prior":
            for entity in roster.get("entities") or []:
                entity["source"] = "metadata_prior"      # 文件名来路如实标记
    pack = {
        "schema": KNOWLEDGE_PACK_SCHEMA,
        "prompt_version": BOOTSTRAP_PROMPT_VERSION,
        "source": source,
        "work_identity": {
            "title": roster_title,
            "installment": vote_a.get("installment") or vote_b.get("installment") or "",
            "tier": tier, "status": "proposed",
            "votes": {"a": vote_a, "b": vote_b},
            "gate": gate,
            "metadata_title": metadata_title,
        },
        "franchise_slug": roster.get("franchise_slug"),
        "entities": roster.get("entities") or [],
        "relations": roster.get("relations") or [],
        "plot_hypotheses": roster.get("plot_hypotheses") or [],
        "provenance": {
            "timestamps": timestamps,
            "informative_frames": [str(p) for p in informative],
            "slideshow_audit": str(audit), "slideshow": str(slideshow),
            "raw_head": str(vote_a)[:200],
            "duration_source": duration_source,
        },
    }
    pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    logger.info("[bootstrap %s] pack → %s（tier=%s entities=%d）", source,
                pack_path, tier, len(pack["entities"]))
    common.emit_status_line("ok", source=source, tier=tier,
                            entities=len(pack["entities"]),
                            pack=str(pack_path))
    return pack_path


def _watch_frame_subset(video: Path, work_dir: Path, frames: list[Path],
                        prompt: str, cfg, runner) -> dict:
    """A/B 子集各自拼 slideshow 再识别（独立投票的实现）。"""
    sub_dir = work_dir / f"vote_{len(frames)}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, frame in enumerate(frames):
        target = sub_dir / f"f{idx:02d}.jpg"
        shutil.copy2(frame, target)
        names.append(target)
    audit, slideshow, _frames = build_slideshow(
        video, sub_dir, timestamps=[], informative_frames=names,
        ffmpeg_bin=cfg.perception.get("ffmpeg_bin", "ffmpeg"))
    return parse_identity_answer(runner.watch(slideshow, prompt).text)


# ---------------------------------------------------------------- 验证回路

PACK_ENTITY_STATES = ("proposed", "supported", "verified", "conflict", "unobserved")


def update_pack_from_annotations(cfg: AppConfig, source: str,
                                 gt_path: Path | None = None) -> dict:
    """绑定状态机（bootstrap --verify）：

    proposed（pack 初始）→ 单窗 binding=supported → supported →（多窗 + 独立
    盲证据：人工 GT 文件）→ verified；标注侧报 binding=conflict → conflict；
    0 出现 → unobserved。"verified" 不由自我一致升级（循环论证），只认独立 GT。
    同时回填 entity_registry.auto.json 的 source_entities（不覆盖手写表）。
    """
    shots_dir = cfg.paths.library_dir / "shots" / f"{source}__narrative"
    pack_path = shots_dir / "knowledge_pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    annotations = json.loads(
        (shots_dir / "narrative_annotations.json").read_text(encoding="utf-8"))
    bindings_by_canonical: dict[str, dict] = {}
    source_entities: dict[str, set] = {}
    for shot_idx, row in (annotations.get("shots") or {}).items():
        if not isinstance(row, dict):
            continue
        for binding in row.get("bindings") or []:
            canonical = str(binding.get("canonical_entity_id") or "")
            status = str(binding.get("binding_status") or "")
            if not canonical:
                continue
            entry = bindings_by_canonical.setdefault(
                canonical, {"windows": set(), "confidences": [], "conflicts": 0})
            if status == "conflict":
                entry["conflicts"] += 1
            elif status == "supported":
                row_window = row.get("window_idx")
                entry["windows"].add(str(row_window))
                entry["confidences"].append(
                    float(binding.get("binding_confidence") or 0))
                local = str(binding.get("local_entity_id") or "")
                if local:
                    # V4：窗口作用域引用（库侧 id 各窗各自编号，不带窗口的
                    # 引用会把别的窗口同号 id 也归到同一 canonical）
                    scope = f"/w{int(row_window)}" if isinstance(row_window, int) else ""
                    source_entities.setdefault(canonical, set()).add(
                        f"{source}{scope}/{local}")
    gt = {}
    if gt_path is not None and Path(gt_path).exists():
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    verified_set = {str(v) for v in gt.get("verified_entities") or []}
    report = {"source": source, "entities": {}}
    for entity in pack.get("entities") or []:
        canonical = entity.get("canonical_id")
        observed = bindings_by_canonical.get(canonical)
        if observed is None:
            entity["status"] = "unobserved"
        elif observed["conflicts"] and not observed["windows"]:
            entity["status"] = "conflict"
        elif observed["windows"]:
            entity["status"] = ("verified" if canonical in verified_set
                                else "supported")
            entity["binding_windows"] = sorted(int(w) for w in observed["windows"]
                                               if str(w).isdigit())
            entity["binding_avg_confidence"] = round(
                sum(observed["confidences"]) / max(1, len(observed["confidences"])), 3)
        report["entities"][canonical] = {"status": entity["status"]}
    pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    _merge_auto_registry(cfg, source_entities)
    logger.info("[bootstrap %s] verify: %s", source,
                json.dumps(report["entities"], ensure_ascii=False))
    return report


def _merge_auto_registry(cfg: AppConfig, source_entities: dict[str, set]) -> None:
    """auto registry 并入 library_dir/entity_registry.auto.json——只补
    source_entities，不覆盖手写 config/entity_registry.json；load_entity_registry
    负责合并（手写别名优先）。"""
    auto_path = cfg.paths.library_dir / "entity_registry.auto.json"
    auto: dict = {}
    if auto_path.exists():
        try:
            auto = json.loads(auto_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            auto = {}
    for canonical, refs in source_entities.items():
        entry = auto.setdefault(canonical, {"aliases": [], "source_entities": []})
        existing = set(entry.get("source_entities") or [])
        existing |= {f"{ref}" for ref in refs}
        entry["source_entities"] = sorted(existing)
    auto_path.parent.mkdir(parents=True, exist_ok=True)
    auto_path.write_text(json.dumps(auto, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def load_pack(cfg: AppConfig, source: str) -> dict | None:
    pack_path = cfg.paths.library_dir / "shots" / f"{source}__narrative" / "knowledge_pack.json"
    if not pack_path.exists():
        return None
    try:
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return pack if isinstance(pack, dict) and pack.get("schema") == KNOWLEDGE_PACK_SCHEMA else None


def pack_sha(pack: dict | None) -> str:
    """参与标注断点键：pack 的 roster 内容变化 → 受影响窗口重标。"""
    if not pack:
        return ""
    return _stable_hash({"view": annotation_view(pack), "tier":
                         (pack.get("work_identity") or {}).get("tier")})
