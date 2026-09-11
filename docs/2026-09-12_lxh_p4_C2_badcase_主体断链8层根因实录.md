# 2026-09-12 lxh_p4_C2 Badcase：主体断链 8 层根因实录与 V3 修复

## 结论先行

C 档（need+验证+重搜）闭环机械上全部转通，但成片盲看失败：**字幕在讲一个故事，
画面在播三个不相关片段**。外部分析 + 双路核证（产物 6 断言全属实）定性：
不是单模块出错，是**一串错误被后续模块不断"合法化"**；本质病灶是
**叙事主体不连续**——三槽三个主角（仓库三妖 / 紫发施法者 / 白发男+倒地小孩），
"谁"都在换，"发生了什么/为什么/结果"必然不成立。

$$主体稳定 > 事件连续 > 因果成立 > 语义匹配 > 视觉效果$$

分析包：`桌面 lxh_p4_C2_badcase_20260912.tar.gz`（65MB，含 Omni 全链输入输出
+ README badcase 地图）。

## 根因链（8 层，每层已核证）

| # | 层 | 核证证据 | 为什么没被拦住 | 修复归属 |
|---|---|---|---|---|
| 1 | 文本显著锚定：开场字幕 OCR 同句×5 + ASR×1 灌入 Omni | ocr.json 0~4.178s 每秒 1 帧共 5 次 | build_narrative_material 无去重聚合 | P0 |
| 2 | 固定时长切窗压扁多事件：21.934s 均分 2×10.967s，跨镜头边界 7.133/14.033 | narrative_agent initial 窗口区间实录 | plan_narrative_windows 不读任何信号边界 | P0 |
| 3 | 窗口层过早赋 story_role + 观察不稳定：同窗两次观看 hook→context、拥抱 vs 户外微笑 | 各窗 result.json observations | Observation 与 Interpretation 未分离 | P0 |
| 4 | 薄参考弧（2 events/0 causal）被强补 conflict：故事语法选错（断言→反证型被套"冲突"弧） | story_plan arc_expanded 字段 | _expand_thin_arc 自动补弧 | P1 |
| 5 | 参考事实泄漏 target need（跆拳道服女性）→ 罗小黑库必然检索扑空 | 槽 reason 字段原文 | 4835df2 修在 C2 跑完之后 | P1（form 层结构隔离） |
| 6 | 检索粒度错位：事件聚合 caption vs 实际 7.33s 切片（caption 说奔跑、画面是施法） | story_candidates 36 条 causal 全空；槽1 caption≠画面 | 事件=镜头并集但裁剪从事件起点盲切 | P2 |
| 7 | 单槽验证全 PASS 但无全局：三槽三主角各自"像那么回事" | verification_round_2 三 PASS vs 成片盲看失败 | Local Correctness ≠ Narrative Coherence | P3 |
| 8 | 文案替乱剪讲故事 + 裁判失效：punchline「可它还是救了他」无救助画面；critic 理解题自评 5/5 与连贯性 0 同文件矛盾；edit critic 幻觉出参考片的跆拳道画面 | copy cues / narrative_critic_round_2 / edit_critic_round_2 原文 | copy 早生成无证据绑定；critic 拿全文 Program+Plan 自评 | P4 |

另：罗小黑 WEB-DL 内嵌中文字幕与烧录文案同屏打架（盲看实锤「你是要」）→ P4 渲染侧裁字幕带。

## V3 修复总览（2026-09-12 实施完毕）

**主线**：参考片 → 跨模态文本合并 → hard/soft 边界切窗 → 事实观察 →
Narrative Form（表达结构+Entity Role Schema）→ **Entity Continuity Contract**
（主角锁定，身份约束>相似度）→ 事件级检索（canonical 实体注册表跨片同人）→
硬约束过滤→软排序 → 槽级验证 → **Deterministic Global Check**（零模型）→
渲染 → **Blind Video Check**（零上下文）→ **copy 最后生成并逐句绑证据**。

- **P0 参考理解**（narrative_agent.py）：dedupe_text_signals 跨模态合并（同句
  只出现一次，保留 OCR×5+ASR×1 元数据）；plan_narrative_windows hard（镜头）
  /soft（OCR/ASR）切点对齐，target_window_s 降级合并上限；WINDOW_PROMPT v2
  删 story_role；material_mode 四条件消融（A 纯视频/B +合并OCR/C 现状/D 纯文本）。
- **P1 叙事形态**（narrative_form.py 新增 + story_planner.py）：删 _expand_thin_arc；
  NARRATIVE_FORMS 映射表（7682→assertion_visual_payoff：premise→counter_evidence
  →expansion→payoff）；entity_schema（protagonist global、supporting local）+
  每槽 entity_requirements；_rank_under_contract 主角假设枚举——首个主角槽选材
  即锁定，主角槽 allowed 硬过滤（0.99 分冒名者输给 0.5 分主角，测试钉死）；
  protagonist 持续性禁止 relaxed。
- **P2 事件粒度**（story_planner.py + entity_registry.py 新增）：merge_event_candidates
  带成员镜头明细 + 同窗时序前驱（库内因果普遍为空，不造假）；_select_evidence_window
  预算内覆盖证据重合度最高的连续镜头子序列（窗口跟证据走，caption 只留裁入镜头）；
  config/entity_registry.json canonical 注册表（小黑/无限/风息…别名+源内ID双路），
  两片同人归一。
- **P3 双层验证**（verify_slots.py + pipeline.py）：deterministic_story_check 零模型
  三指标（Protagonist Switch Count=0 / Required-slot Presence=100% / Unexplained
  Transition=0），失败槽进契约内重搜；blind_video_check 零上下文盲看成片；
  re_search 受主角契约约束（换件不换人，枯竭→unsupported 不换角）。
- **P4 文案与裁判**（copywriter/critic/pipeline/renderer/config）：copy 移到收敛后
  最后生成，cue 绑 {slot_idx, subject_entity, evidence}；validate_copy_grounding
  无证据丢弃/动作断言无画面即换无断言兜底（「可它还是救了他」负例测试钉死）；
  narrative_critic 删自评 correct、全文 Program+Plan 改 ≤2000 字槽位事实摘要；
  edit critic 只看 Recipe 骨架 + reference_terms 污染过滤（跆拳道条弃用测试钉死）；
  luoxiaohei1/2 渲染侧裁底部 12% 字幕带（索引侧保留）。

## 验收指标（确定性，不依赖 critic）

| 指标 | 目标 |
|---|---|
| Protagonist Switch Count | 单主角 form 必须 0 |
| Required-slot Protagonist Presence | 100% |
| Unexplained Entity Transition Rate | 0 |
| punchline 有据率 | 100%（无据即被 grounding 替换/丢弃） |
| 盲看五问 | 用户能复述故事主线（终审） |

## 遗留与后续

- 消融四跑（A/C/D 主判 text-dominant，B 测 ASR 边际）→ 结果单独进 docs；
- 负例四连验收：无救助 punchline 应拒 / 参考词 issue 应弃 / 跨槽换主角应
  Deterministic FAIL / 主角素材枯竭应 unsupported；
- 注册表 source_entities 待夜链 1080p 重标完成后按新 entity_id 补录；
- guimie B/C 因卡争用丢失的对照已由 V3 重跑取代。
