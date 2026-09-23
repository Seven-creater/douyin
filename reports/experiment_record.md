# 实验1：冻结结构下的真实文本创作基线

## 1. 任务要求

### 1.1 总目标

在不新增 Creative Intent、不改变 `creative_structure_spec_v1` 的前提下，让真实文本模型依次生成一条 Theme → Story Blueprint → Screenplay，判断当前创作接口本身能产生什么。此实验不是图片、视频或创意质量验收。

### 1.2 输入与边界

- 唯一参考派生输入是冻结的 `CREATIVE-STRUCTURE-V1`，SHA `5a22fb5af29588b6220fbe9ccda751c922abe617fca1716b2f97f2942218a967`；`user_brief={}`。
- 三阶段各调用文本 Omni 一次；任一阶段校验失败即停止，不在同一运行目录重试。模型只得到结构及已经生成的上游产物，不得到参考视频、claim、Narrative Interpretation 或 Editing Grammar。
- 使用现有 `theme_candidate_v1`、`story_blueprint_v1`、`screenplay_v1` 契约。只进行确定性结构和来源边界校验，不进行故事质量评估、候选选优或媒体生成。

### 1.3 复现入口

代码版本：首次 `a9d38d0`，机械映射修订版 `2f1b309`。入口为 `scripts/run_real_text_baseline.py`，服务器 Python 3.10.20，GPU 0/1。两次运行分别保存在本地 `data/agentic_runs/r2d_real_text_baseline_20260923_a9d38d0/` 与 `data/agentic_runs/r2d_real_text_baseline_20260923_2f1b309/`，服务器对应目录位于 `/data02/usr/wangqihao/Demo/research/data/agentic_runs/`。每阶段保留完整请求、原始响应、模型计量及校验结果；失败运行不覆盖。

## 2. 实际做法与进展顺序

### 第一步：建立隔离的真实文本入口

新增独立实验入口，不修改 R2-D fixture 编排、生产 DAG 或结构规格。Theme 和 Story 的创意字段由模型产生；固定 ID、父 SHA 和 Story 的结构绑定由程序编译；Screenplay 的场景、动作、节拍和文本提示由模型产生，程序只编译固定元数据、来源和可确定的 trace。请求前执行来源边界检查。模型原始响应先落盘，再解析与验证。

本地完整测试在首次提交时为 `1129 passed, 1 skipped`；修订后本地相关测试 `18 passed`，服务器专项测试 `4 passed`。完整测试数字不代表修订版已重新跑全量。

### 第二步：首次真实运行（`a9d38d0`）

Theme 生成并通过结构校验。Story 原始内容包含城市地铁“鬼影”传闻、监控证据与记者澄清，但模型把 `R1_INFORMATION_UPDATE.prior/evidence/updated` 填成事件 ID，而现有契约要求固定组件 ID。`02_story/validation.json` 记录 `story_relation_binding_invalid`，运行在两次模型响应后停止；Screenplay 没有调用。原始 Story 响应 SHA256 为 `6f2af1c38baae2d4a20598cc0d127a5d442f716d1d59c5b467e5860bed715424`。

### 第三步：机械映射修订与第二次真实运行（`2f1b309`）

程序改由模型给出的 `event_relations` 生成固定结构绑定，不改模型的故事事件或关系。新运行目录中 Theme 与 Story 均通过现有确定性校验，并进入 Screenplay。Screenplay 原始响应包含 3 个场景、6 个 beat，但同一角色事件被绑定到多个 beat，触发 `screenplay_role_beat_ambiguous`。三次响应均已落盘；没有接受的 `screenplay_v1`。

### 第四步：核对内容与证据

第二次运行的 Story 把 `prior_event_id` 指向“发现监控录像”，`evidence_event_id` 指向“记者调查”，`updated_event_id` 指向“公众恐慌升级”。这些 ID 形式合法，但其描述并不呈现从旧解释经新证据到更新解释的关系。Screenplay 又把末段写成恐慌和文字 `THE GHOST IS REAL`，没有完成主题所写的“鬼影实为避寒者”的认知更新。因此本实验不把 Story 的结构 `PASS` 解释为语义成立，也不人工改写原始输出或放行媒体生成。

## 3. 出现的问题与解决过程

### 3.1 把固定契约字段交给模型导致非创意性失败

**现象：** 首次 Story 的 `R1_INFORMATION_UPDATE` 内容保留了叙事意图，但其三个角色字段填写事件 ID，触发 `story_relation_binding_invalid`。

**根因：** Prompt 要模型同时创作事件和重复一套固定组件 ID 约定；后者是可由事件关系唯一确定的机械映射。根因由原始响应及 `contracts.py` 的字段要求直接确认。

**解决：** 修订版仅要求模型输出事件和 `event_relations`，程序按关系字段生成 `structure_bindings`。保留原始响应供审计，不修饰故事文本。

**验证：** 第二次运行的 Story 通过同一现有校验，产生候选与 envelope，并进入 Screenplay；服务器专项测试 `4 passed`。这只验证了格式映射，不验证故事语义。

### 3.2 结构校验通过不能证明信息更新成立

**现象：** 第二次 Story 的校验为 `PASS`，但事件描述与 I0/E1/I1 的语义不匹配；Screenplay 的 6 个 beat 没有唯一覆盖所绑定的角色事件，并以未澄清的恐慌收束。

**根因：** 当前确定性校验验证 ID 存在、关系类型、引用与 distinctness，不判断事件描述是否真正蕴含“旧解释被新信息更新”。Screenplay 的重复事件引用另触发了明确的 trace 歧义。语义缺口是对已保存文本与契约条件的分析，不是模型审计结论。

**处理：** 不新增针对当前案例的答案型 gate，不把原始 Story/Screenplay 改写成合格产物；以 `BLOCKED` 保存本次负结果。

**验证：** `02_story/candidate.json` 中可直接核对三项事件描述与绑定；`03_screenplay/raw_response.txt` 保留全部 6 个 beat；`result.json` 记录 `screenplay_role_beat_ambiguous`、3 次有响应的模型调用和 `production_committed=false`。

## 4. 简洁实验报告

### 4.1 主要结果

| 运行 | Theme | Story | Screenplay | 有响应模型调用 | 终态 |
|---|---|---|---|---:|---|
| `a9d38d0` | 结构通过 | 固定 ID 映射失败 | 未调用 | 2 | `BLOCKED` |
| `2f1b309` | 结构通过 | 结构通过，语义不成立 | 已生成原始响应，trace 歧义 | 3 | `BLOCKED` |

第二次运行计量：Theme `821/307`、Story `1152/522`、Screenplay `1578/968` input/output tokens；合计 `3551` input、`1797` output，模型调用耗时合计 `127.16 s`，不含模型加载。三份第二次原始响应的 SHA256 分别为 `ffcbd904b9ea2b6dc7b618d61997fb17fe31ca380d9d73ccbe97214f921be518`、`40df14d20b2b957e9a615ad02e832355b4dff8fe7a23a9d638c6e63dd91fbc23`、`70d7925ae2744b03d5ad862a13d79d1c1bc3c5734e7df5e6321a5b4782996fce`；本地副本与模型调用记录逐一核对相同。

### 4.2 结论

在本次单链、空 brief、无 Intent、无质量选优的条件下，模型能产生具体 Theme 和 Story 文本，但当前系统未得到通过契约的真实 Screenplay。更关键的是，现有 Story 结构 `PASS` 并不保证语义上的信息更新。此单例负结果既不能证明 Creative Intent 必要，也不能证明它无效。

### 4.3 尚未完成

- 没有接受的真实 Screenplay、人工或模型创意质量评估，也没有图片、视频生成。
- 没有多随机种子、Best-of-N 或“有 Intent”对照；不能据此估计成功率或因果改进。
- 本实验只保留并报告失败，不对第二次 Story 或 Screenplay 做事后修正。
