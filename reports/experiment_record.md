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

# 实验2：Story 直接修订与 Critic→修订的受控单例对照

## 1. 任务要求

### 1.1 总目标

在不修改 R2-D 生产 DAG、不增加 Creative Plan 或整套 Reviewer Layer 的条件下，检验一次独立 Story Critic 是否比同等调用次数的直接修订更能纠正已观察到的信息更新语义绑定错误。

### 1.2 共同输入与边界

两臂共享实验1第二次运行的同一 Theme 和 Story（Story JSON SHA256 `5204f7025ec99561a1c31a1e1ff0dcd5b2b473c41a537c4ac761314bfd3f4dba`），及冻结的 `CREATIVE-STRUCTURE-V1`。A 臂为直接修订两次，B 臂为独立 Critic 一次、依据反馈修订一次。随后对最终两个 Story 做左右顺序互换的匿名比较两次。每个模型调用只执行一次，无 retry；保留请求、原始回复和计量。模型输入只包括公开结构、Theme、当前 Story 与实验约束，不包括参考视频及 Reference Zone 证据。仅生成文本 Story；无 Screenplay、媒体或生产提交。

### 1.3 复现入口

独立代码入口为 `scripts/run_creative_critique_trial.py`；模型为服务器 Python 3.10.20 环境下的文本 Omni，GPU 0/1。三个按版本隔离的 run 位于 `data/agentic_runs/r2d_critique_trial_20260923_7989e4e/`、`..._3223bc4/`、`..._6a7fc78/`，各目录不覆盖。最终有效对照为 `6a7fc78`；前两次为保留的失败/受污染 pilot，不并入有效对照。

## 2. 实际做法与进展顺序

### 第一步：隔离实验入口和静态校验

新增 `critique_trial.py` 与 CLI，不接入生产 workspace。相同 Story 编译与现有结构校验用于两臂；Critic 与 Judge 的原始响应独立保存。每个请求保存完整 prompt、payload SHA、token 上限；每个响应保存原文、SHA、token 与耗时。本地全量测试 `1133 passed, 1 skipped`，服务器专项测试 `7 passed`，均在首次真实运行前完成。

### 第二步：首轮格式失败（`7989e4e`）

直接修订第 1 次的回复把 `event_relations` 写成对象而非契约要求的单元素数组；现有 `_compile_story` 报 `story_event_relations_invalid`，运行在 1 次有响应调用后停止。原始回复及失败记录未覆盖。原因是实验 prompt 写了“exactly one object”但未明确“array containing exactly one object”；后续只澄清 JSON 容器形状，不放宽 schema。

### 第三步：第一次完整 pilot 暴露输入污染（`3223bc4`）

六次调用均完成，两次交换顺序的 Judge 都给 `tie`。但实验输入把运行状态 `text_only=true`、`media_generation=false` 误放进了创作约束，Critic、修订器和 Judge 将其解释为“故事必须纯文本交付”，产物及评价出现 voiceover/text-only 假设。这使 pilot 不适于判断正常短视频创作质量。修订仅从模型 payload 删除这两个运行标记；输出元数据仍记录未生成媒体。

### 第四步：干净单例对照（`6a7fc78`）

新目录运行全部六次调用，现有 Story 契约均通过。A 臂最终仍将 `I0` 绑定到“发现录像”，将 `I1` 绑定到“群众准备攻击”；B 臂最终将 `I0` 绑定到“公众相信鬼魂”、`E1` 绑定到“录像显示避寒者”、`I1` 绑定到“记者意识到真人处境”。两次匿名 Judge 分别在左右顺序下选择 `right`、`left`，均指向 B 臂。此结果也可直接从两个最终 `candidate.json` 的事件描述与 `event_relations` 核查，而不只依赖 Judge 文本。

## 3. 出现的问题与解决过程

### 3.1 JSON 形状歧义使对照在首调用中止

**现象：** `7989e4e/01_direct_revision_1/raw_response.txt` 中 `event_relations` 是对象；`result.json` 为 `BLOCKED`，原因 `story_event_relations_invalid`。

**根因：** 实验 prompt 对“恰好一条关系”的措辞未说明数组容器，而 Story 契约需要列表；这是提示词与契约不一致，非质量判定。

**解决：** 仅明确输出为“a JSON array containing exactly one object”。

**验证：** 后续两个独立 run 的 Story 输出均能通过同一个 `_compile_story` 和 `validate_story_structure`；没有修改 Story schema 或失败目录。

### 3.2 实验执行标记污染创作评估

**现象：** `3223bc4` 的 Critic 把“不生成媒体”当作生产可行性优点；Judge 认为两组都是“text-only delivery”，均判平局。

**根因：** `text_only` 与 `media_generation` 本意是本次实验的执行范围，却被放进所有模型请求的 `experiment_constraints`。

**解决：** 模型侧约束只保留 30 秒目标与最多 3 场；未生成媒体只在运行结果元数据中记录。

**验证：** 新增测试断言六个请求不含这两个标记；`6a7fc78` 的实际请求与回复不再以“纯文本交付”作为创作条件。本次纠正不改变结构 spec、Theme、基线 Story 或两臂算法。

### 3.3 结构 PASS 掩盖语义绑定错误

**现象：** 干净 run 中 A、B 的 Story 均通过确定性结构校验，但 A 的 `prior_event_id` 指向录像发现，`updated_event_id` 指向群众攻击；B 则形成可辨认的旧解释→证据→更新解释。

**根因：** 现有校验只验证字段、引用和关系形式，不检验事件文本是否承担其所绑定的信息角色。这一点已在实验1观察到。

**处理：** 此对照没有新增 answer-shaped gate。独立 Critic 明确指出“录像发现不是旧解释、群众恐慌不是更新解释”；修订器根据反馈产出另一 Story。Critic 的个别具体建议仍有瑕疵，因此不把其每句话都当成正确答案。

**验证：** `6a7fc78/03_critic/critique.json`、两份最终 `candidate.json` 和顺序互换的两份 `judgment.json` 提供相互可核查的原始证据。Judge 只是一台同源模型的重复排序，不构成独立人类 Gold。

## 4. 简洁实验报告

### 4.1 主要结果

| Run | 有响应调用 | 终态 | 比较结论 |
|---|---:|---|---|
| `7989e4e` | 1 | `BLOCKED` | 首调用 JSON 形状失败，不比较 |
| `3223bc4` | 6 | `PASS`，但输入受污染 | Judge 两次平局；不纳入有效对照 |
| `6a7fc78` | 6 | `PASS` | 两次换序均选 Critic→修订，且角色绑定可人工核查 |

干净 run 的 A 臂两次直接修订为 `3560` input、`1025` output tokens，模型耗时合计 `76.09 s`；B 臂 Critic 加修订为 `4083` input、`1133` output tokens，耗时 `83.32 s`。两次 Judge 另用 `4686` input、`670` output tokens，耗时 `49.70 s`。六次调用合计 `12329` input、`2828` output tokens，模型调用耗时合计 `209.11 s`，不含加载；B 臂比 A 臂多 `523` input、`108` output tokens 和 `7.23 s`，所以并非等 token 成本对照。最终 run 的输入 SHA、两臂 Story SHA、顺序结果见其 `result.json`。

### 4.2 结论

本次一份真实基线 Story 的受控探索中，Story Critic 能指出确定性 validator 漏掉的语义角色错配；Critic→修订产物修正了该错配，而两次直接修订未修正。两次同模型、交换顺序的 Judge 与这项可直接核对的差异一致。证据支持继续检验“语义 Reviewer 是否有用”，**不支持**据此直接上四类 Reviewer、20 个 Theme 或生产级自动修复循环，也不证明整体原创性、观众吸引力或视频质量提高。

### 4.3 尚未完成

- 仅一份 Theme/Story、一个模型配置；没有随机种子或跨题材重复，无法估计成功率或泛化。
- Critic、修订器、Judge 使用同源模型，Judge 不是独立 Gold；没有人工盲评。
- 未运行 Screenplay、Storyboard、媒体生成或视频评价；两组仅是 Story Blueprint。
- 未把实验入口接入生产 DAG；`production_committed=false`，`media_generation=not_run`。
