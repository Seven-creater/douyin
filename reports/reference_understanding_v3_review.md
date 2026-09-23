# 参考视频理解 v3：受限媒体复核对照记录

日期：2026-09-24。判定：**未通过参考理解验收，不可进入主题、剧本或剪辑生产链。** 这是单片、同一 Omni 模型的试验，没有人工真值，也没有跨参考片泛化测试。本文的“支持”“不足”是程序对证据定位和模型契约的判定，不等于视觉语义准确率。

## 冻结输入与执行边界

- 参考原片 SHA256：`2f95e24edd2cf4b79cc1f40f7e202174c53a6abcf084e728bb49e3ca92938a17`。使用原有 31 条 accepted claims、10 个 accepted events、17 个内容镜头、16 个相邻编辑边和 6 段末段转场；这些均为既有输入，不是 v3 新检出结果。v2 对照固定为 `reference_understanding_v2_20260923/model_run_003`。
- 代码在 `codex/reference-understanding-v3`，最终真机代码提交为 `aa0e210`。仅新增 `reference_understanding_v3.py`、独立运行脚本及测试；没有改动冻结的 `creative_structure_spec_v1` 或主题、剧本、图片、视频生产代码。GPU 仅使用服务器 0/1，未使用 2/3。
- 完成目录：`data/agentic_runs/reference_understanding_v3_20260924/model_run_004`。`model_run_001`、`002`、`003` 是保留原始 request、response、采样帧与失败点的未完成尝试，未覆盖或改写。完成目录的 8 条局部媒体证据中，前 5 条精确复用 `model_run_003` 原始 trace，经 request、媒体 SHA 和计划范围核对；后 3 条为新增模型调用。复用链见每条 `replay_provenance.json`。
- **预算解释必须诚实：** 完成目录的证据集恰好 8 次局部媒体调用，但计入前三次失败尝试，整个开发/试验过程实际发生了 12 次不同的媒体调用（1+3+5+3）。因此它不能被表述成“一次端到端 8 调用即成功”的干净实验。失败重启源于本轮接口/校验实现缺口，不是视频内容方面的新证据。

## 实际执行机制

`validated_reference` + 原片 + 静音遮字副本 → `prepare_readout()` 测量时轴与音频候选 onset → `plan_probes()` 先覆盖跨镜粗动作及动作到后继状态，再检查结尾和剩余编辑边 → `_call_media()` / `_replay_media()` 保留 request、原始响应、真实采样时间和重建帧 → `validate_local_observation()` 将越界/未采样的状态变化判不足 → `build_editing_graph()` 把确定性剪辑事实和语义边状态分开 → 一次故事图、一次剪辑功能、一次文本一致性检查 → 三个隔离的文本反事实读数 → 仅在故事与一致性检查都过关时才尝试私有主题简报。本次未达到该门槛。

感知请求不含人工参考答案；V 请求使用无音轨遮字片段，AV 请求仅在局部片段中读取原声，并把 T 文字明确标为“视频作出的陈述”。每条媒体请求的实际采样帧索引、相对/原片时间码与重建预览均在 `calls/probe_*/sampling_audit.json` 和 `sampled_frames/`。重建图是按同一帧索引从源片另行解码，不是模型输入 tensor 的逐字节备份。

## v2 → v3 可测对照

| 项目 | 冻结 v2 | 本轮 v3 完成证据集 |
|---|---:|---:|
| 局部媒体调用 | 7 | 8（5 复用、3 新增） |
| 关键粗动作覆盖 | C02–C04，3/5 | C02–C06，5/5 |
| 同问题/范围/方式重复请求 | 1 | 0 |
| 16 条编辑边状态 | 7 provisional、2 contested、7 unobserved | 8 provisional、1 contested、7 insufficient |
| 全部模型调用 | 12 | 14（含 5 条复用 trace、3 条文本反事实） |
| 输入/输出 token | 75,190 / 7,474 | 160,126 / 11,988 |
| 模型调用耗时合计 | 544.37 秒 | 939.03 秒 |

以上来自 `v2_v3_comparison.json`。v3 的结构、调用任务与反事实调用不同，token/耗时是成本记录，不是严格等量的算法效率对照。报告中的 `localized_change_count=20` **只是时码/采样门槛下保留下来的模型主张数，绝不是 20 次正确动作检出**。

## 关键 bad case 与下游影响

1. **比赛动作与结果仍未被可靠定位。** v3 已覆盖 C02–C08，但原始输出仍把某些“踢中导致倒下”写为确定因果。`probe_03` 将倒下首次变化标在片段 0 秒；C07 的变化也写成 0 秒，落在它自己的镜头之外。越界主张被判 `insufficient`，`probe_08` 以不同采样率做中性复核却没有给出可定位的新变化。镜头覆盖率上升不等于动作因果准确率上升。若此处直接进入剧本/剪辑，容易把“准备—若干踢击—首次可见结果”的节拍错排成虚构连续动作。
2. **短镜头限制部分生效，但语义误认仍在。** S3 C01 仅 0.1 秒，`probe_06` 对其运动变化只得到一帧，程序判为不足。另一方面，模型在较长片段里仍把多张照片、场景变化和人物身份混淆；`model_run_002/probe_03` 甚至把 C05–C08 四镜几乎全部说成同一段微笑特写。v3 的时间/采样门槛挡住了明显越界时码，却不能证明时码落入镜头的描述本身是真实的。人物身份和照片关系仍不可用于创作绑定。
3. **故事图没有提炼出主题机制。** 完成轮输出 10 个 events，基本复述了既有事件；7 条关系大多只是“先后出现/场景转到”，没有解释开头视频文字提出的泛化判断如何被后续证据改变。两个 stance 分别只有 `Subjective` 和 `Objective`，`ending_relation` 仅指末幅风景照之间的 R7；结尾“但不会剪脚指甲”的语气作用未进入故事解释。故事图还把“踢击→对手卧倒”写成确定因果，甚至称作胜利。`story_validation.json` 报 `event_local_ref_unknown`、`relation_unsupported`、`stance_local_ref_unknown`；文本一致性模型也给出 `internally_consistent=false`。这会直接导致主题偏移，不能给编剧 skill 使用。
4. **剪辑事实保住，语义功能未过关。** `editing_graph_v3.json` 保留了既有的时长、切点、转场和音频未验证状态；`beat_synced_status` 仍为 `unverified`。`editing_functions_v3.json` 尽管通过 ID/边引用格式校验，却有“男选手开始旋转后踢”“镜头摇到女方反应”等未被可靠媒体证据支持的叙述，并漏掉 1 条相邻边的函数输出。结构校验通过不能证明功能解释正确；逐秒模仿参考片的信息选择仍不可用。
5. **反事实检查只给出弱敏感性。** 遮去末尾文字后，模型没有逐字复述它；改变 section 顺序后，输出也不是逐字相同。但三个版本的 `ending_reading` 都是 `unknown`，对顺序变化仍给出相近的励志/赋能立场。故 `counterfactual_diagnostics.json` 不能证明模型理解了结尾或叙事时序，更不能作为准确率。

文本一致性检查自身也产生了可疑推断（例如把归因/身份争议说成某项视觉事实的反证），所以其阴性结果只是另一条诊断证据，而非人工真值。`editing_function_validation` 的 PASS 只检查结构及引用，不是媒体语义 PASS。

## 结论与产物入口

v3 的**取证纪律**比 v2 更好：粗动作镜头覆盖完整、无同范围同方式重复、实际采样帧可追溯，越界时间和 0.1 秒镜头运动不再被直接当作已支持事实。但**理解质量没有达到目标**：动作因果、人物身份、主题立场/结尾作用及若干剪辑功能仍错误或未决；调用成本反而增加。模型自审不能补足独立真值。本轮没有生成 `theme_brief_candidate_v3.json`，也没有运行任何主题、剧本、图像或视频生成。

阅读顺序：`reference_understanding_v3.json`（封锁状态）→ `v2_v3_comparison.json`（数值对照）→ `story_graph_v3.json` / `story_validation.json`（主题 bad case）→ `editing_graph_v3.json` / `editing_functions_v3.json`（剪辑）→ `calls/`（逐次原始 request/response、采样、replay provenance）。本地复制的完成结果文件 SHA256 为 `c85e1be0e4a5ddd0e9201f0e861060460987c7865b8a3bf9690d300526264d86`，与服务器一致；JSON 内部 artifact SHA 为 `c1a81a7aecd73f566fcf6a1398c06009c72fbbc2d31b3e1949b788c5fea923ce`。

所有正式开关为 `story_ready=false`、`editing_ready=false`、`production_release_allowed=false`、`human_truth_confirmed=false`。下一步若要研究改进，应以这些原始 trace 为依据，先补独立的人类时码/主题核验，再比较更短的邻镜观察和更严格的故事抽象是否真有收益；**本轮不把这种建议实施进冻结生产链**。
