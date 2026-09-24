# 实验：参考主旨的跨题材关系抽象 v5

## 1. 任务要求

### 1.1 总目标

检验 Qwen3-Omni 能否从已冻结的参考片理解中抽取可跨题材迁移的观众判断与证据关系，而不把原片人物设定变成创作硬条件。v4 的旧、新理解各处理一次；不重看视频，不修改旧 run，不生成剧本或媒体。

### 1.2 验收边界

先验证独立评审能否指出 v4 公开简报的来源绑定，再运行 v5 抽象、来源审核、公开审核及四例开发回归。只有这些环节均通过才生成六个诊断主题；本次失败应原样保留并停止。C01–C04 的故事与标签不进入抽象请求。

## 2. 实际做法与进展顺序

### 第一步：冻结输入并实现独立 v5 路径

实现见 [reference_message_v5.py](/C:/Users/29785/Desktop/douyin/src/agentic_video/reference_message_v5.py) 和 [run_reference_message_v5.py](/C:/Users/29785/Desktop/douyin/scripts/run_reference_message_v5.py)。v5 输入不再复制 `source_statement` 到 `takeaway_candidate`；私有 kernel 保存来源绑定、证据 ID、匿名功能和关系，公开简报由白名单构造。旧理解和新理解的来源 SHA、提示词 SHA 保存在 [input_lineage.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/input_lineage.json)。代码在 `codex/reference-message-transfer-v5`，当前提交为 `d59dab6`。

### 第二步：诊断冻结的 v4 简报

第一次诊断的 [result.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/diagnose_run_001/result.json) 是 `judge_unreliable`：旧 v4 portability 评审把两份仍含身体设定的公开字段全判为 `portable`。因此没有直接进入 v5 主题生成。改用已经校准的逐字段 mapping 评审，对冻结简报与 C01/C02 跨题材故事做诊断，结果见 [diagnose_run_002/result.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/diagnose_run_002/result.json)：旧、新简报在两个案例的 `audience_takeaway` 均被拒，诊断通过。此处只证明评审能识别该已知 bad case，不证明其普遍可靠。

### 第三步：各运行一次 v5 文本抽象和来源审核

两个分支各产生一份 kernel 和一份审核；原始请求、原始响应、模型配置及 token 记录均在 [trial_run_001/calls](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/calls)。两份 kernel 通过 JSON/schema 检查，但两份来源审核都只返回 `source_claim` 的一条检查，未覆盖要求的其余字段。因此 [trial_run_001/result.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/result.json) 为 `grounding_failed / grounding_audit_invalid`，公开审核、四例回归和主题生成均未执行。

## 3. 出现的问题与解决过程

### 3.1 v4 独立可迁移审核假阳性

**现象：** 第一次重放将两份含“双手／身体部位”必要条件的公开简报判为全部可迁移。原始判定在 [diagnose_run_001/calls](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/diagnose_run_001/calls)。

**根因：** 从响应可确认评审宽松地改述了字段含义，而非检查原句在跨题材故事中的映射；它也在替换情境中引入了原片细节。仅靠抽象器自编情境或该 portability 布尔值，不能作为独立门槛。

**解决：** 复用既有的、8 例两遍校准结果，改为逐字段核对 v4 原句与 C01/C02 具体故事。案例仅进入评审请求，不进入抽象请求。

**验证：** 第二次诊断中，旧、新两份简报各对两个案例的 `audience_takeaway` 判为无法映射，共 4/4 个已知冲突被指出；没有发布新简报。

### 3.2 v5 抽象仍携带原片题材

**现象：** [old_kernel.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/old_kernel.json) 的公开候选 `audience_goal` 仍写“subject as disabled and limited”；角色功能写“societal assumption about disability”。[new_kernel.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/new_kernel.json) 的 `audience_goal` 仍写“losing hands defines a person as useless”。两者 `free_slots` 都为空。

**根因：** 输入重复锚定虽已去除，单次抽象仍没有把来源特征留在私有绑定层。模型输出表明“匿名 role ID”本身不足以保证 role function 和观众目标不带来源名词；这是本次观测，不据此推断模型永远做不到。

**解决：** 本轮按预注册方案保留失败结果，不根据本片或四例答案现场修改抽象提示词，也不把这些 kernel 发布为公开简报。

**验证：** 两份原始 kernel 均可直接定位上述来源绑定字段；后续公开审核未运行，因此不能声称已有独立迁移判决。

### 3.3 来源审核未覆盖全部字段

**现象：** 旧 kernel 应审核 8 个路径，新 kernel 应审核 11 个路径；[old_grounding.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/old_grounding.json) 与 [new_grounding.json](/C:/Users/29785/Desktop/douyin/data/agentic_runs/reference_message_v5_20260924/trial_run_001/new_grounding.json) 各仅返回 1 个 `source_claim` 检查。程序因此拒绝两份结果。

**根因：** 已确认模型未按请求中的字段列表逐项作答。提示词虽写“每个路径一条”，但给出的 JSON 示例只有 `source_claim` 一条；示例诱导是可能原因，尚未用受控实验验证。

**解决：** 不补造剩余 7/10 条结论，不把一次检查视为全量审核；保留失败并停止。此次没有为了通过而重试或改 prompt。

**验证：** `validate_audit` 严格比较实际路径顺序和数量，两支均得到 `grounding_audit_invalid`，主题数为 0。

## 4. 简洁实验报告

### 4.1 主要结果

| 环节 | 实际结果 |
|---|---|
| v4 旧诊断 | 2 次模型调用；错误放行两份来源绑定简报 |
| 校准后逐字段诊断 | 1 次模型调用；4/4 个预期的 `audience_takeaway` 冲突被指出 |
| v5 抽象 | 旧、新各 1 次；两个 kernel 均通过 schema，但均保留身体设定 |
| v5 来源审核 | 旧、新各 1 次；分别只返回 1/8、1/11 个必需检查 |
| 后续公开审核、四例回归、主题 | 均未运行；主题 0 个 |
| 视频调用与生产发布 | 0 次；`production_release_allowed=false` |

v5 试验的 4 次模型调用合计 `9,728` 输入 token、`1,504` 输出 token、`109.35` 秒；不含两次诊断的 3 次调用。来源输入与配置均可由各调用的 `model_call.json` 和 `request.txt` 复核。本地定向测试为 `19 passed`；初版代码的全量本地测试为 `1131 passed, 1 skipped`，诊断器修订后未重跑全量。

### 4.2 结论

本次确认了两个独立断点：旧评审确实会忽略来源限定；校准后的逐字段映射能抓到这一已知问题。但 v5 单次抽象依然携带来源题材，且审核响应覆盖不足。不能将这轮称为跨题材主旨抽象成功，也不能据此生成或筛选主题。

### 4.3 尚未完成

公开层迁移审核、C01–C04 回归、六个诊断主题、第二条不同机制参考片与独立人工评审都未执行。需要在另一次预注册试验中分别检验审核覆盖问题和抽象字段的来源残留；本轮冻结，不在失败产物上补写结论。
