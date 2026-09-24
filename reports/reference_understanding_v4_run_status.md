# 参考视频理解 v4：受控调用状态（盲审前）

日期：2026-09-24。代码提交 `8ee77ef`，分支 `codex/reference-understanding-v4`。这是独立实验，不接主题、剧本、图片或视频生产链；旧 v1–v3 run、31 条接受证据和确定性时轴未被修改。

## 冻结输入与运行

- 原片 SHA256：`2f95e24edd2cf4b79cc1f40f7e202174c53a6abcf084e728bb49e3ca92938a17`。输入含 31 条 accepted claims、10 个 accepted events、既有 17 个内容镜头、16 条相邻边和 6 段转场；这些是输入，不计作 v4 检出成绩。
- 服务器首次环境预检失败保留于 `model_run_001`（缺少 `qwen_omni_utils`）；第二次保留于 `model_run_002`（`torchvision::nms`/PyTorch 环境冲突）。两次均未生成模型响应，也没有复用为完成轮证据。
- 完成轮是 `data/agentic_runs/reference_understanding_v4_20260924/model_run_003`。Qwen3-Omni thinker-only 使用服务器 `omni_src` 环境、GPU 0/1；GPU 2/3 未使用。恰好 12 次调用：两次原声原片全片、8 次原声局部（首批 6、不同采样率复核 2）、一次文本剪辑功能、一次文本证据审核。单层无重试。两次全片的提示词、模型配置 SHA 与媒体 SHA 相同；差异是第二次增加经采样审计的局部观察。
- 完成轮总输入/输出 token：205,925 / 15,848；模型调用耗时合计 1,212.85 秒。8 次局部调用的实际采样元数据均标记 `sampling_verified=true`，但采样可追溯不等于语义正确。
- 项目测试：本地 `python -m pytest -q tests` 为 1065 passed、1 skipped；服务器隔离工作树的 v3/v4 相关测试为 16 passed。直接在仓库根目录无范围收集会误入此前引入的第三方 `ai-video-agent/test_resolve.py`，其导入时退出；故项目测试以 `tests/` 为准。

## 当前门控与盲审

- `reading_validation.json` 两次均没有未知 ID 或越界时间等结构问题。这不表示叙事内容正确。
- `editing_validation.json` 报 `edge_coverage_invalid`：16 条期望相邻边都存在，但输出把其中一条重复了一次，形成 17 行。没有修改原始输出。
- `audit_validation.json` 报 `audit_coverage_invalid`：审核未覆盖全部待审节点。模型给已有检查项的结论不能代替遗漏项，也不能证明视觉事实。
- 完成状态仍是 `candidate_human_review_pending`；`story_ready=false`、`editing_ready=false`、`production_release_allowed=false`、`human_truth_confirmed=false`、`brief_candidate_generated=false`。结果文件的 SHA256 为 `0f5968d125f9f106de3ca0118a0f12d7849528b838cd1be97976902d959e3eca`。本地复制与服务器一致。
- 为保留盲审，先只给评审者 `human_review/first_watch_form.json` 与原片；必须先原速有声观看并记录独立观察，再给 `human_review/blind_variants.json`。顺序映射另存 `blind_key_private.json`，不应在评审前查看。原始请求、响应、采样帧和全部候选都保留在完成目录。

## 贴文建议的采用范围

本轮只采用“剪辑测量与功能解释分开”和“相邻镜头新增信息”这两项与当前 bad case 直接相关的原则。节拍网格、FilmOps、source-reuse graph、速度曲线、look track 和 Editing DNA 均未接入；现有音频能量 onset 仍标为候选，不声称已证明卡点。未训练或下载新权重，也未复制第三方代码。

本报告刻意不比较盲审版本的内容或揭示版本顺序。人工评审完成后再写语义对照与是否进入下一步的结论。
