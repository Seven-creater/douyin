# generation — MiniMax 批量翻拍生成层（Phase 4，已实现）

输入 = Phase 3 Template JSON（`data/perception/<id>/template/result.json`），
输出 = `data/generation/<id>/variants/<vid>/final.mp4`（H.264 544×960 竖屏 + AAC + 内嵌字幕）。

分层铁律：MiniMax 只做"按 prompt 生成 ≥5.2s 片段"（t2va，原生同步音频）；
规划/合并/剪辑节奏全部由确定性代码 + FFmpeg 承担。LLM（Qwen3-Omni 文本模式 ask()）
只出场两次：变体提议、prompt 重写。

## 一条命令

```bash
# 服务器（全流程：plan → variants → rewrite → generate → assemble）
python -m src.generation.run_generation --template-id <aweme_id> --variants 3 --instances 4

# 分段执行 / 常用参数
... --stage plan|variants|rewrite|generate|assemble   # 幂等：已有产物自动跳过
... --variant-file variants.json                      # 跳过 LLM 提议，手工指定变体
... --allow-missing                                    # 装配时缺片段用黑场占位（调试）
... --force                                            # 忽略缓存重跑该阶段

# MiniMax 服务单独管理（冷启动 ~20min，可提前拉起）
python -m src.generation.minimax_client start [--instances 4]   # 8 卡 → 4 实例 (8300-8303)
python -m src.generation.minimax_client status
python -m src.generation.minimax_client stop                     # 用完即释（8 卡独占）
```

## 流水线五段

```
template/result.json
 ├─ planner   timeline 段 → role 感知贪心合并成 5.2~10s 生成单元（17n+5 帧数夹 [124,260]）——全变体共享
 ├─ variant   ask() 提议 N 套替换方案（fixed 保留 + replaceable 换血）；失败→预设池兜底
 ├─ rewrite   ask() 把 (单元视觉简报 + 变体替换) 重写为英文 t2va prompt（一次全部单元）；失败→中文直填
 ├─ generate  4 实例并行调度（每实例 1 在途 job，超时只计 running，失败 seed+1 重提 ≤2 次）
 └─ assemble  片段裁回时间线 → concat → drawtext 字幕 → final.mp4 + assembly.json 全程审计
```

## 工具卡片

| CLI | 作用 | 产物 |
|---|---|---|
| `python -m src.generation.planner --template-id <id> [--dry-run] [--force]` | timeline → 生成单元（规则 A 不足吸 / B 同 role 吸 / C 超限拆） | `data/generation/<id>/plan/result.json` |
| `python -m src.generation.rewrite --template-id <id> --variant <vid> [--force]` | 单变体英文 prompt 重写（json 解析失败 repair 1 次 → 中文 fallback） | `variants/<vid>/prompts/result.json` |
| `python -m src.generation.minimax_client {start\|status\|stop}` | 服务生命周期（多实例直跑 serve.py，绕过 serve.sh 单实例设计） | 端口 8300+N 就绪；日志 `logs/mm_instance_<port>.log` |
| `python -m src.generation.assemble --template-id <id> --variant <vid>` | 裁剪/拼接/字幕（时长不足自动收缩窗口） | `variants/<vid>/final.mp4` + `assembly.json` |
| `python -m src.generation.run_generation --template-id <id> [--stage ...]` | 全流程编排（各段幂等，断点续跑） | 3×final.mp4 + `data/processed/generation_run_summary_<ts>.json` |

内部模块：`models.py`（dataclass + fnv1a 确定性 seed）、`prompts.py`（t2va 风格头 +
重写/变体提议 prompt + 复读去重）、`variant.py`（propose/preset/file 三模式 + 模糊校验）。

## 产物目录（gitignored）

```
data/generation/<template_id>/
  plan/result.json            # units[]：span/roles/segments/num_frames/seed_base/visual_brief
  variants.json               # VariantSpec[]（提议结果缓存，重跑走 cached 模式）
  manifest.json               # clips 台账：is_done 只信本地 clips/uXX.mp4 存在
  variants/<vid>/prompts/result.json   # 单元英文 prompt + seed（json|fallback 模式记账）
  variants/<vid>/clips/uXX.mp4         # MiniMax 原始生成片段
  variants/<vid>/work/                 # trim 段 + concat.mp4（中间产物）
  variants/<vid>/final.mp4  assembly.json   # 成品 + 每步 ffmpeg 命令全记
data/generation/metrics.jsonl          # generate_clip ok/error 逐条打点
```

## 成本量级（2026-09-07 实测，124f@544×960）

- 生成：124f（5.2s）≈ 13min/段；192f（8.0s）≈ 20-21min/段；4 实例并行
- LLM：变体提议 ~2-3min、prompt 重写 ~2-3min（每变体，Qwen3-Omni ~14 tok/s）
- 装配：秒级（纯 ffmpeg）
- 幂等：任一段重跑全 skip；单段失败重跑只补缺口（manifest 台账）

## 已知限制（v1）

- 生成分辨率 544×960（MiniMax 32 倍数约束下的近似 9:16），抖音上传建议再 1080×1920 放大
- 段切换处不保证镜头连贯（按单元生成再裁回时间线是分层固有代价；role 感知合并 +
  "先…后…"时序措辞尽量缓解）
- 台词/字幕来自模板 timeline 原文（drawtext 直叠），MiniMax 生成的是环境音+拟声，
  无人声口型同步
- 音频为 MiniMax 原生 AAC，未做 BGM 对齐/节拍卡点（模板 beat_points 已留存待用）
- serve.sh 是单实例设计（共享 serve.pid）——多实例必须直跑 `python serve.py --port N`
  （已在 minimax_client 内处理）
