# perception — 视频感知工具层（Phase 2，已实现）

所有工具 CLI 化（`python -m src.perception.<tool>` --aweme-id 必填），输出统一落
`data/perception/<aweme_id>/<tool>/result.json`（utf-8 JSON 信封，存在即跳过，--force 重跑），
每次真实执行追加一行到 `data/perception/metrics.jsonl`（error 也记）。
stdout 最后一行为单行 JSON 状态（未来 agent 直接解析）。

## 工具卡片（给 coding agent 的说明书）

| 工具 | CLI | 输入 | 输出（output 字段） | 成本量级（实测） |
|---|---|---|---|---|
| inspect_video | `python -m src.perception.inspect_video --aweme-id <id>` | video.mp4 | duration_s/width/height/fps/音视频编码/码率/旋转 | CPU，<1s |
| extract_frames | `... extract_frames --aweme-id <id> [--fps 1] [--max-frames 64]` | video.mp4 | frames[]（index/t_s/file）+ 实际 fps（超出上限自动降） | CPU，秒级 |
| detect_shots | `... detect_shots --aweme-id <id> [--threshold 0.3]` | video.mp4 | shots[]（start/end/mid/duration）+ boundaries_s | CPU，秒级 |
| ocr_frames | `... ocr_frames --aweme-id <id> [--stride 1]` | **依赖 extract_frames 产物** | frames_ocr[]（t_s/lines[text,conf,bbox]）+ text_events[]（字幕常驻合并）+ full_text | CPU ~0.1s/帧 |
| transcribe_audio | `... transcribe_audio --aweme-id <id> [--language auto]` | video.mp4 | full_text（带标点）/segments[ms]/audio_events（BGM/APPLAUSE/LAUGHTER/SPEECH）/emotions | GPU 轻占，RTF≈0.05-0.13 |
| omni_watch_full | `... omni_watch_full --aweme-id <id> [--question ...] [--gpus 0,1]` | video.mp4 | 六小节结构化分析（内容概述/叙事时间线/语音归纳/BGM音效/镜头画面/**模板要素猜测**）+ answer.md + usage（tokens/vram） | **高成本**：GPU 2×48G，~150-400s/条 |
| omni_watch_clip | `... omni_watch_clip --aweme-id <id> --start S --end E --question "..."` | video.mp4 的 [S,E) 段 | clip.mp4（重编码留存）+ answer_text + usage | 中成本：按片段时长 |

## 批处理入口

```bash
python -m src.perception.run_all          # 轻中工具全量（模型各加载一次；幂等续跑）
python -m src.perception.run_baseline     # whole-video Omni baseline（模型加载一次，按时长升序）
python -m src.perception.cost_report      # 聚合 metrics.jsonl → 成本报表（Phase 3 对比底表）
```

## 运行环境（服务器 omni_src，wangqihao@10.1.4.86）

- Qwen3-Omni thinker-only（`Qwen3OmniMoeThinkerForConditionalGeneration`，纯文本输出省 ~10GB），
  bf16 双卡 device_map=auto，实测峰值 [30.4, 30.8]GB（含 92s 长视频余量充足）
- **输入路径走 qwen-omni-utils 三步**（transformers 5.8.0 原生 apply_chat_template 在
  use_audio_in_video=True 时有 audio 占位符 StopIteration bug，2026-09-06 实测）；
  inputs 必须 `.to(device).to(dtype)`（缺 dtype 会炸 conv）
- 贪心解码有复读倾向 → repetition_penalty=1.05 + 段落级截断双保险
- SenseVoiceSmall 首次自动从 ModelScope 下载 ~1GB；whisper 对照实测中文准确率 SenseVoice 更优
- 模型加载：冷缓存 ~220s，热缓存 ~7s

## 已知限制（v1）

- transcribe 无时间戳分段（此版本 funasr 的 SenseVoice+VAD 不返回 sentence_info）；
  Phase 3 如需字幕级时间轴可换 Paraformer（原生时间戳）
- detect_shots 固定阈值（渐变转场检测不到，抖音硬切为主影响小）
- omni 输入 fps 由 qwen-omni-utils 内部决定（config fps 仅作用于原生路径）
