# template — 热点模板抽取层（Phase 3，已实现）

输入 = Phase 2 感知产物（omni_full 六小节 / ASR / OCR / 镜头 / 节拍 / 元数据），
输出 = 结构化 Trend Template JSON（spec 第十二节），供 Phase 4 MiniMax 批量翻拍。

合成模型 = Qwen3-Omni thinker **文本模式**（`OmniRunner.ask()`，processor.apply_chat_template
纯文本路径；实测 input≈1.5k token、输出 2048、~157s/条）。

## 工具卡片

| CLI | 作用 | 产物 |
|---|---|---|
| `python -m src.template.extract_template --aweme-id <id> [--force] [--dry-run]` | 单条模板抽取 | `data/perception/<id>/template/result.json`（+失败时 raw_answer.txt） |
| `python -m src.template.run_templates [--limit N] [--ids a,b] [--gpus 0,1]` | 批跑（模型加载一次，按时长升序，单条失败不杀批） | 每条 result.json + `data/processed/templates_run_summary_<ts>.json` |
| `python -m src.template.aggregate_templates` | 跨视频聚合 | `data/processed/template_aggregate_<ts>.json` |
| `python -m src.perception.detect_beats --aweme-id <id>` | （上游）librosa 节拍点 | `data/perception/<id>/beats/result.json` |

## Template JSON 结构

```json
{"trend_summary": "...", "core_meme": "...",
 "timeline": [{"start": 0.0, "end": 3.2, "role": "setup|buildup|twist|climax|ending|other",
               "visual": "...", "speech": "...|null", "text": "...|null"}],
 "audio": {"bgm": "...|null", "beat_points": [0.5, 1.2], "speech": []},
 "fixed_elements": ["翻拍必须保留：..."], "replaceable_elements": ["可替换：..."],
 "generation_plan": ["给生成模型的分段指令"]}
```

## 防编造三层防线

1. **Prompt 硬规则**：timeline 时刻必须取自镜头边界；beat_points 只能从给定列表选；
   无依据内容标 uncertain/null；text 必须来自 OCR 事件
2. **stdlib 校验器**（`schema.py`）：缺键/时间越界/区间重叠/role 非法/beat 自造数字 → 硬拒；
   时刻偏离镜头网格 >0.5s → 警告
3. **失败链路**：json 解析失败 → repair 重试 1 次（附错误清单）→ raw_decode 逐键降级（partial）
   → 仍失败不写产物（幂等重跑即重试），raw 存档

## 已知限制（v1）

- 短输入下模型有复读倾向（ask 冒烟实测）——已加 repetition_penalty + JSON 键级截断去重；
  模板任务输入 ~1.5k token，批跑中 json 模式占比见 aggregate 的 parse_modes
- timeline 段数偏保守（模型倾向合并）——如需更细拆段可升 prompt v2（强制 ≥3 段）
- 适合度为透明加权启发式（分量见 aggregate 报告），供人工筛选非 AI 评分
