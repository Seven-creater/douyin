# TASKS（2026-09-13）

## IN PROGRESS
- 72 窗 pack-on 夜链收尾：film1 36/36、film2 33/36（3 窗顽固 parse 失败降级）；
  V4 成片 lxh_p4_V4_C2 跑在卡 4,5，video_done 标记待落。消融 full/text_only
  已出（缓存命中），ocr_dedup 在跑。

## TODO（按序）
- 验收夜链：HANDOFF_codex.md §6 六步（markers→completed→bindings 抽查→
  auto registry→overall→拉片听混音）
- 成片若 blocked：按 reasons 因子定位，别急着改 prompt
- 消融四档对比分析（文字主导度：A vs C vs D）
- 三臂实验补齐：Searchable-Unknown（等用户冷门片）+ Private（自编角色素材）
- V4.1 记档：copy→verified evidence、loudnorm/ducking、cutaway 智能选择
- 待用户拍板：context 槽 optional→required；文案烧录位置与电影字幕区分

## BLOCKED
- film2 那 3 窗（两轮 parse 失败，需看 raw 答案找模式再修 prompt）

## DONE（近期）
- V4 全套（身份诚实化/对白锚定/禁盲切/mix 终混/overall 门控）落地+验收
- 3 窗门控五轮（9 bug）→ 72 窗放行
- 分片高并发基建 + 夜链加固（flock/watchdog 有界/防白跑三道保险）
- Codex 协作基建（tools/ask-codex + .ai/ 共享上下文）
