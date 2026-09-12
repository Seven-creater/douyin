# CONTEXT（长期稳定信息，2026-09-13 建）

## 项目目标
全自动抖音追热点视频 Agent：热门参考片 → 叙事结构分析 → 本地电影素材库
检索 → 契约式选材 → 渲染成片（1920×1080 横屏）→ 多层验证（det/blind/
overall 门控）。当前素材库=罗小黑两部剧场版；参考模板=7682 型情感叙事。

## 技术栈与部署形态
- 本地 Windows 仓库 = **唯一代码权威**；服务器 10.1.4.86（/data02/usr/
  wangqihao/Demo/research）只 pull 运行（omni_src 环境跑 GPU；h3 只跑轻脚本）
- GPU：8×A6000 49GB 共享；Qwen3-Omni-30B 每实例 2 卡，并发=多进程多实例
  （分片基建：run_narrative_annotations(output_path=) + merge_annotation_shards）
- 模型：Qwen3-Omni（看片/验证/critic）、Qwen2-VL（caption）、E5（检索）
- 测试：`python -m pytest tests/ -q`（407 个，提交前必须全绿）

## 核心概念（快速词汇表）
- **两层 Identity**：本地 vis_ id + 旁挂 canonical bindings（supported/
  verified 才进硬约束）；外观/头衔别名绝不参与身份判等
- **knowledge_pack**：P1.5 电影知识包（metadata_prior 档）；pack_validity
  拦坏包；grounded 实体=归并到 config/entity_registry.json 手写真值
- **契约选材**：主角锁 canonical → 跨窗同人选材；cutaway=带声明的自由槽
- **localize or reject**：>2×budget 无锚粗窗禁盲切，Omni 定位或槽降级
- **overall 门控**：det ∧ plan_complete ∧ copy 轨道 ∧ blind 三态；失败=
  delivery blocked（诚实失败是特性不是 bug）

## 关键约束（为什么）
- 剪辑成片一律 1920×1080 横屏（用户规定）
- 需求文本零参考事实泄漏（C2 教训：跨库荒谬检索）
- 每 45s 窗独立标注，身份靠 bindings 跨窗（窗口级 id 会假等价）

## 深入阅读（按需）
docs/ 下按日期的问题实录（每个决定带 why）；config/default.yaml 注释；
HANDOFF_codex.md（服务器/数据/命令全景）。
