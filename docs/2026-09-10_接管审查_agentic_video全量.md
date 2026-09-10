# 2026-09-10 接管审查：agentic_video 25 提交全量（进度核实 + 代码审查）

## 背景

另一会话（Codex）在我方 8603029 之后连做 25 个提交，落地 M0-M5 全骨架并新增 narrative 线。
用户指令：先搞清楚它做到哪（鬼灭全片叙事精标 4 卡进行中），顺便审核代码和文件。
本轮全部只读（不碰服务器进程、不改代码），审查用三路并行只读 agent + 人工帧级验证。

## 进度核实（服务器实测）

- **e2e 已跑通一轮**：`data/agentic_runs/7668161826782078931/full_v2/` 全套 M5 产物
  （Recipe v2 22 ops / 检索 0 缺口 / rendered.mp4 / critic_round_1+patch / report.md / manifest）。
- **多操作窗口的突破坐实**：Recipe 里 `text_layer_animation`（NANCHANG 文字层 ±1.2-3.5s）被识别——
  正是 D2-D4 基准里漏检的图层操作，v2 电池（operations[]）起效。
- **能力声明诚实**：2 个 montage_burst 显式 unsupported，报告单列"未执行操作"，无静默降级。
- **critic_round_1 的 0.0 分是幻觉**：它指控成片"全是城市/石狮/猫/NANCHANG 文字"，
  帧级抽验 4 帧全部为真鬼灭作画（炭治郎水之呼吸/花街战斗），无文字层无实拍。
  patches 恰好为空，未造成伤害。→ critic 裁决必须帧级复核后才能驱动 patch（与
  beat_cut critic letterbox 误报同病，已两次实证）。
- **当前正在跑**：鬼灭叙事精标 `--profile narrative --skip-captions --skip-embeddings`
  （PID 1360886，4 卡，CPU 142%，annotation 文件持续更新，已标 13 镜头/36 窗口）。
  高动作库（guimie__high_action）已建；检索槽曾全部命中"鬼舞辻无惨"——caption 存在 IP 偏置
  （后续提交 remove IP bias 正在修）。

## 审查发现（三路 agent 36 条 → 归并后按严重度）

### High（直接产生错误产物或崩溃）

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| H1 | narrative_index.py:261 | 模型对白 start_s/end_s **切片坐标未换算回电影时间轴**，且覆盖 ASR 真值对白 | 窗口在 3700s 时对白 12.5s → 槽位从电影 12.5s（片头）切素材，字幕全错位 |
| H2 | renderer.py:119,128 | drawtext 转义与外层单引号包装不兼容（ffmpeg 6.1.1 实测复现）；`%` 同源 | 文本含撇号 → final 编码整体失败，全部 segment 工作作废 |
| H3 | renderer.py:206 | rendered.mp4 缓存只看路径存在，recipe_hash/theme/canvas 全不参与 | 改主题重跑不加 --force → 返回旧主题成片，exit 0 |
| H4 | renderer.py:86-98 | speed_ramp/color/zoom 等特效作用于**整个 slot**，op interval 不参与 `enable=between` | 区间跨 slot 边界 → 两个 15s slot 整段变速+tpad 冻结帧，manifest 仍报 executed |
| H5 | narrative_index.py:339 | 续跑唯一状态 narrative_annotations.json **非原子写**（无 tmp+replace）；attach_transcript 同病 | 正在跑的 4 卡任务若 kill 在写入半途 → 文件截断 → 已完成标注全丢（当前任务的真实暴露面） |
| H6 | agent.py:249-260 | 初始窗越窗弃用时刻的操作被**静默钉到 t=0** 并保留置信度 | 35s 处的操作以 interval=[0,0.2] status=supported 进 Recipe——击穿证据锚定反编造主线 |

### Med（错误结果/资源浪费，场景明确）

- M1 narrative_index.py:294-312：标注状态与索引产物零版本绑定（不校验 video_sha256/窗口集合）；
  annotation_clips 按窗口号复用旧片段 → 换片源/改阈值后**静默错位**。
- M2 pipeline.py:251-271：critic JSON 解析失败记 score 0.0 并进入停止阈值 → "评审器坏了"被当
  "视频没变好"提前止损，丢弃已算好的 patch（与 critic 幻觉叠加，双通道不可信）。
- M3 manifest.py:29-42：config_sha256/input 哈希**只写不读**，"内容寻址续跑"未实现；
  retrieval_results 存在即复用（pipeline.py:182），主题变更不失效。
- M4 cli.py:255+omni_runner.py:36：**"只允许两卡或四卡"约束不存在于任何代码**（仅 config 注释）；
  --gpus 任意串直通 CUDA_VISIBLE_DEVICES。用户被告知"已修正"的那条约束在仓库里找不到
  （可能在会话层的 wrapper，需向用户澄清）。
- M5 renderer.py:64：grounded_sam 子进程未传 env → 继承 Omni 的 CUDA_VISIBLE_DEVICES，
  与正在跑的精标任务争显存 → OOM 风险。
- M6 recipe.py:125-135：证据锚集过密（64 候选+窗口自报时刻全入锚）→ ±0.25s 反编造校验在
  密集视频上近乎虚设（30s 64 锚≈每 0.47s 一个，伪造 t_s >90% 概率通过）。
- M7 decompose/agent：`{"operations":[]}`（合法无操作）与解析失败不可区分 → 控制窗必然
  重探，浪费 3/16 精探预算；agent 精化逐任务 result 只写不读（无断点续跑）。
- M8 beat_cut.py:94：素材时长过滤 `min(dur, min_len_s)` 应为 `>= dur` → 长切段读穿镜头边界
  混入未描述画面（旧库路线，低优先）。
- M9 renderer.py:294+：concat/final 全片重编码用默认 300s 超时（比单 segment 更紧）→ 长片超时全废。

### Low（摘要）

recipe_v2 校验器对脏 duration_s 自崩 / migrate 可产非法 v2；signals radial 仍是阻尼余弦
（zoom 召回低 10-15%）+ span_s 单位是帧不是秒；本地无 cv2 时 signals 光流全零产生假
beat_freeze、probes 直接 ImportError（本地 4 个挂测试的根因）；实体注册表 8000 字符硬截断；
narrative 精化窗近零时长+跨轮无去重；缓存 JSON 读无容错+写非原子；discovery 签名 URL
只有天粒度新鲜度；run/render 复用 decompose 的 manifest 溯源失真；--story-plan 静默丢弃。

## 处置建议（按序）

1. **正在跑的 4 卡任务**：不动。结束后立即备份 narrative_annotations.json（H5 暴露面）。
2. H1/H6 先修（错产物直接进入素材层和 Recipe 主线）；H2/H3/H4 其次（渲染正确性+缓存失明）；
   H5/M1 补原子写+版本绑定。
3. M2+critic 幻觉：critic 裁决接帧级复核闸门（抽 3-5 帧独立 VLM 盲验）再允许 patch 生效。
4. M4 向用户澄清：两卡/四卡约束在仓库代码里不存在。
5. 修完 H 档后重跑 full_v2 对比（rendered.mp4 与 critic 流程）。

## 学到什么

1. 两个会话并行开发同一仓库时，"用户被告知已修正"的约束要向仓库找证据（M4 实例）；
2. critic 类组件的幻觉有系统性（两次实证），一律要客观复核兜底；
3. 时间坐标换算是本项目事故密度最高的面（H1/H4/M8 三个独立实例），任何"窗口/切片→
   全片"的数据回流都要显式 offset 校验和测试。

## 修复回填（2026-09-10 同日，c7bdc95..b61c3c9）

H1/H2/H3/H4/H5/H6 + M4 全部修复并验证：

| 项 | 修复 | 验证 |
|---|---|---|
| H1 | parse_window_annotations 加 time_offset_s：模型对白 +窗口偏移归一电影轴并夹窗口范围；build_index merge_dialogue——ASR 时间戳为准、模型补翻译（≥50% 重叠才采）、空数组不再覆盖真值；存量数据 coord_system 自动迁移（+窗口起点，立即落盘） | 单测 3 条（偏移/夹界/迁移，3700s 窗口场景）；**正在跑的任务的 13+ 旧镜头由迁移兜底** |
| H2 | renderer 弃用 escape_drawtext，改 drawtext_text_value（单引号包裹+' 关-转-开 注入+expansion=none） | 本地 ffmpeg 实测撇号/百分号/冒号/逗号全过 |
| H3 | render_cache_key（recipe/theme/槽位/选材/画布/叙事 sha256），rendered.mp4 存在但键失配 → 重渲覆盖 | 单测：theme/画布/模式四键互异 |
| H4 | segment_filter 按 op∩slot 求交：eq 加 enable=between 局部窗口；setpts/scale/tpad 无 timeline 支持（ffmpeg 实测报错证实）→ 部分覆盖跳过并记 partial_interval_skipped 进 render_manifest | 单测 2 条（整段/部分覆盖两态） |
| H5 | narrative_annotations 与 attach_transcript 改 tmp+os.replace 原子写；损坏文件明确报错（提示备份恢复而非静默重标） | 单测（两写无 .tmp 残留） |
| H6 | 初始窗行补窗口边界；越窗弃时刻操作：区间=窗口边界、status=uncertain、time_source=window_bounds，不再钉 t=0 伪装 supported | 单测（无 task 行的 None 时刻场景） |
| M4 | _validated_gpu_spec：只允许 2/4 张不重复数字卡号，尾逗号/空白容忍归一，其余 SystemExit | 单测 8 用例 |

回归：**311 passed / 0 failed**（含之前 4 个本地 cv2 环境挂——装 opencv-python-headless 后转绿）。
服务器已 ff-only 拉到 b61c3c9；正在跑的标注进程不受影响（模块已加载），其存量旧坐标数据由
coord_system 迁移在下次加载时自动归一。残留待办：M1/M2/M3 等 9 条中危未修（见上文处置建议）。
