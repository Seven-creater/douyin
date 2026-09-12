# 交接文档（给 Codex，2026-09-12 深夜）

你接手的是「抖音自动追热点视频 Agent」项目。本文告诉你**先读什么、服务器怎么用、
正在跑什么**。读本文 ≈ 15 分钟，可省你半天摸索。

---

## 0. 一句话现状

管线已迭代到 V4（身份诚实化+对白锚定+交付门控，407 测全绿）。**此刻服务器上
72 窗 pack-on 全量重标夜链正在跑**（4 分片八卡并行 → 合并 → 回填 → 索引 →
V4 成片），预计今晚 23:35-00:00 出片 `data/agentic_runs/lxh_p4_V4_C2/rendered.mp4`。
你的第一件事大概率是**验收这条链**（见 §6）。

## 1. 重点读什么（按序，读完就有全貌）

1. **docs/ 下按日期命名的问题实录**（倒序读最近 5 篇，每篇 5 分钟）：
   - `2026-09-12_3窗门控五轮实录与72窗放行.md` ← 最新，今晚全量重标的来龙去脉
   - `2026-09-12_V4_身份诚实化与对白锚定取材.md` + `2026-09-12_V4_终版门控混音与交付判读.md`（V4 设计决定）
   - `2026-09-12_P1.5_Film_Knowledge_Bootstrap设计与冒烟实录.md`（知识包/两层 Identity）
   - `2026-09-12_lxh_p4_C2_badcase_主体断链8层根因实录.md`（V3 的起点 badcase）
2. **config/default.yaml** —— 所有配置的单一来源（素材源/字幕带裁切/验证开关/
   bootstrap/bgm 混音音量），注释即文档。
3. **核心代码地图**（`src/agentic_video/`，按数据流）：
   ```
   discovery → narrative_agent（参考片→叙事程序，弧段带 form_function enum）
   → story_planner（契约选材：身份键+对白锚定+coarse 禁盲切）★最核心
   → verify_slots（槽级验证+det_check_v2+localize_coarse_slots+盲看）
   → renderer（1920×1080 横屏；audio 三态 mix/bgm；字幕带裁切）
   → copywriter（文案+grounding）→ pipeline（总编排：审谁交谁+overall 门控）
   ```
   配套：`src/library/`（build_index/entity_registry/film_bootstrap/caption_shots）。
4. **tests/**（407 个，`python -m pytest tests/ -q` 必须全绿才能提交）——测试名
   就是行为规格史，很多测试 docstring 记着 badcase 出处。

## 2. 硬规则（违反会出事故，均来自实际踩坑）

1. **本地 Windows（本仓库）是唯一代码权威**。服务器只 `git pull` 运行，禁止在
   服务器改代码。GitHub（origin）时好时坏；备用通道：本地 `git push server
   main:refs/heads/from_local` + 服务器 `git merge --ff-only from_local`。
2. **git 提交只用用户身份，不加任何 AI 署名/Co-Authored-By**。
3. **每个非平凡修复/取舍必须写 docs/ 问题实录**（模板见 docs/_模板.md）。
4. 服务器长任务一律 `setsid nohup … < /dev/null &`（ssh 断了不影响）；ssh 长连接
   会断（~1h 实锤），不要把多阶段链串在一个前台 ssh 里。
5. `pgrep/pkill -f` 的模式串会匹配你自己的 ssh wrapper —— 用完整路径锚定
   （`^/data02/...python /tmp/xxx.py`）或按精确 PID 杀。
6. 服务器跑码前过 `ast.parse(feature_version=(3,10))` + 真实 import 冒烟。

## 3. 服务器与环境

- **地址**：`ssh wangqihao@10.1.4.86`（lthpc，8×A6000 49GB **共享**，先
  `nvidia-smi` 看占用再占卡）。仓库：`/data02/usr/wangqihao/Demo/research`。
- **conda 环境**（坑最多的一条）：
  - `omni_src`（`/data02/usr/wangqihao/miniconda3/envs/omni_src/bin/python`，
    3.10.20）——**GPU 全流程唯一可用**（有 qwen_omni_utils/torch/transformers）。
  - `h3`（3.12）——只用于纯 json/ffmpeg 轻脚本；**跑 Omni 会在
    qwen_omni_utils 导入处崩**。
- **GPU 纪律**：每个 Omni 实例占 2 卡（~35GB×2），必须
  `CUDA_VISIBLE_DEVICES` 钉死空闲卡；8 卡都空时可 4 实例并行（分片基建已入库：
  `run_narrative_annotations(output_path=)` + `merge_annotation_shards`）。

## 4. 模型

| 模型 | 用途 | 位置/备注 |
|---|---|---|
| Qwen3-Omni-30B-A3B（thinker-only） | 看片/标注/验证/critic/盲看 | `/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct`；裸 transformers 部署（无 vllm serving），并发=多进程多实例；watch 长视频会 OOM → `prepare_watch_copy` 720p 转码后喂 |
| Qwen2-VL-7B | 镜头 caption | captions 走 `src/library/caption_shots.py`，prompt v2 |
| E5 | 检索 embedding | `src/library/build_index.py`；index 在 `data/library/index/` |

## 5. 数据

| 数据 | 路径（服务器） |
|---|---|
| 素材电影 | `data/raw/luoxiaohei/`（罗小黑1 4K·16:9 镶 scope；罗小黑2 原生 2.376:1） |
| 每源标注 | `data/library/shots/luoxiaohei{1,2}__narrative/`：`result.json`（窗口/镜头）、`captions.json`、`narrative_annotations.json`（**正被夜链重写**，旧版备份 `.pre_v3.json`）、`knowledge_pack.json`、分片 `shard_{a,b}.json` |
| 全局索引 | `data/library/index/`（shots.jsonl + cap_emb.npy，一次 build 全量扫两片） |
| 人物真值 | `config/entity_registry.json`（四键：aliases 身份名 / role_labels 头衔 / appearance_aliases 外观——**后两者绝不参与身份判等**，只做绑定引导）+ 服务器回填的 `data/library/entity_registry.auto.json` |
| 参考片 | `data/videos/7682719919410072847/video.mp4`（7682 型情感叙事模板的来源） |
| 运行产物 | `data/agentic_runs/`（V4_C=下午门控验收跑；**V4_C2=今夜目标成片**） |

## 6. 正在跑的夜链 + 怎么验收

- 编排：`/tmp/night_72c2.py`（flock 单实例、收养在跑分片、rc 门禁、标记
  `data/agentic_runs/72c_markers/`）；看门狗 `/tmp/watchdog_72c.sh` 每 5min
  重拉；日志 `data/agentic_runs/night_72c2.log`、`job_*.log`、`night_tail.log`。
- 脚本本体在本地 `data/preview/decomposer_d1/diag/`（night_72c2/watchdog_72c/
  night_tail2/job_shard）。
- **验收顺序**：
  1. `ls 72c_markers/` 有 `video_done` → 链完成；
  2. `night_72c2.log` 尾部 `NIGHT_72C2_DONE`；
  3. 两片 `narrative_annotations.json` completed=36/36，bindings 抽查（supported
     的 canonical 应是 char:xiaohei/wuxian/fengxi 真值系）；
  4. `data/library/entity_registry.auto.json` 生成且 source_entities 窗口作用域
     （`source/w{N}/{id}`）；
  5. `lxh_p4_V4_C2/final_review.json`：`overall.passed` 与 reasons（若
     false，读 reasons 定位：det/plan/copy/blind 哪个因子）；
  6. 拉片：`scp` rendered.mp4 回本地看（1920×1080、听对白是否 mix 终混）。
- **若链挂在中间**：重跑 `/tmp/night_72c2.py` 即续（completed 键扣除断点，
  收尾各步幂等）；看门狗理论上已自动做了。

## 7. 之后的路线图（与用户已对齐）

1. 72 窗重标质量验收（上面的 §6）→ 若成片仍 blocked，按 reasons 定位；
2. 三臂实验补齐：Searchable-Unknown（等用户给冷门片）+ Private（3 个自编角色
   1-2 分钟自制视频，GT 全知）；
3. 消融 material_mode 余下三档（video_only 已出）；
4. V4.1 记档未做：copy→verified evidence 消费、loudnorm/ducking、
   cutaway_function 由 planner 智能选择；
5. 待用户拍板过的小项：context 槽 optional→required、文案烧录位置与电影字幕
   的视觉区分。

## 8. 快速命令备忘

```bash
# 本地测试（407 绿才提交）
python -m pytest tests/ -q

# 服务器：全流程跑一次（omni_src + 钉卡是铁律）
ssh wangqihao@10.1.4.86
cd /data02/usr/wangqihao/Demo/research && git merge --ff-only from_local
CUDA_VISIBLE_DEVICES=0,1 setsid nohup /data02/.../envs/omni_src/bin/python \
  -m src.agentic_video.cli run --reference data/videos/7682719919410072847/video.mp4 \
  --theme 罗小黑_妖灵与人类的羁绊成长 --library luoxiaohei1,luoxiaohei2 \
  --output data/agentic_runs/<名字> --target-duration 22 > <名字>.log 2>&1 < /dev/null &

# 3 窗诊断标注（指定窗）
... -m src.agentic_video.cli index --source luoxiaohei1 --windows 14,19,24 \
  --skip-captions --skip-embeddings   # pack 开关在 config library.film_bootstrap.enabled
```

—— 交接完毕。拿不准的设计决定，先查 docs/ 对应实录（每个决定都有 why）。
