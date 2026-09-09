# douyin — Agentic Video Edit Program Induction

目标流水线：抖音热门剪辑 → Agent 主动收集多模态证据 → 多轨 Recipe v2 →
一句话主题驱动的真实素材检索 → 确定性渲染 → Critic 结构化修订。

项目核心是 **Video Edit Program Induction**：从渲染后的成片反推出有证据、可校验、
可重新执行的剪辑程序。成片质量是结果，剪辑理解的可测量性是第一指标。

## 工作规则（必须遵守）

1. **代码只在本地写/改**（本仓库），push 到 `https://github.com/Seven-creater/douyin.git`
2. 服务器 `wangqihao@10.1.4.86` 的 `/data02/usr/wangqihao/Demo/research` **只 pull + 运行，禁止改代码**
3. 服务器直连 GitHub 不通，拉取走 gh-proxy 加速（origin 已配好，直接 `git pull`）
4. Agentic 主链运行环境：服务器 conda env `omni_src`；GroundingDINO+SAM2 由
   `grounded_sam2` 环境子进程执行；旧 MiniMax 管线仍使用 `h3`

# Phase 1：热点发现与下载（已完成）

## 架构（2026-09-06 网络实测决定）

| 环节 | 运行端 | 原因 |
|---|---|---|
| Wellbyte 五榜单抓取/解析/排序 | 两端皆可（服务器 h3 / 本地） | api.wellbyte.net 两端可达 |
| 视频下载（CDN mp4 直链） | **仅本地** | 服务器防火墙屏蔽字节系全域 CDN（douyinvod/douyinstatic/douyinpic 实测秒断）|
| 视频回传 | 本地 → scp → 服务器 | Phase 2 的 Qwen3-Omni 感知在服务器 GPU |

关键事实：Wellbyte 榜单的 `item_url` 字段直接就是 CDN mp4 直链（带签名、有时效），
本地 requests + 浏览器 UA + Referer 即可下载，**无需 Cookie、无需 TikTokDownloader**。
图集（media_type=2）的 `item_url` 是其 BGM mp3（留存于数据，可供以后建 BGM 素材库）。

## 一条命令

```bash
# 本地（全流程：抓榜→解析→去重→Top20→下载→回传服务器）
python -m src.pipeline.collect_trends

# 常用参数
python -m src.pipeline.collect_trends --no-download        # 只做榜单阶段（服务器模式）
python -m src.pipeline.collect_trends --refetch            # 当日 raw 已存在也重拉（花 10 credits）
python -m src.pipeline.collect_trends --max-downloads 3    # 阶梯验证用
python -m src.pipeline.collect_trends --top-n 30           # 更多候选
```

## 安装

```bash
# 本地（anaconda base 即可，依赖：requests/python-dotenv/PyYAML/rich/pytest）
pip install -r requirements.txt
cp .env.example .env   # 填 WELLBYTE_API_KEY（.env 已 gitignore）

# 服务器
cd /data02/usr/wangqihao/Demo/research && git pull
# h3 env 已含全部依赖；.env 需手工创建（scp 本地 .env 或手写，不进 git）
```

## 数据产物（全部 gitignored）

```
data/raw/wellbyte/YYYY-MM-DD_<sub_type>.json   # Wellbyte 原始响应（字节原样，当日复用省 credits）
data/processed/trending_videos.jsonl           # 去重合并后全量（含 raw、各榜名次、final_rank）
data/processed/trending_videos.csv             # 人工检查用（utf-8-sig，Excel 直开）
data/processed/run_summary_<ts>.json           # 每次运行的完整统计
data/videos/manifest.json                      # 下载台账（resume/跳过唯一事实源，错误历史追加）
data/videos/<aweme_id>/video.mp4               # 视频文件（H.264+AAC）
data/videos/<aweme_id>/metadata.json           # 榜单信息+原始数据（Phase 2 perception 直接可读）
logs/collect_trends_<ts>.log                   # 运行日志（密钥自动脱敏）
```

## 热点排序规则（透明，无 AI 评分）

按 sort_key 降序：`(上榜榜单数, 总榜名次分, 低粉爆款名次分, 高点赞名次分, likes, aweme_id)`
名次分 = 21 − rank（page_size=20）；CSV 每列可审计。

## 计费

2 credits/榜单/次（成功才扣；1 credit=$0.001）。全 5 榜一轮=10 credits。
当日 raw 已存在自动跳过（0 credits）。注册赠 500。

## 测试

```bash
python -m pytest -q            # 258 个单测（默认不打真实 API/大模型）
python -m tests.smoke_real_api # 真实 API 冒烟（手动，花 credits）
```

## 排障

- 下载 403 → 签名 URL 过期：`--refetch` 重拉榜单刷新 URL 再跑（失败记录保留在 manifest，重跑只补失败）
- 下载 too_small / not_mp4 → CDN 返回错误页，同上处理
- 回传失败 → 不致命，检查 ssh 免密；单独重推：`scp -r data/videos/<id> wangqihao@10.1.4.86:/data02/usr/wangqihao/Demo/research/data/videos/`
- `.env` 缺失 → `cp .env.example .env` 并填 key

# Phase 2：多模态感知（已完成）

Qwen3-Omni-30B thinker + FunASR/RapidOCR/librosa/ffmpeg 工具层，
一条命令 `python -m src.perception.run_all --gpus 0,1,2,3`，产物 `data/perception/<id>/<tool>/result.json`。
详见 `src/perception/README.md`。

# Phase 3：模板抽取（已完成）

感知产物 → 结构化 Trend Template JSON（timeline/role/fixed/replaceable），
三层防编造防线，12/13 条 json 一次过。详见 `src/template/README.md`。

# Phase 4：MiniMax 批量翻拍生成（历史分支，已冻结）

```
Template JSON → planner（role 感知单元合并）→ variant（LLM 提议 3 套替换）
→ rewrite（英文 t2va prompt）→ MiniMax-H3 4 实例并行生成 → FFmpeg 装配 → final.mp4
```

```bash
# 服务器一条命令（全流程幂等，断点续跑）
python -m src.generation.run_generation --template-id <aweme_id> --variants 3 --instances 4
```

2026-09-07 实测（纸箱梗模板 7681612687865084623）：1 模板 × 3 变体（抽象文字舞/都市购物/
赛博朋克），2 生成单元 × 3 = 6 片段全部一次成功（124f≈13min、192f≈21min，4 实例并行），
3 个 final.mp4（544×960 竖屏 H.264+AAC+字幕，10.14s）装配完成；重跑全 skip。
详见 `src/generation/README.md`。

# Phase 5：Agentic Narrative Video Editor

统一入口：

```bash
# 生成 144 个程序化真值视频；给出预测目录时同时计算准确率
python -m src.agentic_video.cli benchmark

# 获取有内容的热门参考（最近 7 天，三类各最多 4 条）
python -m src.agentic_video.cli discover \
  --days 7 --per-category 4 --output data/reference_pool/2026-09-09

# 鬼灭长片：低成本扫描 → 对白/动作/情绪/收束配额 → 叙事窗口索引
python -m src.agentic_video.cli index --source guimie --profile narrative

# 只反编译参考视频
python -m src.agentic_video.cli --gpus 0,1 decompose \
  --reference data/videos/<id>/video.mp4 --mode both --output data/agentic_runs/<run>

# 完整链路
python -m src.agentic_video.cli --gpus 0,1 run \
  --reference data/videos/<id>/video.mp4 \
  --theme "鬼灭：守护与牺牲" --library guimie --output data/agentic_runs/<run>
```

固定交付包括 `run_manifest.json`、`reference.narrative.json`、
`reference.recipe.json`、`story_plan.json`、`asset_plan.json`、
`retrieval_results.json`、`rendered.mp4`、双 Critic JSON/补丁和 `report.md`。
Narrative Program 负责人物、事件、因果和情绪；Recipe v2 只负责可执行剪辑操作。
素材不足、人物切换无法解释、对白会被截断或不支持的操作都会显式标记，不能静默拿随机镜头填充。
首版成片固定为 45–75 秒、1920×1080，保留日语原声并叠加有时间依据的中文字幕。

# 服务器资源

详见 `minimax_linux.md`。
