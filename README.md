# douyin — 自动追热点短视频 Agent

全自动流水线：抖音热点发现 → 下载 → 分析（ASR/BGM/镜头/节奏/文案）→ 抽象热门模板 → 换 IP 素材生成（MiniMax-H3）→ FFmpeg 剪辑 → 批量生成 → 发布。

立意：**Autonomous Video Agent**（自动捕获热点、解析模板、批量生产），不是比单条视频画质。

## 工作规则（必须遵守）

1. **代码只在本地写/改**（本仓库），push 到 `https://github.com/Seven-creater/douyin.git`
2. 服务器 `wangqihao@10.1.4.86` 的 `/data02/usr/wangqihao/Demo/research` **只 pull + 运行，禁止改代码**
3. 服务器直连 GitHub 不通，拉取走 gh-proxy 加速（origin 已配好，直接 `git pull`）
4. 运行环境：服务器 conda env `h3`；本地 anaconda base（Python 3.13）

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
python -m pytest -q            # 50 个单测（全 mock，不打真实 API）
python -m tests.smoke_real_api # 真实 API 冒烟（手动，花 credits）
```

## 排障

- 下载 403 → 签名 URL 过期：`--refetch` 重拉榜单刷新 URL 再跑（失败记录保留在 manifest，重跑只补失败）
- 下载 too_small / not_mp4 → CDN 返回错误页，同上处理
- 回传失败 → 不致命，检查 ssh 免密；单独重推：`scp -r data/videos/<id> wangqihao@10.1.4.86:/data02/usr/wangqihao/Demo/research/data/videos/`
- `.env` 缺失 → `cp .env.example .env` 并填 key

# Phase 2+（规划中）

- perception（视频理解）：Qwen3-Omni 感知工具层，见 `src/perception/README.md`
- generation（素材生成）：MiniMax-H3，见 `src/generation/README.md`
- 服务器资源详情见 `minimax_linux.md`
