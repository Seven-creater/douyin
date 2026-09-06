# douyin — 自动追热点短视频 Agent

全自动流水线：抖音热点发现 → 下载 → 分析（ASR/BGM/镜头/节奏/文案）→ 抽象热门模板 → 换 IP 素材生成（MiniMax-H3）→ FFmpeg 剪辑 → 批量生成 → 发布。

立意：**Autonomous Video Agent**（自动捕获热点、解析模板、批量生产），不是比单条视频画质。

## 工作规则（必须遵守）

1. **代码只在本地写/改**（本仓库），push 到 `https://github.com/Seven-creater/douyin.git`
2. 服务器 `wangqihao@10.1.4.86` 的 `/data02/usr/wangqihao/Demo/research` **只 pull + 运行，禁止改代码**
3. 服务器直连 GitHub 不通，拉取走 gh-proxy 加速：
   ```bash
   cd /data02/usr/wangqihao/Demo/research && git pull
   ```
4. 运行环境：conda env `h3`（torch 2.9.1+cu128 / diffusers 0.40.0 / transformers 5.14.1）

## 服务器资源

- 视频生成：MiniMax-H3 t2va 常驻服务，`/data02/usr/wangqihao/Demo/test/minimax_h3/`，
  `./serve.sh start` → `0.0.0.0:8300`，`POST /generate`（详见该目录 README）
- 单条 124帧@960×544 ≈ 11.5 min，双卡 int8；8×A6000 最多可并行 4 个实例
- 部署踩坑史见 `minimax_linux.md`
