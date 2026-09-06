# generation — Phase 3/4 占位

素材生成层（Phase 1 完成后再实现，不提前写空代码）。

规划：Agent 产出 generation_plan → 调用 MiniMax-H3（服务器 10.1.4.86 常驻服务
`/data02/usr/wangqihao/Demo/test/minimax_h3`，`./serve.sh start` → `0.0.0.0:8300`，
`POST /generate`）生成视频片段（t2va，原生同步音频）→ FFmpeg 剪辑合成。

MiniMax 只是 generation tool，不负责理解/规划/剪辑——能力必须分层。
约束：单条 124 帧起（17n+5）、分辨率 32 倍数且 ≤1344、单条约 11.5 分钟（124f@960×544）。
