# perception — Phase 2 占位

视频理解工具层（Phase 1 完成后再实现，不提前写空代码）。

规划中的工具（每个都是 CLI/函数可调用的 perception tool，Agent 按需选择，
不默认整段视频喂全模态模型）：

- `inspect_video`：ffprobe 元数据（时长/分辨率/帧率/音轨）
- `transcribe_audio`：ASR 转写
- `extract_frames`：抽帧
- `detect_shots`：镜头切分
- `ocr_frames`：帧内文字 OCR
- `omni_watch_clip`：Qwen3-Omni 局部片段理解（服务器 `omni_src` env，
  模型 `/data02/pretrained_model/cvr_learn/cvr_model/03_audio_vlm2vec_backbone/qwen3-omni-30b-a3b-instruct`）
- `omni_watch_full`：整段视频理解（高成本 Action，仅当 Agent 判断必要时）

第一版 baseline：whole-video Qwen3-Omni；后续对比 Agent 动态选择工具的
准确率 / token 消耗 / 调用次数 / 时延 / 成本。
