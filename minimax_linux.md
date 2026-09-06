# MiniMax-H3 FL2VA 本地部署实录

> 2026-08-21 ~ 08-22 · 服务器 `10.1.4.86`（lthpc）· 2×RTX A6000 48GB · Ubuntu 20.04 (glibc 2.31) · 驱动 575 (CUDA 12.9)
> 结果：**✅ 全流程跑通**，t2va（文生视频+原生音频）单条 5.2s/960×544 视频 12.6 分钟，双卡 int8，显存 31/35GB

## TL;DR — 复现

```bash
ssh wangqihao@10.1.4.86
cd /data02/usr/wangqihao/Demo/test/minimax_h3

# 生成视频（改 prompt 编辑 gen_test.py 的 PROMPT；改分辨率/帧数改 NUM_FRAMES/HEIGHT/WIDTH）
setsid nohup /data02/usr/wangqihao/miniconda3/envs/h3/bin/python gen_test.py > out.log 2>&1 &
tail -f out.log        # 看 [test] 分段计时

# 完整流水线（含校验+ffprobe，写 PIPELINE_RESULT.txt）
setsid nohup ./auto_pipeline_v2.sh > auto_v2.log 2>&1 &
```

产物：`results/*.mp4`（H.264 + AAC 32kHz 立体声，音频为模型同步生成）

---

## 一、目标与约束

**目标**：MiniMax-H3（33B 全模态视频生成）FL2VA 变体，2 张卡部署，跑通 t2va 端到端。

**三道墙（决定了所有技术选型）**：

| 约束 | 影响 |
|---|---|
| Ubuntu 20.04，glibc **2.31** | sglang ≥0.5.11 只发 `manylinux_2_34` wheel（需 glibc 2.34）→ **sglang 装不上** |
| 驱动 575 = CUDA **12.9** | vllm 0.26+（H3 最低版本）全是 **cu13 构建**（锁 torch≥2.11，而 cu128 的 torch 最高 2.9.1）→ **vllm 跑不了**；升驱动要 root |
| 服务器→ModelScope 限速 **~4MB/s** | 144GB 权重下了 ~19 小时；换 aria2c/96 连接无效（按 IP 限速）；HF 仓库 gated 无镜像 |

另一台前人留下的烂摊子：sglang 环境装的是不含 H3 的 0.5.10、下载进程被 SIGSTOP 挂起、自动化流水线没跑。

**结论：唯一可行的路 = diffusers（纯 Python，不受 glibc/CUDA 构建限制）+ 官方转换脚本 + 双卡 int8。**

## 二、最终架构

```
ModelScope MiniMax/MiniMax-H3 (~144GB, FL2VA/ 原始格式, sglang/vLLM 用)
   │  modelscope CLI ×6 组并行，断点续传，~19h @4MB/s
   ▼
/data02/usr/wangqihao/Demo/checkpoints/MiniMax-H3/FL2VA     ← 原始权重（保留）
   │  官方 convert_minimax_h3_to_diffusers.py（流式，峰值内存 ~5GB，~40min）
   │  + text_encoder/tokenizer/processor 软链（它们本来就是标准 HF 格式）
   ▼
/data02/usr/wangqihao/Demo/checkpoints/MiniMax-H3-diffusers  ← diffusers 布局（72GB+软链）
   │  ModularPipeline workflow="t2va"
   │  GPU1: text_encoder(Qwen3-VL 62GB) int8 ≈ 35GB 驻留
   │  GPU0: transformer(61.7GB) int8 + VAE ≈ 31GB 驻留
   ▼
 124帧 @960×544, 49步去噪 → 视频+音频 latents → VAE 解码 → encode_video 合成 mp4
```

## 三、分步过程

### 1. 权重下载与校验
- `parallel_download.py`：按大小贪心分 6 组，每组一个 modelscope CLI 进程，组内重试，文件级 size 校验
- `kill -CONT` 救活被挂起的进程（断点续传安全）
- **坑**：ModelScope 文件清单含**目录条目**（Size=0），目录 `getsize()=4096≠0` 被判"永久缺失"→ group2 空转 199 次后整个监管退出，`DOWNLOAD_DONE` 没写上
- **修复**：校验时跳过 `os.path.isdir()` 条目（修正脚本见 `parallel_download.py` 的教训，手工校验后补写标记）

### 2. 环境（conda env `h3`）
```
python 3.12 / torch 2.9.1+cu128 / diffusers 0.40.0 / torchao 0.15.0
transformers 5.14.1 / accelerate 1.14.0 / safetensors 0.8.0
```
- diffusers 0.40.0 当天 tuna 还没同步 → 本地下载 wheel scp 上去 `pip install --no-deps`
- torchao 必须 ≥0.15.0（diffusers 0.40 的 `TorchAoConfig` 硬检查；abi3 wheel 与 torch 2.9 兼容）
- transformers 必须 ≥5.10.1（`Qwen3VLProcessor.create_mm_token_type_ids`），避开 5.15（上游已知回归）

### 3. 格式转换（`assemble_h3.sh`）
- 转换脚本来自 diffusers 仓库（PR #14355 随 0.40.0 发布）：重命名 + QKV 拆分 + SwiGLU 半区交换，纯映射无转置
- 只转 `transformer/ vae/ audio_vae/` + 写 scheduler 配置和 `modular_model_index.json`
- **`text_encoder/`（66.7GB）不用转**：FL2VA 里的它就是标准 HF Qwen3VLForConditionalGeneration，软链即可——省了一半下载量
- 已加跳过守卫：`ASSEMBLE_DONE` 存在则不重跑

### 4. 生成（`gen_t2va.py` / `gen_test.py`，官方 2×48GB 卡配方）
- 两个 `ComponentsManager`：conditioner 子管线 → cuda:1，其余 → cuda:0，auto_cpu_offload 兜底
- 双大组件 `TorchAoConfig(Int8WeightOnlyConfig())` 量化加载（不转换的模块清单抄官方文档）
- **坑**：diffusers 0.40 量化加载**禁止** `low_cpu_mem_usage=False`（官方文档示例是给更新版本的）
- t2va 参数：`num_frames=124`（帧数必须 17n+5）、960×544（32 的倍数，比原生 1344×768 快 2.3×）、seed 固定可复现
- 兜底方案（未启用）：单管线 + block_level group offload 流式

### 5. 自动化（`auto_pipeline_v2.sh`）
`DOWNLOAD_DONE` → assemble（有守卫）→ gen → ffprobe 验证 → `PIPELINE_RESULT.txt`
（v3 的 vllm 版和 STAGE1.5 等卡逻辑保留在文件里，但 vllm 路线已废弃）

## 四、性能实测（2026-08-22，页缓存热）

| 阶段 | 耗时 |
|---|---|
| 加载 133GB + int8 量化 | 62s（冷启动 ~20min）|
| prompt 编码（Qwen3-VL 一次）| 23s |
| 去噪 49 步 + VAE 解码 | 669s（12.7s/步）|
| 封装 | ~2s |
| **总计** | **756s ≈ 12.6 min** |

换算：1344×768 ≈ ×2.3；时长每秒 ≈ 每步线性变慢（步数不变）；常驻服务可省加载。
显存峰值：GPU0 31GB / GPU1 35GB（48GB 卡有余量）。

## 五、文件清单（`/data02/usr/wangqihao/Demo/test/minimax_h3/`）

| 文件 | 用途 |
|---|---|
| `gen_test.py` | **单次生成 + 分段计时（日常用这个）** |
| `gen_t2va.py` | 冒烟版（带 group-offload 兜底）|
| `auto_pipeline_v2.sh` | 一键流水线（下载完→转换→生成→验证）|
| `assemble_h3.sh` / `convert_h3.py` | FL2VA→diffusers 转换 |
| `parallel_download.py` / `download_h3.sh` | 权重下载（已完成使命）|
| `PIPELINE_RESULT.txt` / `gen.log` / `test_cats.log` | 结果与日志 |
| `results/smoke_t2va.mp4` / `results/test_cats.mp4` | 产物 |
| `vllm-omni/`、`serve_h3_vllm.sh`、`auto_pipeline_v3.sh` | vllm 路线遗物（cu13 跑不了，仅存档）|
| conda env `h3vllm` | vllm 路线遗物，可删 |

## 六、踩坑速查（现象 → 原因 → 修复）

1. 下载进程 `Tl` 状态不动 → 被 SIGSTOP → `kill -CONT`
2. 下载"永远缺 8 个文件 0.0GB" → 清单目录条目被当缺失 → 校验跳过 isdir
3. `pip 装不上 sglang/vllm 新版` → manylinux_2_34 / cu13 与 glibc 2.31 / 驱动 12.9 冲突 → 用 diffusers
4. `TorchAoConfig requires torchao>=0.15` → 升 torchao（abi3 兼容 torch 2.9）
5. `low_cpu_mem_usage cannot be False when using quantization` → 删掉该参数
6. `'Qwen3VLProcessor' object has no attribute 'create_mm_token_type_ids'` → transformers 5.3→5.14.1
7. 流水线"环境超时"假失败 → pip 限速拖过 2h 等待窗 → 重启流水线（标记幂等）
8. GPU 被他人占用 → 流水线加"等两张 ≥40GB 空卡"逻辑，自动挑最空的卡
9. `pkill -f xxx` 把自己 ssh 会话杀了 → 模式匹配到自身命令行，用 `pgrep` 拿 PID 再 kill

## 七、后续可做

- 常驻服务：FastAPI 包 `gen_test.py` 的加载部分（模型常驻，请求只花 ~11.5min）
- 质量：1344×768 + 50 步（慢 2.3×）；fl2va 任务加 `image=`/`last_image=` 首尾帧
- Ref2VA 变体（参考图/视频/音频生成）：需另下 `Ref2VA/` 144GB + 转换
- 有 sudo 后可选：升驱动走 vllm-omni（OpenAI 兼容 API，吞吐更好）或 sglang docker
