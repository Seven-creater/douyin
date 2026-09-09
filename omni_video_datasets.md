# Omni Video 数据集全面调研

> 调研日期:2026-09-07。服务器 10.1.4.86 已核实:**huggingface.co 直连不通,hf-mirror.com 可用**,`aria2c` 已装,`/data02` 剩 36T。
> 数据根目录:`/data02/pretrained_model/cvr_learn/cvr_data/`(下称 `cvr_data/`)。
> 状态标记:✅ 可直接下 / 🔓 需登录同意条款或邮件申请 / ⚠️ 上游已撤或衰减,靠镜像 / ❌ 已不可用。

---

## 〇、服务器已有 vs 本次新发现(缺口)

| 类别 | 服务器已有 | 本次调研新增候选 |
|---|---|---|
| Omni 视频理解 QA | DailyOmni、WorldSense、AVATAR、AVE、AVQA、AVSCapBench | OmniVideoBench、OmniBench、AV-Odyssey、OmniEval、AVHBench、Video-Holmes、MUSIC-AVQA-v2.0、DAVE、AVUT、LongVALE、OmniMMI |
| Composed Video Retrieval | OmniCVR(完整)、EgoCVR、COVR(WebVid2M/8M)、dense COVR | TF-CoVR、CoVR-R、CoVA(AV-Comp)、MUVR、CC-CoIR、SynthTriplets18M |
| 视频-文本检索/预训练 | MSR-VTT、VATEX(仅脚本)、AudioCaps(部分)、VGGSound(部分) | DiDeMo、ActivityNet Captions、WavCaps、AudioSet 镜像、AVSET-10M、ACAV100M、MiraData、OpenVid-1M、InternVid(元数据) |
| Talking head / 语音人脸 | HDTF、VoxCeleb1+2(354G) | CelebV-HQ、CelebV-Text、MEAD(均需申请) |
| 长视频评测 | LVBench、CG-Bench(部分)、LSDBench(部分) | LongVideoBench、Video-MME/v2、MLVU、EgoSchema、MovieChat-1K、Vript、MVBench、MMBench-Video |

---

## 一、Omni 视频理解 QA 评测集(视频+音频联合,核心)

| 数据集 | 论文 | 哪方面 | 规模 | 下载 | 状态 |
|---|---|---|---|---|---|
| **DailyOmni** | arXiv 2505.17862(复旦) | 日常开放场景,音视频**时序对齐**推理,30/60s 子集 | 684 视频 / 1,197 MCQ | HF `liarliar/Daily-Omni`(CC BY-NC-SA) | ✅ **服务器已有** |
| **WorldSense** | arXiv 2502.04326(小红书+上交) | 真实世界 omni 理解,26 任务,speech/sound/music 三类音频 | 1,662 视频 / 3,172 MCQ | HF `honglyhly/WorldSense`(CC-BY-4.0);镜像 `lmms-lab/WorldSense` | ✅ **服务器已有**(103G) |
| **OmniVideoBench** | arXiv 2510.10689(南大 NJU-LINK) | 音画互补(synergistic)协同推理,8 大类 68 子类,13 题型 | 628 视频 / 1,000+ 题,秒级~30min | HF `NJU-LINK/OmniVideoBench` | ✅ **建议下** |
| **OmniBench** | arXiv 2409.15272(m-a-p,NeurIPS'25 D&B) | 三模态(image+audio+text)同步推理;注意本质是"图+音"(视频的抽帧+音轨) | ~9-10K MCQ | HF `m-a-p/OmniBench`;训练集 `m-a-p/OmniInstruct_v1` | ✅ 建议下 |
| **AV-Odyssey** | arXiv 2412.02611(HKU) | AV 属性(音量/音高/速度/数量)推理;附 DeafTest | 26 任务 / 4,555 题 | HF `AV-Odyssey/AV_Odyssey_Bench` + `AV-Odyssey/Deaftest_dataset`;README 要求邮件 libohao1998@gmail.com 申请(学术) | 🔓 HF 事实上未设门禁 |
| **OmniEval** | arXiv 2506.20960(OpenBMB) | **中英双语** AV 同步理解:属性/时序/推理 | 810 视频(285 中/525 英,均 211s) / 2,905 MCQ | GitHub `OpenBMB/OmniEval` + OmniEvalKit 的 hf_download.py;HF id 未确认 | ✅ 经 GitHub |
| **AVHBench** | arXiv 2410.18325(ICLR'25,KAIST) | AV 跨模态**幻觉**:匹配/顺序/计数/STC,5 任务 | 基于 VALOR+AudioCaps 合成 | GitHub `kaist-ami/AVHBench`(README 直链);HF id 未确认 | ✅ 经 GitHub |
| **MUSIC-AVQA** | CVPR 2022 Oral(GeWu-Lab) | 乐器演奏场景 AVQA,audio/visual/av 三类问题 | 9,288 视频 150h / 45,867 QA | GitHub `GeWu-Lab/MUSIC-AVQA`(GDrive+百度网盘);去偏 v2.0:HF `DraculaDragon/MUSIC-AVQA-v2.0` | ✅(服务器已有 AVQA videos 可能即此系) |
| **Video-Holmes** | arXiv 2505.21374(Tencent ARC) | 悬疑短片找线索,复杂视觉推理(视频为主音频为辅) | 270 短片 / 1,837 题 | HF `TencentARC/Video-Holmes` | ✅ Qwen3-Omni 官方三 AV 基准之一 |
| **DAVE** | arXiv 2503.09321(NeurIPS'25) | 受控变量式 AV 诊断(同步/时序推理) | 2,426 样本 | GitHub `gorjanradevski/dave` | ✅ |
| **AVUT** | arXiv 2503.19951(EMNLP'25,字节) | audio-centric 视频理解,过滤"文本捷径" | ByteDance 自建 | GitHub `lark-png/AVUT` | ✅ |
| **AVSBench** | ECCV'22 + IJCV'24 | 像素级**声源分割**(S4/MS3/AVSS),非 QA | object+semantic 两子集 | 邮件 opennlplab@gmail.com 申请;GitHub `OpenNLPLab/AVSBench` | 🔓 |
| **LongVALE** | CVPR 2025(arXiv 2411.19772) | 首个"视-听-语言**事件**"长视频:定位/检索/QA | 8.4K 长视频(均 235s) / 105,730 时间边界标注事件 | GitHub `ttgeng233/LongVALE` | ✅ 对 composed retrieval 有直接参考价值 |
| **OmniMMI** | CVPR 2025(arXiv 2503.22952) | **流式**视频多模态交互(全局/重定向/回溯推理) | 1,121 互动长视频 | GitHub `OmniMMI/OmniMMI` | ✅ |

> Qwen3-Omni 官方 AV 评测只用了三个:WorldSense、DailyOmni、VideoHolmes。复现其设定:fps=2、无 system prompt、文本问题置于多模态输入之后。

## 二、Composed Video Retrieval(核心方向)

| 数据集 | 论文 | 哪方面 | 规模 | 下载 | 状态 |
|---|---|---|---|---|---|
| **OmniCVR** | ICLR 2026 D&B(清华/CAU) | **首个音频为一等公民的 CoVR**:源视频+修改指令→2000 候选检索;audio-center 1000 / visual-center 1141 / integrated 2858 | 5,000 queries / 16,316 视频,26.8G | HF `Jun-Yang/OmniCVR`(CC-BY-4.0) | ✅ **服务器已有完整版** |
| **CoVR (WebVid-CoVR)** | AAAI 2024(arXiv 2308.14746)+ CoVR-2 TPAMI | 参考视频+修改文本→目标视频,事实标准 | 训练 1.6M triplets(WebVid2M),测试 5,483 人工标注 | 标注 HF `lucas-ventura/WebVid-CoVR`;视频走官方脚本;镜像 `TempoFunk/webvid-10M` | ⚠️ WebVid 官方已撤,**视频尽快落盘** |
| **Dense-WebVid-CoVR** | ICCV 2025(arXiv 2508.14039) | 稠密长修改文本(原版 7 倍信息量),测试全人工核验 | 1.6M 样本 | GitHub `OmkarThawakar/BSE-CoVR` → HF(具体 id 从 repo 进) | ✅ **服务器已有** |
| **EgoCVR** | ECCV 2024(arXiv 2407.16658) | egocentric **细粒度时序/动作**修改;global/local 双 gallery | 2,295 queries | GitHub `ExplainableML/EgoCVR`(clips zip 直链+预计算嵌入,零门槛) | ✅ **服务器已有** |
| **TF-CoVR** | NeurIPS 2025 D&B(arXiv 2506.05274) | **时序细粒度**(体操圈数/跳水规格),每 query 3.9 个合法目标 | 178,639 triplets | HF `ucf-crcv/TF-CoVR`(标注 5.15MB+AIM/BLIP-2 嵌入,MIT);视频另走 FineGym/FineDiving 官方 | ✅ **建议下** |
| **CoVR-R** | arXiv 2603.20190(MBZUAI,CVPR'26 Workshop 挑战赛) | **推理型**:需推断 after-effects(状态/动作顺序/镜头/节奏),带推理轨迹 | 2,800 triplets(WebVid+SSv2 源) | GitHub `mbzuai-oryx/CoVR-R` → HF | ✅ **建议下** |
| **CoVA (AV-Comp)** | ICASSP 2026(arXiv 2601.22508) | 首个考虑**音频差异**的 CoVR:视觉相似但音频不同 | AV-Comp benchmark | GitHub `PerceptualAI-Lab/CoVA` | ✅ 数据发布细节待 repo 更新 |
| **MUVR** | NeurIPS 2025 D&B | 未剪辑 53K 长视频 + 长文本/标签/多模态复合查询 | 53K | GitHub `debby-0527/MUVR` | ✅ |
| **CC-CoIR** | CoVR-2 附带 | 图像侧 3.3M triplets(Conceptual Captions) | 3.3M | CoVR repo `download_annotation.sh coir` | ✅ |
| **SynthTriplets18M** | TMLR 2024(NAVER) | 扩散模型合成 image CIR triplets | 18M | HF `navervision/SynthTriplets18M` | ✅ |
| **Ego4D-NLQ** | CVPR 2022 | 长第一人称视频自然语言 moment 定位(EgoCVR 母体) | 27k queries / 3,670h | ego4d-data.org,**签学术 license,~48h 批准**,CLI 下载 | 🔓 |
| image 侧参考 | CIRR / FashionIQ / CIRCO | CIR 经典基准 | 36k / 30k / 1k | GitHub;FashionIQ 建议走 Kaggle 镜像 | ✅ |

> 2026 方法论文(ReCoVR、UniCVR、MoRe、tara、ZeroSight、OmniRet 等)基本都复用上表数据集;survey 见 arXiv 2503.01334 + Awesome-Composed-Multi-modal-Retrieval。
> 澄清:**MECD / CVCube 查无此 composed retrieval 数据集**(MECD 是 NeurIPS'24 视频因果发现),勿混用。

## 三、视频-音频-文本大规模数据(预训练/检索)

| 数据集 | 哪方面 | 规模 | 下载 | 状态 |
|---|---|---|---|---|
| **VGGSound** | 音视频对应,309 类,10s 片段 | 200K / 550h | 官网已撤;HF 镜像 `Loie/VGGSound`(338G 含视频)或 `hhc1997/vggsound_download` 重抓 | ⚠️ **服务器已有部分(32G seed),建议核对缺口** |
| **AudioSet** | 527 类声音事件,多标签 | 2M 段 | 官方只剩 CSV;HF `agkphysics/AudioSet` 2023-03 快照(仅音频 2.44T,缺~15%);16k 版 `confit/audioset-16khz-wds` | ⚠️ 快照可下,视频副本不存在 |
| **WavCaps** | 音频 caption(检索/训练) | 403K clips / 7,500h | HF `cvssp/WavCaps` | ✅ |
| **AVSET-10M** | 2025 新,自称最大公开**音视频对应**数据集 | 10M | HF `avset10m/avset10m`(下载前核对 README) | ✅ 待核验 |
| **ACAV100M** | AV 对应性筛选片段池 | 100M×10s | acav100m.github.io 分包 | ✅ |
| **VideoCC3M/10M** | image-caption→视频迁移的 AV caption | 3.3M/10M | GitHub `google-research-datasets/videoCC-data`(仅 URL 元数据) | ⚠️ YouTube 衰减 |
| **MSR-VTT** | 开放域 text-to-video 检索基准 | 10K clips / 200K captions,**mp4 含原音轨** | HF `friedrichor/MSR-VTT`(2.26G 含视频) | ✅ **服务器已有** |
| **DiDeMo** | Flickr 段落检索+moment 定位 | 10K 视频 | HF `friedrichor/DiDeMo`(含 mp4) | ✅ 建议下 |
| **ActivityNet Captions** | 长视频 dense caption+检索 | 20K / 849h / 100K 句 | HF `friedrichor/ActivityNet_Captions` | ✅ |
| **VATEX** | 中英双语 caption/跨语言检索 | 41,250 视频 / 825K captions | 官网 Drive 不稳;HF `HuggingFaceM4/vatex`+Kaggle 镜像 | ⚠️ **服务器仅有脚本,视频没下** |
| **AudioCaps** | 音频 caption(AudioSet 子集) | 46K | 服务器已有 master 仓库 | ✅ 已有 |
| **HowTo100M** | 教学视频 ASR 弱标注 | 1.22M 视频 / 136M 对 | HF `HuggingFaceM4/howto100m` **仅元数据**;视频本体存活率 ≤76% 且流程老化 | ❌ 视频不可得,价值降级 |
| **InternVid** | 7M 视频 clip-caption | 760K h / 234M 对 | HF `OpenGVLab/InternVid`(10M-FLT 元数据 13.4G,gated 勾条款;视频自抓) | ⚠️ 元数据可下 |
| **Panda-70M** | 多教师 caption 的 8s HD 片段 | 70M | GitHub `snap-research/Panda-70M`(CSV+脚本) | ✅ 视频自抓 |
| **OpenVid-1M/2M** | T2V 精选训练池 | 1M/2M | HF `nkp37/OpenVid-1M`(视频直下 ~430G) | ✅ |
| **MiraData** | 长时高质+结构化长 caption | 330K | HF `TencentARC/MiraData` | ✅ |
| **AVCaps** | 分模态 caption(音频/视觉/AV 各一套) | 2,061 clips / 28.8h | HF `TUT-ARG/AVCaps` / Zenodo 14536325 | ✅ 对齐评测好用 |
| **Ego4D / Ego-Exo4D** | 第一人称 3,670h / ego+exo 1,286h | 3.85M narration | ego4d-data.org 签 license(~48h),CLI | 🔓 |

## 四、Talking head / 人脸语音(服务器已很全)

| 数据集 | 哪方面 | 规模 | 下载 | 状态 |
|---|---|---|---|---|
| **VoxCeleb1+2** | 说话人/语音,talking head 最大源 | 7,000+ ID | **服务器已有 354G** | ✅ |
| **HDTF** | 高清 talking face | ~362 个 720p/1080p | GitHub `NLOS/HDTF` | ✅ 已有 |
| **AVATAR** | ICCV'25 音视频时空定位 | 高分辨率时序标注 | HF `mipal/AVATAR` | ✅ 已有 |
| **CelebV-HQ** | 35,666 clips / 15,653 ID / 83 属性 | ECCV 2022 | celebv-hq.github.io 填表→OneDrive | 🔓 有版权收紧风险,要就早下 |
| **CelebV-Text** | 70K clip × ~20 条结构化文本(动作/外观/情绪) | CVPR 2023 | celebv-text.github.io 填表 | 🔓 同上 |
| **MEAD** | 60 演员×8 情绪×3 强度×7 视角情感朗读 | ACM MM 2020 | wywu.github.io/projects/MEAD 申请 | 🔓 |
| **MER2024/2025** | 中文影视多模态情感 | 28,544 片段 | HF `MERChallenge/MER2024/2025`(签 EULA) | 🔓 |
| **MELD** | Friends 对话三模态情感 | 13,708 utterance | affective-meld.github.io 直链 | ✅ |

## 五、长视频理解评测(纯视觉为主,补充)

| 数据集 | 哪方面 | 音频? | 规模 | 下载 | 状态 |
|---|---|---|---|---|---|
| **LVBench** | 极长视频 QA(均 68min) | 无 | 103 视频 / ~1,549 题 | HF `zai-org/LVBench` **仅元数据,视频要 yt-dlp 重抓** | ⚠️ 服务器已有(检查视频齐不齐) |
| **LongVideoBench** | 帧-字幕交错长上下文 | 字幕代音频 | 3,763 视频 / 6,678 题 | HF `longvideobench/LongVideoBench` **视频直下** | ✅ 建议下 |
| **Video-MME (v2)** | 900 视频 6 域,短中长三档 | 有音频文件但无官方 audio 设置 | 2,700 题 | HF `lmms-lab/Video-MME`;v2: `MME-Benchmarks/Video-MME-v2`(1080p 直下) | ✅ 建议下 |
| **MLVU** | 9 任务长视频 | 无 | 3,102 题 | HF `MLVU/MVLU`(gated 勾条款即下) | 🔓 |
| **CG-Bench** | 电影 clue-grounded QA,**vision+audio+subtitle 三路** | **有** | 1,219 视频 / 12,129 QA | HF `CG-Bench/CG-Bench`(gated);衍生 `CG-Bench/CG-AV-Counting` | 🔓 **服务器部分下,建议补齐** |
| **EgoSchema** | egocentric 3min 长程推理 MCQ | 无 | 5,063 题 | HF `zobwink/EgoSchema`(~106G) | ✅ |
| **MovieChat-1K** | 电影长视频 QA+dense caption | 无 | 1K / 14K QA | GitHub `wenhaochai/MovieChat`;HF `Enxin/MovieChat-1K-test` | ✅ |
| **Vript** | 剧本式 dense caption(均 145 词)+ Vript-HAL 幻觉/RR 检索 | 无 | 12K clip | HF `Mutonix/Vript`、`Mutonix/Vript-HAL`;ModelScope `mutonix/Vript` | ✅ |
| **LSDBench** | 长视频采样困境(视频重建式) | — | ICCV 2025 | GitHub `JIA-Lab-research/LSDBench` | ⚠️ 服务器部分下 |
| **MVBench / TempCompass / MMBench-Video / VNBench** | 时序理解/短视频 | 无 | 4K/7.5K/2K/1.35K | HF `OpenGVLab/MVBench`、`mmaaz60/tempcompass`、`opencompass/MMBench-Video`、`videoniah/VNBench` | ✅ |

## 六、下载操作手册(服务器,走 hf-mirror)

```bash
# 服务器通用前缀
export HF_ENDPOINT=https://hf-mirror.com

# 小体积评测集(一次性拉全,数 GB 内)
for ds in NJU-LINK/OmniVideoBench m-a-p/OmniBench AV-Odyssey/AV_Odyssey_Bench \
          TencentARC/Video-Holmes DraculaDragon/MUSIC-AVQA-v2.0 \
          longvideobench/LongVideoBench lmms-lab/Video-MME \
          ucf-crcv/TF-CoVR cvssp/WavCaps friedrichor/DiDeMo; do
  nohup hf download "$ds" --repo-type dataset \
    > /data02/pretrained_model/cvr_learn/cvr_data/_cache/dl_$(basename $ds).log 2>&1 &
done

# gated 但只需登录勾条款的(先 hf auth login)
hf download MLVU/MVLU --repo-type dataset
hf download CG-Bench/CG-Bench --repo-type dataset
hf download OpenGVLab/InternVid --repo-type dataset

# 大件(单独跑,aria2c 可加速 hf-mirror 直链)
# OmniCVR 26.8G(服务器已有,无需)
# VGGSound 镜像 338G:核对服务器 32G seed 缺口后再决定
# AudioSet 快照 2.44T:仅音频,需要音频侧预训练再下
```

注意事项:
1. **WebVid 一切依赖数据(CoVR/dense-CoVR 视频侧)尽快落地**:上游官方已声明不再提供,现在靠作者脚本+镜像还能下。
2. **CelebV-HQ/Text、MEAD 有 2025-09 版权点名风险**,要就尽早申请。
3. yt-dlp 重抓类(LVBench 视频、AudioSet 官方、HowTo100M)国内服务器成功率低,优先选 HF 直下版本。
4. HF dataset viewer 报错(如 OmniCVR)不影响文件拉取,用 `hf download` / `huggingface-cli download` 即可。

## 七、优先级建议(结合 omni/composed retrieval 方向)

1. **立刻下(小而精,直接补齐 omni 评测矩阵)**:OmniVideoBench、OmniBench、Video-Holmes、AV-Odyssey、WavCaps、TF-CoVR、DiDeMo、LongVideoBench、Video-MME — 全部 HF 直下,合计 <100G。
2. **尽快决策(有衰减风险)**:WebVid-CoVR 视频子集补齐、VGGSound 缺口核对、AVSET-10M 核验。
3. **需要申请(提前发起,48h~数天)**:Ego4D(若做 egocentric)、CelebV-HQ/Text、MEAD、MER2024/25(签 EULA)。
4. **低优先**:HowTo100M(视频已死)、TVR/LSMDC(无原片)、纯视觉长视频全家桶(LVBench 已有)。
