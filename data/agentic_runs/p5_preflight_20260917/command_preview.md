# P5 命令预览（未执行）

本文件只是下一次资源可用时的操作顺序。当前 8 张 GPU 均被其他用户任务占用；Gate A/B 均未开始。执行前必须重新只读核验 GPU、服务器代码、工作树和参考视频 SHA，不得沿用本次空闲结论。

1. 服务器工作树目前有两个未跟踪文件。保留它们；在符合“工作树干净”的约定或得到明确处理方式前，不执行同步。服务器 HEAD 目前为 `3568812`，落后本地 `origin/main` 的 `22e99a8`。同步只能用 `git pull --ff-only`，不能 reset、强推或覆盖未跟踪文件。
2. Gate A 的命令形态：

   ```bash
   python -m src.agentic_video.cli reference-program-v9 \
     --reference data/videos/7682719919410072847/video.mp4 \
     --output data/agentic_runs/p5_v9_reference_20260917 \
     --gpu-pairs '<VERIFIED_FREE_PAIR>'
   ```

   自动输出后先停止；逐 Section 审核并确认匿名 Requirement Transfer Test。只有审核文件绑定当前 reference 和三个 Program 的 SHA、所有必选项通过，才可单独调用 `reference-program-v9-accept`。本次不生成或填写审核答案。
3. Gate B 前可生成 CPU 计划：

   ```bash
   python -m src.agentic_video.cli reference-generate-v9g \
     --phase capability --plan-only \
     --fixture-dir '<P5_DIAGNOSTIC_FIXTURE_DIR>' \
     --output data/v9g_runs/p5_capability_20260917
   ```

   当前仓库的非 `--plan-only` capability 命令会显式阻断，不能把它当作真实 S0–S7 执行命令。Gate A 未冻结时不得启动 H3；即使 Gate A 通过，仍须先有能保存真实请求、媒体与效果证据的执行路径，不能回退到“MP4 存在即 effective”的旧探测。

禁止顺手运行 Generation Unit Pilot、Repair 或完整视频生成。
