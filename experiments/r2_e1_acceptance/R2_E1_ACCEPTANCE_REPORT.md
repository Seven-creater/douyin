# R2-E.1 controlled real generation acceptance

Status: **PASS for generation and artifact integrity**. Semantic video evaluation was not run.

- Code: `2e637585` on `creative-structure-v1`.
- Model: `MiniMaxAI/MiniMax-H3`, version `MiniMax-H3-diffusers-c73ad285`.
- Checkpoint content manifest SHA256: `c73ad285ecdca57fc32250b6befe820ecc3d7bc5c97952e8b2491f78fc4d8f11`. This is the SHA256 of the sorted per-file SHA256 manifest, excluding the model directory's Hugging Face `.cache`.
- Server run root: `/data02/usr/wangqihao/Demo/research/data/agentic_runs/r2_e1_20260923/` on `10.1.4.86`.
- Successful workspace: `workspace_v2/`; media: `media_v2/`; result: `workspace_v2/r2_e1_acceptance_result.json`.
- Result file SHA256: `6ff4456e4852f24720fbf8239516f6685bcafed8044609926c219f3b6eb47e2d`.
- Workspace state SHA256: `46f063c7a32b04495f38de96c4f78172167a36c1e0638fe454a764980eff748d`.
- Embedded lineage report SHA: `ba7cca9413956a1154e89c8e1a141a17e481566a827b75c0af1eb4c074fcf2df` (recomputed and matched).

The three committed shot contracts cover `I0_PRIOR_INTERPRETATION`, `E1_NEW_INFORMATION`, and `I1_UPDATED_INTERPRETATION`. Each shot generated three independent five-second candidates at 9:16 with 50 inference steps, for nine model calls and nine MP4 files. Each real candidate is in its shot's `creative:real_video_pool:*` with status `generated`. No real candidate selection or semantic evaluation artifact was created; the result records `evaluation=not_run`, `selection=not_run`, and `full_video_generation=false`.

| Candidate | Seed | MP4 SHA256 |
| --- | ---: | --- |
| `RVID_PROD_SHOT_001_01` | 201 | `2d15c57b1d246d49b305bd7a29eab58a6181b8172a595bf8ff4cf3920b59f30f` |
| `RVID_PROD_SHOT_001_02` | 202 | `26ae057c46997c9d71682b2fa191ae1ad53262e4950dbdfb444acb07ba22a6ca` |
| `RVID_PROD_SHOT_001_03` | 203 | `85f18bd0bf0290d73a48dc672e3ac44c9c7150e5aaa6758ba36fcf2da1d5a401` |
| `RVID_PROD_SHOT_002_01` | 201 | `fece1ea04c19cba4a36e63125852ae13684cd7ffd84eaa1dbc068d39ddb5b98c` |
| `RVID_PROD_SHOT_002_02` | 202 | `dc593dbe02818a9c903f9c31b966c1b96645cc57e1d81ce84ce2e3682710ddb6` |
| `RVID_PROD_SHOT_002_03` | 203 | `1a0626b9138ad266b8ac63f2a57d2dd85e61a7cb899623ab8eabfcc9be6cf51b` |
| `RVID_PROD_SHOT_003_01` | 201 | `dc732dd6a3cbbd395efa5dbdbbf44064bc009b509283b9f127081310726c3e79` |
| `RVID_PROD_SHOT_003_02` | 202 | `d4bcb22ee841cc818c464ab8161e98c18577fb70cdd32fc36f872c2326657265` |
| `RVID_PROD_SHOT_003_03` | 203 | `ab8dbeb5ed740b6363d1ce0e0d0dad7cfb40fdd240bf8f49156c780a2d18b52d` |

Verification read all nine media files and recomputed their SHA256 against both the generated-media artifact and the candidate's content SHA. It also checked current input artifact SHAs, model identity, model SHA, per-shot prompt SHA, generation config SHA, seed, committed artifact status, pool counts, unique media paths, structure-role coverage, and lineage report SHA. All checks passed. The common generation config SHA is `05ce477ccdea1fc6fae957adba3420bd626d1648e9a3ef0f45a68cb40f3853b4`. Per-shot prompt SHAs are preserved in each generation experiment record.

The first attempt used ten-second shots and failed all three first-shot calls with CUDA OOM. Its log is preserved at `acceptance.log` with service traces under `services/`. The successful attempt used five-second shots and PyTorch expandable CUDA segments; its records are separate under `workspace_v2/`, `media_v2/`, and `services_v2/`. GPUs 0/1, 4/5, and 6/7 were used. The three services started for this experiment were stopped after verification.

The upstream R2-D theme, screenplay, shot plan, and storyboard used here are deterministic fixtures. In particular, the screenplay action remains a contract placeholder. This result verifies the real model transport and provenance path; it does not establish that the rendered videos convey the intended narrative structure, asset consistency, or creative quality.
