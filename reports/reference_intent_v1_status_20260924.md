# Reference intent v3 trial — 2026-09-24

## Scope and lineage

- Isolated branch: `codex/reference-intent-v3`, currently `17a0365`.
- Existing v1–v4 runs, accepted reference, precise timeline, frozen `creative_structure_spec_v1`, and production chain were not modified.
- Input: the existing accepted reference for `data/videos/7682719919410072847/video.mp4`; static section bundles came from `reference_understanding_v4_20260924/model_run_003/static_review.json`.
- This experiment used the same Qwen3-Omni model for all calls. The user's interpretation and old expected story were not included in model requests.
- Raw requests, responses, configuration hashes, model-call metadata, and failures are preserved in `data/agentic_runs/reference_intent_v1_20260924/`.

## Observed result

The new private intent analysis identified the video's central invitation as challenging a reductive assumption about the person; it connected the opening on-screen statement, competitive action, later attributed statements about multiple abilities, and the final light limitation. The independent model audit marked all four core checks supported, with no critical conflict. This is a **model-checked candidate**, not human-verified truth. The model returned `tone` and `limitations` in shapes different from the optional prompt schema; the current core validator does not reject those optional shape errors. Some audit wording also blurs whether later abilities are shown visually or stated in text, so attribution merits human review before broader claims.

The public creative brief did **not** pass. All three separate abstraction attempts and their audits are retained:

| Run | Public-draft problem | Audit / gate |
| --- | --- | --- |
| `brief_run_001` | Explicitly retained loss of hands and specific original activities; no free slots | `structure_optional=false`; blocked |
| `brief_run_002` | Recast the same condition as loss of a “primary tool for interaction” | `source_surface_absent=false`; blocked |
| `brief_run_003` | Generalized only to a “physical trait,” still requiring a physical-difference story | `source_surface_absent=false`; blocked |

These are not three samples used to select a lucky pass. Each failed output remains visible; the second and third calls followed committed, general abstraction-boundary changes. After the third failed audit, model work stopped rather than adding another case-specific prompt patch. No theme, story, screenplay, image, video, or editing task was launched.

## Status against the v3 plan

- Core communicative-intent extraction and independent grounding audit: **model-checked candidate**; 0 local probes.
- White-listed, source-independent `creative_story_brief_v1`: **blocked** by semantic source-binding leakage.
- Theme hard-constraint / story-soft-preference adapter and unit tests: **implemented but not run against a published brief**.
- Three cross-domain themes, critique, top-two synopses, positive/negative live candidate test: **not run**, because the input brief did not pass.
- Separate editing acceptance and clean-BGM clearance: **not established**; `editing_candidate_ready=false`, `production_release_allowed=false`.
- Second reference with a different story mechanism: **not run**; no generalization claim.

Eight model calls occurred: 2 full audiovisual intent/audit calls and 6 text abstraction/audit calls, totaling 27,765 input tokens, 2,483 output tokens, and 298.17 seconds recorded model-call time. The run does not demonstrate transfer-interface success. It does demonstrate that the narrower understanding task recovered a plausible central reading while the present natural-language abstraction could not reliably separate the source's physical condition from the transferable expression mechanism.

## Artifacts

- `model_run_001/reference_intent_v1.json`: private intent, source SHA, audit and status.
- `model_run_001/calls/`: actual full-media requests and raw responses.
- `brief_run_001/`, `brief_run_002/`, `brief_run_003/`: each draft, independent audit, failure record, and raw request/response.
- Source code and isolated tests: `src/agentic_video/reference_intent_v1.py`, `src/agentic_video/creative_pipeline/intent_trial.py`, `scripts/run_reference_intent_v1.py`, `scripts/run_intent_story_trial.py`, and `tests/test_reference_intent_v1.py`.
