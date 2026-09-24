# Transfer-kernel v1 run 001: manual integrity review

- Code: `a29bce1` on `codex/reference-intent-v3`.
- Input: the frozen `reference_intent_v1` candidate and accepted reference; no new full-video call.
- Raw run: `data/agentic_runs/transfer_kernel_v1_20260924/run_001/` (also retained at the corresponding server run path). All three text requests, raw responses, parsed results, and call metadata are preserved.
- Calls: one kernel, one independent kernel audit, one contrast evaluation; 5,630 input tokens, 1,698 output tokens, 324.28 seconds in reported model-call time. No retry, theme, story, screenplay, image, or video generation.

## Machine outcome and observed contradiction

The run's mechanical result says `model_checked_candidate` and 4/4 contrast matches. This is **not an accepted transferable brief**. Its public `stance` says “lacks a specific physical capability”; role V1 requires a “physical condition”; V2 requires achievement “despite physical constraints”; and the free slot is `physical characteristic`. These are source-domain restrictions in the public invariant, not merely private `source_binding` entries. An unrelated basis for underestimation, such as the exam-score case C02, cannot satisfy those statements literally.

The independent audit nevertheless marked every element `supported` and reported zero binding leaks. Its reason for V1 explicitly treated substituting *other physical conditions* as sufficient rebinding. The contrast evaluator then accepted C02 by describing only a generic “initially judged incapable” relation and omitting the brief's physical-condition constraint. Thus 4/4 agreement reflects evaluator inconsistency, not demonstrated cross-domain transfer.

## Decision

Treat the run as a **blocked bad case** despite its machine status. Do not pass `creative_story_brief_v2.json` to theme generation or production. Preserve the machine result unchanged for diagnosis; this report is a separate review disposition, not a rewrite of model output. The planned downstream Theme → Story trial was not run. The four fixtures are development contrasts, not independent generalization evidence, and a second reference remains untested.
