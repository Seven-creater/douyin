# Reference transfer v2 — model-check status (2026-09-24)

This is an isolated candidate experiment. It did not modify the frozen creative specification or start theme, screenplay, image, or video generation.

## Inputs and run lineage

- Source video SHA-256: `2f95e24edd2cf4b79cc1f40f7e202174c53a6abcf084e728bb49e3ca92938a17`.
- Accepted deterministic review SHA: `c9055fed48ac9e312faa8fb1804e2b1ae790ac58f553672421b50c1725baca79`.
- `model_run_001`: eight local gap probes and a full audiovisual analysis; full-media audit failed with GPU OOM. The run is retained unchanged.
- `model_run_002`: three section analyses, three section audits, a global story, and a global audiovisual audit. Its audit outputs combined multiple requested IDs into single checks, so they are not valid coverage evidence. The run is retained unchanged.
- `model_run_003`: independent section/global audiovisual audits of the frozen `model_run_002` analysis. All four calls used the same Qwen3-Omni model, on GPU pairs 0–1, 2–3, 4–5, and 6–7. Raw requests and responses are retained. The deterministic compile did not alter the model's semantic claims.

## Actual result

| Check | Result |
| --- | --- |
| Accepted content-shot slots | 17 |
| Adjacent shot edges | 16 |
| Explicit transition slots | 6 |
| Narrative units proposed | 3 |
| Requested audit IDs returned | 53 / 53 |
| Audit verdicts | 35 supported; 18 insufficient; 0 contested |
| Theme stance | Model wrote `unknown`; global audit said `insufficient` |
| Ending | Model wrote relation and tone `unknown`; global audit's supported verdict does not supply or repair the missing interpretation |
| Cinematic-technique audit | 17 / 17 insufficient; unverified technique fields are withheld from public slots |
| Story candidate ready | false |
| Editing candidate ready | false |
| Production release allowed | false |

The model can enumerate timed shots and many adjacent transitions, but this run has not established the communicative stance or how the final line changes the preceding information. Its technique descriptions have not passed media audit. No `creative_transfer_spec_v2`, theme, or timed story was generated from this blocked blueprint.

## Audio boundary

The discovered `template_7682.m4a` is the original mixed audio, not a verified clean BGM stem. The v2 audio policy allows reference BGM reuse and new voiceover, dialogue, and on-screen text, but production is blocked until a clean reusable BGM source is verified. Beat synchronization remains unverified.

## Verification

- `python -m pytest -q tests/test_reference_transfer_v2.py`: 12 passed.
- `python -m pytest -q tests`: 1079 passed, 1 skipped.
- Unscoped `python -m pytest -q` is not a valid project test invocation here: it collects the archived third-party AI Video Agent pilot and exits during DaVinci Resolve environment probing.

The original server run is at `/data02/usr/wangqihao/Demo/research/data/agentic_runs/reference_transfer_v2_20260924/model_run_003/`; a byte-for-byte copy of that directory is available in the local workspace at `data/agentic_runs/reference_transfer_v2_20260924/model_run_003/`.
