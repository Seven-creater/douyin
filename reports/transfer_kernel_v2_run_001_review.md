# Transfer-kernel v2 run 001: calibration blocked

- Code and fixtures: commit `1641d0d` on `codex/reference-intent-v3`.
- Inputs: frozen `reference_intent_v1`, accepted reference, and the immutable v1 transfer bad-case brief. No full-video request was made.
- Raw run: `data/agentic_runs/transfer_kernel_v2_20260924/run_001/`, copied without editing from the server run. The complete first request, raw response, parsed response, call metadata, and failure record are retained.
- Model work performed: one text call (`calibration_01_mapping`), 677 input tokens, 1,252 output tokens, 237.23 seconds. All other planned calls were **not run**.

## Failure

The first calibration mapping failed `mapping_quote_or_verdict_invalid`. In K04 the model marked the alarm/flood relation as `conflict` but supplied `candidate_quote: null`; the required exact source span cannot be checked. In K03 it marked the hierarchy stance `unmapped` while supplying a non-null candidate quote. These are contract failures even though the accompanying reasons describe the intended semantic distinction. The program did not silently repair either row.

No second permutation, stance judge, frozen bad-case regression, new kernel, public brief, acceptance credential, theme, or story was produced. The result is **blocked at judge calibration**, not evidence that cross-domain abstraction succeeded or failed. The old v1 `run_001` remains unusable by the new trial entry point because it lacks `trial_acceptance_v2`.
