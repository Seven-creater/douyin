"""P3 fake-only progressive execution harness; never calls H3 or Omni.

The harness tests the order and state boundary of future execution. It does
not construct H3 requests or grant authorization for a real backend.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .generation_p2 import compile_h3_context_preview


REVIEW_STATUSES = {"PASS", "SEMANTIC_FAIL", "UNKNOWN", "INFRA_FAIL"}


class FakeRuntimeBlocked(ValueError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.reason_code = reason_code


def _review_status(value: Any) -> str:
    if not isinstance(value, dict) or value.get("status") not in REVIEW_STATUSES:
        raise FakeRuntimeBlocked("review_status_invalid")
    return str(value["status"])


def _valid_snippets(snippets: Any, duration_s: float) -> float:
    if not isinstance(snippets, list) or not snippets:
        raise FakeRuntimeBlocked("snippets_missing")
    last_end = 0.0
    for snippet in snippets:
        if (not isinstance(snippet, dict) or
                not isinstance(snippet.get("start_s"), (int, float)) or
                not isinstance(snippet.get("end_s"), (int, float))):
            raise FakeRuntimeBlocked("snippet_interval_invalid")
        start, end = float(snippet["start_s"]), float(snippet["end_s"])
        if start < last_end or end <= start or end > duration_s:
            raise FakeRuntimeBlocked("snippet_interval_invalid")
        last_end = end
    return last_end


def run_fake_progressive_runtime(draft: dict[str, Any], catalog: dict[str, Any],
                                 fake: Any, *,
                                 initial_state: dict[str, Any] | None = None) \
        -> dict[str, Any]:
    """Exercise compile→observe→align→verify→crop→commit on fixed fixtures.

    `fake` is a caller-supplied in-memory fixture with seven named methods.
    This function emits no real generation request and cannot authorize use.
    """
    if getattr(fake, "is_fake_backend", False) is not True:
        raise FakeRuntimeBlocked("fake_backend_required")
    if not any(section.get("generation_units") for section in draft.get("sections") or []):
        raise FakeRuntimeBlocked("generation_units_missing")
    state = deepcopy(initial_state or {})
    trace = []
    commits = []
    for section in draft.get("sections") or []:
        for unit in section.get("generation_units") or []:
            unit_id = unit["unit_id"]
            context = compile_h3_context_preview(draft, catalog, unit_id=unit_id)
            record: dict[str, Any] = {
                "unit_id": unit_id, "context_sha256": context["context_sha256"],
                "input_state": deepcopy(state), "execution_authorized": False,
            }
            trace.append(record)
            try:
                raw = fake.generate(context, deepcopy(state))
                duration = float(raw["duration_s"])
                if not 4 <= duration <= 15:
                    raise FakeRuntimeBlocked("raw_duration_invalid", unit_id)
                observed = fake.observe_raw(raw)
                alignment = fake.align_phases(observed, deepcopy(unit))
                if not isinstance(alignment, dict) or set(alignment) != set(
                        unit["semantic_phases"]):
                    raise FakeRuntimeBlocked("phase_alignment_incomplete", unit_id)
                raw_review = fake.verify_contract(observed, alignment, deepcopy(unit))
                record["raw_review"] = deepcopy(raw_review)
                if _review_status(raw_review) != "PASS":
                    record["status"] = _review_status(raw_review)
                    return {"status": "BLOCKED", "trace": trace,
                            "commits": commits, "execution_authorized": False}
                snippets = fake.select_moments(raw, observed, deepcopy(unit))
                endpoint = _valid_snippets(snippets, duration)
                crop_review = fake.review_snippets(raw, deepcopy(snippets))
                record["snippet_review"] = deepcopy(crop_review)
                if _review_status(crop_review) != "PASS":
                    record["status"] = _review_status(crop_review)
                    return {"status": "BLOCKED", "trace": trace,
                            "commits": commits, "execution_authorized": False}
                accepted = fake.state_at(raw, endpoint)
                if (not isinstance(accepted, dict) or accepted.get("pts_s") != endpoint or
                        not isinstance(accepted.get("state"), dict)):
                    raise FakeRuntimeBlocked("snippet_endpoint_state_invalid", unit_id)
                candidate_state = accepted["state"]
                for key, level in section["continuity"].items():
                    if level == "required" and key in unit["continuity_ids"] and \
                            candidate_state.get(key) != unit["continuity_ids"][key]:
                        raise FakeRuntimeBlocked("required_continuity_broken", key)
                state = deepcopy(candidate_state)
                commit = {"unit_id": unit_id, "accepted_snippet_endpoint_s": endpoint,
                          "state": deepcopy(state)}
                commits.append(commit)
                record.update({"status": "PASS", "snippets": deepcopy(snippets),
                               "commit": deepcopy(commit)})
            except FakeRuntimeBlocked as exc:
                record.update({"status": "BLOCKED", "reason_code": exc.reason_code})
                return {"status": "BLOCKED", "trace": trace, "commits": commits,
                        "execution_authorized": False}
            except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                record.update({"status": "INFRA_FAIL", "error": str(exc)})
                return {"status": "BLOCKED", "trace": trace, "commits": commits,
                        "execution_authorized": False}
    return {"status": "FAKE_CHAIN_PASS", "trace": trace, "commits": commits,
            "execution_authorized": False}
