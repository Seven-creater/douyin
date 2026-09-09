from __future__ import annotations

import json

from src.agentic_video.benchmark import build_suite_specs
from src.agentic_video.critic_v2 import (apply_recipe_patches, build_fault_suite,
                                         parse_critique, score_fault_critiques,
                                         should_stop, validate_patch)
from src.agentic_video.recipe_v2 import recipe_hash, validate_recipe_v2


def test_parse_critic_normalizes_aspects_and_unknown_issue():
    raw = json.dumps({"score": 1.2, "aspects": {"recipe_fidelity": 0.8},
                      "issues": [{"aspect": "other", "t_s": 0.2}],
                      "patches": [], "verdict": "x"})
    critique = parse_critique(raw)
    assert critique["score"] == 1.0
    assert critique["aspects"]["theme_relevance"] == 0.0
    assert critique["issues"][0]["aspect"] == "render_integrity"


def test_patch_guard_applies_allowed_change_and_rejects_evidence_edit():
    recipe = build_suite_specs()[12]
    patch = {"op": "replace", "path": "/operations/0/params/rate", "value": 1.5}
    assert validate_patch(recipe, patch) == []
    updated, audit = apply_recipe_patches(recipe, [patch])
    assert audit[0]["status"] == "applied"
    assert updated["operations"][0]["params"]["rate"] == 1.5
    denied = {"op": "remove", "path": "/operations/0/evidence"}
    assert validate_patch(recipe, denied)


def test_patch_that_breaks_recipe_is_rejected():
    recipe = build_suite_specs()[0]
    patch = {"op": "replace", "path": "/operations/0/confidence", "value": 1.5}
    updated, audit = apply_recipe_patches(recipe, [patch])
    assert audit[0]["status"] == "rejected"
    assert updated == recipe and validate_recipe_v2(recipe) == []


def test_bounded_stopping_rules():
    recipe = build_suite_specs()[0]
    assert should_stop([{"recipe_hash": recipe_hash(recipe), "score": 0.5}], recipe)[1] \
        == "recipe_repeated"
    other = build_suite_specs()[1]
    assert should_stop([{"score": 0.5}, {"score": 0.51}], other)[1] \
        == "insufficient_improvement"
    assert should_stop([{"score": 0.5}, {"score": 0.8}], other)[1] == "max_rounds"


def test_fault_suite_and_threshold_score():
    faults = build_fault_suite(build_suite_specs(), count=60)
    critiques = [{"issues": [{"operation_id": row["operation_id"]}]} for row in faults]
    score = score_fault_critiques(faults, critiques, clean_false_positives=1, clean_count=60)
    assert len(faults) == 60
    assert score["passes_thresholds"] is True
