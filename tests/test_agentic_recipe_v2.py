from __future__ import annotations

from src.agentic_video.recipe_v2 import (migrate_v1_to_v2, new_recipe,
                                         recipe_hash, validate_recipe_v2)


def _base():
    return new_recipe(reference_id="r", reference_uri="r.mp4", sha256="a" * 64,
                      duration_s=4.0, fps=24.0)


def test_minimal_recipe_v2_is_valid_and_hash_is_deterministic():
    recipe = _base()
    assert validate_recipe_v2(recipe) == []
    assert recipe_hash(recipe) == recipe_hash(dict(recipe))


def test_recipe_v2_rejects_unknown_track_missing_evidence_and_cycle():
    recipe = _base()
    recipe["operations"] = [{
        "id": "a", "type": "speed_ramp", "interval": [1.0, 2.0],
        "track_id": "missing", "inputs": [], "depends_on": ["a"], "params": {},
        "evidence": [], "confidence": 0.8, "status": "supported",
    }]
    errors = validate_recipe_v2(recipe)
    assert any("track_id unknown" in e for e in errors)
    assert any("requires evidence" in e for e in errors)
    assert any("cycle" in e for e in errors)


def test_v1_migration_builds_layer_tracks_and_preserves_details():
    old = {
        "video": {"aweme_id": "x", "duration_s": 5.0},
        "segments": [{"start": 0, "end": 5}],
        "operations": [{"t_s": 1.0, "type": "tracked_mask_fill",
                        "texture_sequence": ["fire"], "confidence": 0.7,
                        "evidence": {"source": "window", "window_idx": 2}}],
    }
    recipe = migrate_v1_to_v2(old, reference_uri="x.mp4", reference_sha256="b" * 64)
    assert recipe["recipe_version"] == "2.0"
    assert any(t["type"] == "mask" for t in recipe["tracks"])
    assert recipe["operations"][0]["params"]["legacy"]["texture_sequence"] == ["fire"]
    assert validate_recipe_v2(recipe) == []
