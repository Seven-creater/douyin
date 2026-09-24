from __future__ import annotations

import pytest

from scripts.run_reference_theme_trial import brief_for_theme


def test_brief_projection_contains_only_abstract_direction():
    source = {"schema_version": "reference_transfer_brief_trial_v1",
              "communicative_goal": "Reconsider an overgeneralized judgment",
              "information_sequence": ["An initial judgment is made",
                                       "Concrete evidence changes its scope"],
              "ending_relation": "An ordinary limitation changes the tone",
              "tone": "warm", "editing_organization": ["slow to fast"],
              "free_slots": ["domain", "relationship"],
              "uncertainties": ["Exact music alignment unknown"]}
    projected = brief_for_theme(source)
    assert "schema_version" not in projected
    assert "uncertainties" not in projected
    assert projected["information_sequence"] == source["information_sequence"]


def test_brief_projection_rejects_reference_ids_and_missing_sequence():
    source = {"schema_version": "reference_transfer_brief_trial_v1",
              "communicative_goal": "S1_T02 matters",
              "information_sequence": ["Something changes"],
              "ending_relation": "unknown", "tone": "unknown",
              "editing_organization": [], "free_slots": [], "uncertainties": []}
    with pytest.raises(Exception, match="reference_source_id_leak"):
        brief_for_theme(source)
    source["communicative_goal"] = "Some judgment changes"
    source["information_sequence"] = []
    with pytest.raises(ValueError, match="brief_information_sequence_missing"):
        brief_for_theme(source)
