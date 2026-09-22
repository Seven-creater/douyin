"""Screenplay v1 public payload types; no production backend dependencies."""
from __future__ import annotations

from typing import TypedDict


SCREENPLAY_MAX_CANDIDATES = 3
SCREENPLAY_SELECT_K = 1
SCREENPLAY_SELECTION_POLICY = "wave3_first_valid_fixture_v1"


class FormatConstraints(TypedDict):
    schema_version: str
    target_duration_s: float
    min_duration_s: float
    max_duration_s: float
    max_scenes: int
    max_characters: int
    max_props: int


class Character(TypedDict):
    character_id: str
    role: str


class Scene(TypedDict):
    scene_id: str
    setting: str
    beat_ids: list[str]


class Beat(TypedDict):
    beat_id: str
    scene_id: str
    blueprint_event_ids: list[str]
    action: str
    character_ids: list[str]
    prop_ids: list[str]
    duration_s: float


class TextCue(TypedDict):
    cue_id: str
    beat_id: str
    kind: str  # dialogue | on_screen_text
    speaker_id: str | None
    text: str
    duration_s: float


class RelationTrace(TypedDict):
    blueprint_relation_id: str
    prior_beat_id: str
    evidence_beat_id: str
    updated_beat_id: str


class StructureTrace(TypedDict):
    I0_PRIOR_INTERPRETATION: str
    E1_NEW_INFORMATION: str
    I1_UPDATED_INTERPRETATION: str
    R1_INFORMATION_UPDATE: RelationTrace


class ProductionRequirements(TypedDict):
    character_ids: list[str]
    locations: list[str]
    prop_ids: list[str]
    notes: list[str]


class Screenplay(TypedDict):
    schema_version: str
    screenplay_id: str
    blueprint_id: str
    story_blueprint_sha: str  # SHA of the selected immutable envelope
    creative_structure_sha: str  # public spec artifact_sha
    format_constraints_sha: str  # canonical JSON hash of constraints
    target_duration_s: float
    characters: list[Character]
    scenes: list[Scene]
    beats: list[Beat]
    dialogue_or_text_cues: list[TextCue]
    structure_trace: StructureTrace
    production_requirements: ProductionRequirements
