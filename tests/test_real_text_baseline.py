from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.creative_pipeline.planning.story import (
    build_fake_story_blueprints,
)
from src.agentic_video.creative_pipeline.planning.theme import (
    build_fake_theme_candidates,
)
from src.agentic_video.creative_pipeline.real_text_baseline import (
    run_real_text_baseline,
)
from src.agentic_video.creative_pipeline.writing.adapter import (
    FakeScreenplaySkill,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (ROOT / "experiments/creative_structure_minimality/r2_c5/"
             "creative_structure_spec_v1.json")
FORMAT = {"schema_version": "screenplay_format_constraints_v1",
          "target_duration_s": 30.0, "min_duration_s": 25.0,
          "max_duration_s": 35.0, "max_scenes": 3,
          "max_characters": 4, "max_props": 3}


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


class StubRunner:
    def __init__(self, *, leak: bool = False) -> None:
        self.prompts: list[str] = []
        self.leak = leak

    def ask(self, prompt: str, **kwargs) -> SimpleNamespace:
        self.prompts.append(prompt)
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        if len(self.prompts) == 1:
            candidate = build_fake_theme_candidates(_spec()["artifact_sha"])[0]
            result = {key: value for key, value in candidate.items()
                      if key not in {"schema_version", "theme_id",
                                     "parent_structure_sha"}}
            if self.leak:
                result["accepted_claims"] = ["S1_V01"]
        elif len(self.prompts) == 2:
            candidate = build_fake_story_blueprints(
                _spec()["artifact_sha"], payload["selected_theme"])[0]
            result = {key: value for key, value in candidate.items()
                      if key not in {"schema_version", "blueprint_id",
                                     "theme_id", "parent_structure_sha",
                                     "structure_bindings"}}
        else:
            screenplay = FakeScreenplaySkill().run(payload["writer_request"])
            result = {key: screenplay[key] for key in (
                "scenes", "beats", "dialogue_or_text_cues")}
            for index, beat in enumerate(result["beats"], 1):
                beat["action"] = f"Visible test action {index}."
        return SimpleNamespace(text=json.dumps(result), input_tokens=10,
                               output_tokens=20, elapsed_s=0.1)


def test_one_chain_uses_only_public_structure_and_keeps_raw_traces(
        tmp_path: Path) -> None:
    runner = StubRunner()
    result = run_real_text_baseline(
        runner, _spec(), tmp_path / "baseline", format_constraints=FORMAT)

    assert result["status"] == "PASS"
    assert result["model_calls"] == 3
    assert result["intent_used"] is False
    assert result["quality_evaluation"] == "not_run"
    assert len(runner.prompts) == 3
    assert all("narrative_interpretation" not in prompt
               and "accepted_claims" not in prompt for prompt in runner.prompts)
    for stage in ("01_theme", "02_story", "03_screenplay"):
        assert (tmp_path / "baseline" / stage / "request.json").is_file()
        assert (tmp_path / "baseline" / stage / "raw_response.txt").is_file()
        assert (tmp_path / "baseline" / stage / "candidate.json").is_file()


def test_invalid_theme_blocks_downstream_without_retry(tmp_path: Path) -> None:
    runner = StubRunner(leak=True)
    with pytest.raises(ValueError, match="theme_fields_invalid"):
        run_real_text_baseline(
            runner, _spec(), tmp_path / "blocked", format_constraints=FORMAT)

    assert len(runner.prompts) == 1
    result = json.loads((tmp_path / "blocked" / "result.json").read_text())
    assert result["status"] == "BLOCKED"
    assert result["failed_stage"] == "01_theme"
    assert result["model_calls_with_response"] == 1
    assert not (tmp_path / "blocked" / "02_story").exists()


def test_output_directory_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_real_text_baseline(
            StubRunner(), _spec(), output, format_constraints=FORMAT)
    assert (output / "existing.txt").read_text() == "keep"


def test_reference_zone_brief_is_rejected_before_model_call(
        tmp_path: Path) -> None:
    runner = StubRunner()
    with pytest.raises(ValueError, match="reference_boundary_field"):
        run_real_text_baseline(
            runner, _spec(), tmp_path / "blocked", format_constraints=FORMAT,
            user_brief={"accepted_claims": ["S1_V01"]})
    assert runner.prompts == []
