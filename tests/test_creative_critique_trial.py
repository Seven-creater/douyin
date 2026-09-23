from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentic_video.creative_pipeline.critique_trial import (
    STAGES, run_critique_trial,
)
from src.agentic_video.manifest import json_hash


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/agentic_runs/r2d_real_text_baseline_20260923_2f1b309"


def _inputs() -> tuple[dict, dict, dict]:
    paths = (
        ROOT / "experiments/creative_structure_minimality/r2_c5/"
        "creative_structure_spec_v1.json",
        BASELINE / "01_theme/candidate.json",
        BASELINE / "02_story/candidate.json",
    )
    return tuple(json.loads(path.read_text(encoding="utf-8"))
                 for path in paths)


class StubRunner:
    def __init__(self, bad_critic: bool = False) -> None:
        self.prompts: list[str] = []
        self.bad_critic = bad_critic

    def ask(self, prompt: str, **kwargs) -> SimpleNamespace:
        self.prompts.append(prompt)
        index = len(self.prompts)
        _, _, story = _inputs()
        if index in (1, 2, 4):
            keys = ("logline", "characters", "setting", "goal", "stakes",
                    "events", "event_relations", "production_assumptions")
            result = {key: story[key] for key in keys}
            result["logline"] = f"Revision {index}: {story['logline']}"
        elif index == 3:
            result = {"strengths": [], "issues": [], "limitations": []}
            if self.bad_critic:
                result["extra"] = "invalid"
        elif index == 5:
            result = self._judgment("right")
        else:
            result = self._judgment("left")
        return SimpleNamespace(text=json.dumps(result), input_tokens=12,
                               output_tokens=24, elapsed_s=0.1)

    @staticmethod
    def _judgment(preferred: str) -> dict:
        return {"preferred": preferred, "rationale": "Exploratory preference",
                "criterion_notes": {key: "Compared both stories" for key in (
                    "mechanism", "coherence", "filmability", "originality",
                    "feasibility")}}


def test_isolated_trial_shares_baseline_and_swaps_judge_order(
        tmp_path: Path) -> None:
    structure, theme, story = _inputs()
    runner = StubRunner()
    output = tmp_path / "trial"
    result = run_critique_trial(runner, structure, theme, story, output)

    assert result["status"] == "PASS"
    assert result["model_calls_with_response"] == 6
    assert result["order_consistent"] is True
    assert result["production_committed"] is False
    assert len(runner.prompts) == 6
    first = json.loads(runner.prompts[0].split("INPUT_JSON:\n", 1)[1])
    critic = json.loads(runner.prompts[2].split("INPUT_JSON:\n", 1)[1])
    assert json_hash(first["current_story"]) == json_hash(
        critic["current_story"]) == json_hash(story)
    left_right = json.loads(runner.prompts[4].split("INPUT_JSON:\n", 1)[1])
    right_left = json.loads(runner.prompts[5].split("INPUT_JSON:\n", 1)[1])
    assert left_right["left_story"] == right_left["right_story"]
    assert left_right["right_story"] == right_left["left_story"]
    assert all("accepted_claims" not in prompt and "reference_video" not in prompt
               for prompt in runner.prompts)
    assert all('"text_only"' not in prompt and '"media_generation"' not in prompt
               for prompt in runner.prompts)
    for stage in STAGES:
        assert (output / stage / "request.json").is_file()
        assert (output / stage / "raw_response.txt").is_file()


def test_invalid_critic_stops_without_retry(tmp_path: Path) -> None:
    runner = StubRunner(bad_critic=True)
    with pytest.raises(ValueError, match="critic_fields_invalid"):
        run_critique_trial(runner, *_inputs(), tmp_path / "blocked")
    assert len(runner.prompts) == 3
    result = json.loads((tmp_path / "blocked/result.json").read_text())
    assert result["status"] == "BLOCKED"
    assert result["failed_stage"] == "03_critic"
    assert not (tmp_path / "blocked/04_guided_revision").exists()


def test_nonempty_output_dir_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_critique_trial(StubRunner(), *_inputs(), output)
    assert (output / "keep.txt").read_text() == "keep"
