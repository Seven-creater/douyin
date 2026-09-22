"""Frozen R2-D stage graph and deterministic Wave 1 orchestration."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from src.agentic_video.creative_pipeline.contracts import (
    ArtifactEnvelope,
    CandidatePool,
    CandidateRecord,
    ContractError,
    SelectionArtifact,
    artifact_ref,
    build_validation_report,
    validate_candidate_pool,
    validate_selection_artifact,
    validate_story_blueprint,
    validate_theme_candidate,
)
from src.agentic_video.creative_structure_v1.freeze import validate_frozen_spec
from src.agentic_video.manifest import json_hash
from src.agentic_video.provenance import build_lineage_report
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec
from src.agentic_video.workspace import Workspace


FROZEN_STRUCTURE_SHA = (
    "5a22fb5af29588b6220fbe9ccda751c922abe617fca1716b2f97f2942218a967")
SELECTION_POLICY_VERSION = "wave1_first_valid_v1"

FROZEN_STAGE_DAG = (
    {
        "stage": "theme",
        "requires": ["creative_structure_spec"],
        "produces": "theme_candidate_pool",
    },
    {
        "stage": "story_blueprint",
        "requires": ["creative_structure_spec", "theme_selection"],
        "produces": "story_blueprint_candidate_pool",
    },
    {
        "stage": "screenplay",
        "requires": ["creative_structure_spec", "story_blueprint_selection"],
        "produces": "screenplay_candidate_pool",
    },
    {
        "stage": "asset",
        "requires": ["screenplay_selection"],
        "produces": "asset_graph",
    },
    {
        "stage": "shot",
        "requires": ["screenplay_selection", "asset_graph"],
        "produces": "shot_plan",
    },
)


class PipelineBlocked(RuntimeError):
    def __init__(self, reason_code: str, detail: object = "") -> None:
        super().__init__(f"{reason_code}:{detail}" if detail else reason_code)
        self.reason_code = reason_code
        self.detail = detail


class CreativePipelineOrchestrator:
    """Execute only the model-free Theme/Story portion authorized in Wave 1."""

    def __init__(self, workspace: Workspace, registry: SkillRegistry) -> None:
        self.workspace = workspace
        self.registry = registry
        self.call_counts: dict[str, int] = {}
        self._runtime_package_sha = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()

    @staticmethod
    def _producer(spec: SkillSpec) -> dict[str, str]:
        if not spec.package_sha or not spec.prompt_or_instruction_sha:
            raise PipelineBlocked("skill_provenance_incomplete", spec.name)
        return {
            "skill_id": spec.name,
            "skill_version": spec.skill_version,
            "package_sha": spec.package_sha,
            "prompt_or_instruction_sha": spec.prompt_or_instruction_sha,
        }

    def _runtime_producer(self, activity: str) -> dict[str, str]:
        return {
            "skill_id": f"creative_pipeline.{activity}",
            "skill_version": "1.0.0",
            "package_sha": self._runtime_package_sha,
            "prompt_or_instruction_sha": json_hash({"activity": activity}),
        }

    def _write_committed(self, name: str, value: dict[str, Any], *,
                         dependencies: list[str],
                         derived_from: list[dict[str, str]],
                         producer: dict[str, str]) -> dict[str, str]:
        for dependency in dependencies:
            self.workspace.set_dependency(name, dependency, "committed")
        version = self.workspace.write_draft(name, value, metadata={
            "derived_from": [dict(row) for row in derived_from],
            "created_by": producer["skill_id"],
            "schema_version": "artifact_provenance_v1",
            "skill_id": producer["skill_id"],
            "skill_version": producer["skill_version"],
            "package_sha": producer["package_sha"],
            "prompt_or_instruction_sha": producer[
                "prompt_or_instruction_sha"],
        })
        self.workspace.mark_tested(name)
        self.workspace.commit(name)
        sha = self.workspace.get_sha(name)
        if not sha:
            raise PipelineBlocked("artifact_commit_failed", name)
        return {"artifact_id": name, "version": version, "sha": sha}

    def _write_envelope(self, name: str, *, artifact_type: str,
                        payload_schema_version: str,
                        payload: dict[str, Any],
                        dependencies: list[str],
                        parent_refs: list[dict[str, str]],
                        producer: dict[str, str]) -> dict[str, str]:
        existing = self.workspace.state["artifacts"].get(name) or {}
        envelope_version = f"v{len(existing.get('versions') or {}) + 1}"
        envelope = ArtifactEnvelope(
            artifact_id=name,
            artifact_type=artifact_type,
            schema_version=payload_schema_version,
            version=envelope_version,
            payload=payload,
            producer=producer,
            derived_from=tuple(parent_refs),
            policy_versions={"selection": SELECTION_POLICY_VERSION},
            status="committed",
        ).to_dict()
        return self._write_committed(
            name,
            envelope,
            dependencies=dependencies,
            derived_from=parent_refs,
            producer=producer,
        )

    def bootstrap_structure(self, spec: dict[str, Any]) -> dict[str, str]:
        validate_frozen_spec(spec)
        if spec["artifact_sha"] != FROZEN_STRUCTURE_SHA:
            raise PipelineBlocked(
                "structure_sha_mismatch", spec.get("artifact_sha"))
        name = "creative:structure_spec"
        if self.workspace.effective_status(name) == "committed":
            if self.workspace.read_artifact(name) != spec:
                raise PipelineBlocked("structure_workspace_conflict")
            return artifact_ref(name, str(self.workspace.get_sha(name)))
        return self._write_committed(
            name,
            spec,
            dependencies=[],
            derived_from=[artifact_ref(spec["spec_id"], spec["artifact_sha"])],
            producer={
                "skill_id": "r2_c5.freeze",
                "skill_version": "1.0.0",
                "package_sha": FROZEN_STRUCTURE_SHA,
                "prompt_or_instruction_sha": json_hash({
                    "contract": "creative_structure_spec_v1"}),
            },
        )

    def _execute_skill(self, name: str, **kwargs: Any) -> tuple[SkillSpec, dict]:
        spec = self.registry.get(name)
        calls = self.call_counts.get(name, 0)
        self.registry.validate_action(
            {"skill": name},
            self.workspace,
            budget="cheap_text",
            allowed_skill_names={name},
            allowed_permission_profiles={"payload_only"},
            calls_made=calls,
        )
        result = self.registry.execute({"skill": name}, self.workspace, **kwargs)
        self.call_counts[name] = calls + 1
        if not isinstance(result, dict) or set(result) != {"candidates"} \
                or not isinstance(result["candidates"], list) \
                or not result["candidates"]:
            raise PipelineBlocked("skill_output_batch_invalid", name)
        return spec, result

    def _materialize_stage(
            self, *, stage: str, skill: SkillSpec,
            raw_candidates: list[dict[str, Any]],
            candidate_type: str, candidate_schema: str,
            validator_id: str,
            validator: Callable[[dict[str, Any]], None],
            parent_refs: list[dict[str, str]],
            dependency_names: list[str]) -> dict[str, Any]:
        producer = self._producer(skill)
        records: list[CandidateRecord] = []
        for payload in raw_candidates:
            validator(payload)
            candidate_id = str(payload.get(
                "theme_id" if stage == "theme" else "blueprint_id"))
            name = f"creative:{candidate_type}:{candidate_id}"
            committed = self._write_envelope(
                name,
                artifact_type=candidate_type,
                payload_schema_version=candidate_schema,
                payload=payload,
                dependencies=dependency_names,
                parent_refs=parent_refs,
                producer=producer,
            )
            records.append(CandidateRecord(
                candidate_id=candidate_id,
                artifact_id=name,
                artifact_type=candidate_type,
                artifact_sha=committed["sha"],
                parent_shas=tuple(row["sha"] for row in parent_refs),
                status="eligible",
            ))

        validation_payload = build_validation_report(
            stage=stage, validator_id=validator_id, candidates=records)
        candidate_dependencies = [row.artifact_id for row in records]
        candidate_refs = [artifact_ref(row.artifact_id, row.artifact_sha)
                          for row in records]
        validation_name = f"creative:{stage}_validation"
        validation_ref = self._write_envelope(
            validation_name,
            artifact_type="validation_report",
            payload_schema_version="creative_validation_report_v1",
            payload=validation_payload,
            dependencies=candidate_dependencies,
            parent_refs=candidate_refs,
            producer=self._runtime_producer(f"validate_{stage}"),
        )

        pool_payload = CandidatePool(
            pool_id=f"{stage.upper()}_POOL_V1",
            stage=stage,
            candidate_artifact_type=candidate_type,
            parent_refs=tuple(parent_refs),
            candidates=tuple(records),
            selection_policy_version=SELECTION_POLICY_VERSION,
        ).to_dict()
        validate_candidate_pool(pool_payload)
        pool_name = f"creative:{stage}_pool"
        pool_ref = self._write_envelope(
            pool_name,
            artifact_type="candidate_pool",
            payload_schema_version="creative_candidate_pool_v1",
            payload=pool_payload,
            dependencies=[*candidate_dependencies, validation_name],
            parent_refs=[*parent_refs,
                         artifact_ref(validation_name, validation_ref["sha"])],
            producer=self._runtime_producer(f"pool_{stage}"),
        )

        selected = records[0]
        selection_payload = SelectionArtifact(
            selection_id=f"{stage.upper()}_SELECTION_V1",
            pool_ref=artifact_ref(pool_name, pool_ref["sha"]),
            selected_candidate_id=selected.candidate_id,
            selected_candidate_sha=selected.artifact_sha,
            scorecard_shas=(),
            selection_policy_version=SELECTION_POLICY_VERSION,
            decision_origin="deterministic_wave1_fixture",
        ).to_dict(pool_payload)
        validate_selection_artifact(selection_payload, pool_payload)
        selection_name = f"creative:{stage}_selection"
        selection_ref = self._write_envelope(
            selection_name,
            artifact_type="selection",
            payload_schema_version="creative_selection_v1",
            payload=selection_payload,
            dependencies=[pool_name, selected.artifact_id],
            parent_refs=[artifact_ref(pool_name, pool_ref["sha"]),
                         artifact_ref(selected.artifact_id,
                                      selected.artifact_sha)],
            producer=self._runtime_producer(f"select_{stage}"),
        )
        return {
            "pool": pool_payload,
            "pool_ref": pool_ref,
            "selection": selection_payload,
            "selection_ref": selection_ref,
            "selected_record": selected,
        }

    def run_wave1(self, creative_structure_spec: dict[str, Any], *,
                  user_brief: dict[str, Any] | None = None) -> dict[str, Any]:
        structure_workspace_ref = self.bootstrap_structure(
            creative_structure_spec)
        public_structure_ref = artifact_ref(
            creative_structure_spec["spec_id"],
            creative_structure_spec["artifact_sha"],
        )

        theme_skill, theme_batch = self._execute_skill(
            "fake_theme",
            creative_structure_spec=creative_structure_spec,
            user_brief=user_brief or {},
        )
        theme = self._materialize_stage(
            stage="theme",
            skill=theme_skill,
            raw_candidates=theme_batch["candidates"],
            candidate_type="theme_candidate",
            candidate_schema="theme_candidate_v1",
            validator_id="validate_theme_candidate",
            validator=lambda value: validate_theme_candidate(
                value, structure_sha=creative_structure_spec["artifact_sha"]),
            parent_refs=[public_structure_ref],
            dependency_names=["creative:structure_spec"],
        )
        selected_theme_record = theme["selected_record"]
        selected_theme_envelope = self.workspace.read_artifact(
            selected_theme_record.artifact_id)
        selected_theme = selected_theme_envelope["payload"]

        story_skill, story_batch = self._execute_skill(
            "fake_story",
            creative_structure_spec=creative_structure_spec,
            selected_theme=selected_theme,
        )
        story = self._materialize_stage(
            stage="story_blueprint",
            skill=story_skill,
            raw_candidates=story_batch["candidates"],
            candidate_type="story_blueprint",
            candidate_schema="story_blueprint_v1",
            validator_id="validate_story_blueprint",
            validator=lambda value: validate_story_blueprint(
                value,
                structure_sha=creative_structure_spec["artifact_sha"],
                theme_id=selected_theme["theme_id"],
            ),
            parent_refs=[
                public_structure_ref,
                artifact_ref(selected_theme_record.artifact_id,
                             selected_theme_record.artifact_sha),
            ],
            dependency_names=["creative:structure_spec",
                              "creative:theme_selection"],
        )

        self.workspace.record_trace(
            1,
            {"skill": "fake_theme"},
            {"artifact": "creative:theme_pool",
             "sha": theme["pool_ref"]["sha"]},
            {"passed": True, "validators_run": ["validate_theme_candidate"]},
        )
        self.workspace.record_trace(
            2,
            {"skill": "fake_story"},
            {"artifact": "creative:story_blueprint_pool",
             "sha": story["pool_ref"]["sha"]},
            {"passed": True,
             "validators_run": ["validate_story_blueprint"]},
        )
        return {
            "schema_version": "r2_d_wave1_result_v1",
            "status": "PASS",
            "model_calls": 0,
            "structure_public_sha": creative_structure_spec["artifact_sha"],
            "structure_workspace_ref": structure_workspace_ref,
            "theme_selection": theme["selection"],
            "story_blueprint_selection": story["selection"],
            "call_counts": dict(self.call_counts),
            "lineage_report": build_lineage_report(self.workspace),
        }
