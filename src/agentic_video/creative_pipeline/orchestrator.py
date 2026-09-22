"""Frozen R2-D stage graph and deterministic contract orchestration."""
from __future__ import annotations

import hashlib
from copy import deepcopy
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
    validate_artifact_envelope,
    validate_candidate_pool,
    validate_selection_artifact,
)
from src.agentic_video.creative_pipeline.evaluation.structure import (
    validate_story_structure,
    validate_theme_structure,
)
from src.agentic_video.creative_pipeline.planning.story import STORY_SELECT_K
from src.agentic_video.creative_pipeline.planning.theme import THEME_SELECT_K
from src.agentic_video.creative_pipeline.production.asset import (
    FakeAssetGraphAdapter,
)
from src.agentic_video.creative_pipeline.production.asset_validator import (
    validate_production_asset_graph,
)
from src.agentic_video.creative_pipeline.production.shot import (
    FakeShotPlanAdapter,
)
from src.agentic_video.creative_pipeline.production.shot_validator import (
    validate_production_shot_plan,
)
from src.agentic_video.creative_pipeline.production.storyboard import (
    FakeStoryboardAdapter, validate_storyboard,
)
from src.agentic_video.creative_pipeline.writing.adapter import (
    build_blind_evaluation_request, build_screenplay_request,
)
from src.agentic_video.creative_pipeline.writing.screenplay import (
    SCREENPLAY_MAX_CANDIDATES, SCREENPLAY_SELECT_K, SCREENPLAY_SELECTION_POLICY,
)
from src.agentic_video.creative_pipeline.writing.validator import (
    validate_format_constraints, validate_screenplay,
)
from src.agentic_video.creative_structure_v1.freeze import validate_frozen_spec
from src.agentic_video.manifest import json_hash
from src.agentic_video.provenance import build_lineage_report
from src.agentic_video.skills.registry import SkillRegistry, SkillSpec
from src.agentic_video.workspace import Workspace


FROZEN_STRUCTURE_SHA = (
    "5a22fb5af29588b6220fbe9ccda751c922abe617fca1716b2f97f2942218a967")
SELECTION_POLICY_VERSION = "wave1_first_valid_v1"
WAVE2_SELECTION_POLICY_VERSION = "wave2_fixture_shortlist_v1"
WAVE4_PRODUCTION_POLICY_VERSION = "wave4_fake_production_v1"

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
    {
        "stage": "storyboard",
        "requires": ["shot_plan", "asset_graph"],
        "produces": "storyboard",
    },
)


class PipelineBlocked(RuntimeError):
    def __init__(self, reason_code: str, detail: object = "") -> None:
        super().__init__(f"{reason_code}:{detail}" if detail else reason_code)
        self.reason_code = reason_code
        self.detail = detail


class CreativePipelineOrchestrator:
    """Execute the authorized model-free creative contract fixtures."""

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
                        producer: dict[str, str],
                        selection_policy_version: str =
                        SELECTION_POLICY_VERSION) -> dict[str, str]:
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
            policy_versions={"selection": selection_policy_version},
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
            dependency_names: list[str],
            selection_count: int = 1,
            selection_policy_version: str = SELECTION_POLICY_VERSION,
            decision_origin: str = "deterministic_wave1_fixture",
            selection_key: Callable[[CandidateRecord], object] | None = None,
            candidate_parent_resolver: Callable[
                [dict[str, Any]], list[dict[str, str]]] | None = None,
            candidate_dependency_resolver: Callable[
                [dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
        producer = self._producer(skill)
        records: list[CandidateRecord] = []
        for payload in raw_candidates:
            validator(payload)
            candidate_parents = (candidate_parent_resolver(payload)
                                 if candidate_parent_resolver else parent_refs)
            candidate_dependencies = (
                candidate_dependency_resolver(payload)
                if candidate_dependency_resolver else dependency_names)
            candidate_id = str(payload[{
                "theme": "theme_id", "story_blueprint": "blueprint_id",
                "screenplay": "screenplay_id",
            }[stage]])
            name = f"creative:{candidate_type}:{candidate_id}"
            committed = self._write_envelope(
                name,
                artifact_type=candidate_type,
                payload_schema_version=candidate_schema,
                payload=payload,
                dependencies=candidate_dependencies,
                parent_refs=candidate_parents,
                producer=producer,
                selection_policy_version=selection_policy_version,
            )
            records.append(CandidateRecord(
                candidate_id=candidate_id,
                artifact_id=name,
                artifact_type=candidate_type,
                artifact_sha=committed["sha"],
                parent_shas=tuple(row["sha"] for row in candidate_parents),
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
            selection_policy_version=selection_policy_version,
        )

        pool_payload = CandidatePool(
            pool_id=f"{stage.upper()}_POOL_V1",
            stage=stage,
            candidate_artifact_type=candidate_type,
            parent_refs=tuple(parent_refs),
            candidates=tuple(records),
            selection_policy_version=selection_policy_version,
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
            selection_policy_version=selection_policy_version,
        )

        if not 1 <= selection_count <= len(records):
            raise PipelineBlocked(
                "selection_count_invalid", f"{stage}:{selection_count}")
        ranked_records = sorted(records, key=selection_key) \
            if selection_key else records
        selections: list[dict[str, Any]] = []
        for index, selected in enumerate(
                ranked_records[:selection_count], start=1):
            suffix = "" if index == 1 else f":{index:02d}"
            id_suffix = "" if index == 1 else f"_{index:02d}"
            selection_payload = SelectionArtifact(
                selection_id=f"{stage.upper()}_SELECTION_V1{id_suffix}",
                pool_ref=artifact_ref(pool_name, pool_ref["sha"]),
                selected_candidate_id=selected.candidate_id,
                selected_candidate_sha=selected.artifact_sha,
                scorecard_shas=(),
                selection_policy_version=selection_policy_version,
                decision_origin=decision_origin,
            ).to_dict(pool_payload)
            validate_selection_artifact(selection_payload, pool_payload)
            selection_name = f"creative:{stage}_selection{suffix}"
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
                selection_policy_version=selection_policy_version,
            )
            selections.append({
                "selection": selection_payload,
                "selection_name": selection_name,
                "selection_ref": selection_ref,
                "selected_record": selected,
            })
        first = selections[0]
        return {
            "pool": pool_payload,
            "pool_ref": pool_ref,
            "selections": selections,
            "selection": first["selection"],
            "selection_ref": first["selection_ref"],
            "selected_record": first["selected_record"],
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
            validator_id="validate_theme_structure",
            validator=lambda value: validate_theme_structure(
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
            validator_id="validate_story_structure",
            validator=lambda value: validate_story_structure(
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

    def run_wave3(self, creative_structure_spec: dict[str, Any], *,
                  format_constraints: dict[str, Any]) -> dict[str, Any]:
        """Continue committed Wave 2 selections with three fake writer calls."""
        validate_frozen_spec(creative_structure_spec)
        validate_format_constraints(format_constraints)

        def read_current(name: str, artifact_type: str | None = None) -> dict:
            if self.workspace.effective_status(name) != "committed":
                raise PipelineBlocked("screenplay_parent_not_committed", name)
            value = self.workspace.read_artifact(name)
            if not isinstance(value, dict) or json_hash(value) != self.workspace.get_sha(name):
                raise PipelineBlocked("screenplay_parent_sha_mismatch", name)
            if artifact_type:
                validate_artifact_envelope(value)
                if value["artifact_type"] != artifact_type or value["artifact_id"] != name:
                    raise PipelineBlocked("screenplay_parent_type_mismatch", name)
            return value

        current_spec = read_current("creative:structure_spec")
        if (current_spec != creative_structure_spec
                or creative_structure_spec["artifact_sha"] != FROZEN_STRUCTURE_SHA):
            raise PipelineBlocked("structure_workspace_conflict")
        public_ref = artifact_ref(current_spec["spec_id"], current_spec["artifact_sha"])
        pool_name = "creative:story_blueprint_pool"
        pool = read_current(pool_name, "candidate_pool")["payload"]
        validate_candidate_pool(pool)
        records = {row["candidate_id"]: row for row in pool["candidates"]}
        requests: dict[str, dict] = {}
        origins: dict[str, list[dict[str, str]]] = {}
        # Resolve and validate every selection before the first writer call.
        for index in range(1, SCREENPLAY_MAX_CANDIDATES + 1):
            selection_name = "creative:story_blueprint_selection" + (
                "" if index == 1 else f":{index:02d}")
            selection = read_current(selection_name, "selection")["payload"]
            validate_selection_artifact(selection, pool)
            if selection["pool_ref"] != artifact_ref(pool_name, self.workspace.get_sha(pool_name)):
                raise PipelineBlocked("screenplay_selection_pool_sha_mismatch")
            record = records[selection["selected_candidate_id"]]
            blueprint_name = record["artifact_id"]
            if (blueprint_name != f"creative:story_blueprint:{record['candidate_id']}"
                    or not all(char.isalnum() or char == "_"
                               for char in record["candidate_id"])):
                raise PipelineBlocked("screenplay_blueprint_artifact_id_invalid")
            envelope = read_current(blueprint_name, "story_blueprint")
            if (record["artifact_sha"] != self.workspace.get_sha(blueprint_name)
                    or record["status"] not in {"eligible", "scored", "selected"}):
                raise PipelineBlocked("screenplay_selected_blueprint_invalid")
            request = build_screenplay_request(
                creative_structure_spec=current_spec, blueprint_envelope=envelope,
                blueprint_sha=record["artifact_sha"], format_constraints=format_constraints)
            blueprint_id = request["story_blueprint"]["blueprint_id"]
            if blueprint_id in requests or blueprint_id != record["candidate_id"]:
                raise PipelineBlocked("screenplay_blueprint_selection_duplicate_or_mismatch")
            requests[blueprint_id] = request
            origins[blueprint_id] = [public_ref,
                artifact_ref(blueprint_name, record["artifact_sha"]),
                artifact_ref(selection_name, self.workspace.get_sha(selection_name))]

        constraints_name = "creative:screenplay_format_constraints"
        constraints_ref = self._write_envelope(
            constraints_name, artifact_type="format_constraints",
            payload_schema_version="screenplay_format_constraints_v1",
            payload=deepcopy(format_constraints), dependencies=[], parent_refs=[],
            producer=self._runtime_producer("screenplay_format_constraints"),
            selection_policy_version=SCREENPLAY_SELECTION_POLICY)
        for parents in origins.values():
            parents.append(artifact_ref(constraints_name, constraints_ref["sha"]))

        candidates = []
        for index, (blueprint_id, request) in enumerate(requests.items(), 1):
            skill, batch = self._execute_skill("fake_screenplay", request=deepcopy(request))
            try:
                if len(batch["candidates"]) != 1:
                    raise ContractError("screenplay_batch_count_invalid")
                candidate = batch["candidates"][0]
                validate_screenplay(
                    candidate, blueprint=request["story_blueprint"],
                    blueprint_sha=request["story_blueprint_sha"],
                    structure_sha=current_spec["artifact_sha"],
                    format_constraints=format_constraints)
                blind_request = build_blind_evaluation_request(
                    candidate, evaluation_id=f"EVAL_{index:02d}", writer_request=request)
            except ContractError as exc:
                self.workspace.record_trace(index + 2,
                    {"skill": skill.name, "request_sha": json_hash(request)},
                    {"raw_response": batch},
                    {"passed": False, "reason_code": exc.reason_code, "retry_count": 0})
                raise
            candidates.append(candidate)
            self.workspace.record_trace(index + 2,
                {"skill": skill.name, "producer": self._producer(skill),
                 "request": request, "request_sha": json_hash(request),
                 "parent_refs": origins[blueprint_id]},
                {"raw_response": batch, "blind_evaluation_request": blind_request},
                {"passed": True, "validators_run": ["validate_screenplay"],
                 "quality_evaluation": "not_run", "model_calls": 0, "retry_count": 0})

        def validate_candidate(candidate: dict) -> None:
            request = requests.get(candidate.get("blueprint_id"))
            if request is None:
                raise ContractError("screenplay_blueprint_not_selected")
            validate_screenplay(candidate, blueprint=request["story_blueprint"],
                blueprint_sha=request["story_blueprint_sha"],
                structure_sha=current_spec["artifact_sha"],
                format_constraints=format_constraints)

        all_parents = []
        for parents in origins.values():
            for parent in parents:
                if parent not in all_parents:
                    all_parents.append(parent)
        stage = self._materialize_stage(
            stage="screenplay", skill=skill, raw_candidates=candidates,
            candidate_type="screenplay", candidate_schema="screenplay_v1",
            validator_id="validate_screenplay", validator=validate_candidate,
            parent_refs=all_parents, dependency_names=[],
            selection_count=SCREENPLAY_SELECT_K,
            selection_policy_version=SCREENPLAY_SELECTION_POLICY,
            decision_origin="deterministic_wave3_fixture",
            candidate_parent_resolver=lambda value: origins[value["blueprint_id"]],
            candidate_dependency_resolver=lambda value: [
                "creative:structure_spec",
                *[row["artifact_id"] for row in origins[value["blueprint_id"]][1:]]])
        return {
            "schema_version": "r2_d_wave3_result_v1", "status": "PASS",
            "fixture_only": True, "model_calls": 0,
            "quality_evaluation": "not_run", "semantic_entailment": "not_run",
            "screenplay_pool_ref": stage["pool_ref"],
            "screenplay_selection": stage["selection"],
            "committed_screenplay_ref": artifact_ref(
                stage["selected_record"].artifact_id, stage["selected_record"].artifact_sha),
            "call_counts": dict(self.call_counts),
            "lineage_report": build_lineage_report(self.workspace),
        }

    def run_wave4(self, creative_structure_spec: dict[str, Any]) -> dict[str, Any]:
        """Compile the selected screenplay into text-only production artifacts."""
        validate_frozen_spec(creative_structure_spec)

        def read_current(name: str, artifact_type: str | None = None) -> dict:
            if self.workspace.effective_status(name) != "committed":
                raise PipelineBlocked("production_parent_not_committed", name)
            value = self.workspace.read_artifact(name)
            if (not isinstance(value, dict)
                    or json_hash(value) != self.workspace.get_sha(name)):
                raise PipelineBlocked("production_parent_sha_mismatch", name)
            if artifact_type:
                validate_artifact_envelope(value)
                if (value["artifact_type"] != artifact_type
                        or value["artifact_id"] != name):
                    raise PipelineBlocked("production_parent_type_mismatch", name)
            return value

        current_spec = read_current("creative:structure_spec")
        if (current_spec != creative_structure_spec
                or creative_structure_spec["artifact_sha"]
                != FROZEN_STRUCTURE_SHA):
            raise PipelineBlocked("structure_workspace_conflict")
        pool_name = "creative:screenplay_pool"
        pool = read_current(pool_name, "candidate_pool")["payload"]
        validate_candidate_pool(pool)
        selection_name = "creative:screenplay_selection"
        selection = read_current(selection_name, "selection")["payload"]
        validate_selection_artifact(selection, pool)
        if selection["pool_ref"] != artifact_ref(
                pool_name, str(self.workspace.get_sha(pool_name))):
            raise PipelineBlocked("production_selection_pool_sha_mismatch")
        record = next(
            row for row in pool["candidates"]
            if row["candidate_id"] == selection["selected_candidate_id"])
        screenplay_name = record["artifact_id"]
        if (record["status"] not in {"eligible", "scored", "selected"}
                or screenplay_name
                != f"creative:screenplay:{record['candidate_id']}"):
            raise PipelineBlocked("production_selected_screenplay_invalid")
        screenplay_envelope = read_current(screenplay_name, "screenplay")
        if record["artifact_sha"] != self.workspace.get_sha(screenplay_name):
            raise PipelineBlocked("production_screenplay_sha_mismatch")
        screenplay = screenplay_envelope["payload"]

        blueprint_id = screenplay.get("blueprint_id")
        if (not isinstance(blueprint_id, str) or not blueprint_id
                or not all(char.isalnum() or char == "_"
                           for char in blueprint_id)):
            raise PipelineBlocked("production_blueprint_id_invalid")
        blueprint_name = f"creative:story_blueprint:{blueprint_id}"
        blueprint_envelope = read_current(blueprint_name, "story_blueprint")
        constraints_envelope = read_current(
            "creative:screenplay_format_constraints", "format_constraints")
        validate_screenplay(
            screenplay, blueprint=blueprint_envelope["payload"],
            blueprint_sha=str(self.workspace.get_sha(blueprint_name)),
            structure_sha=current_spec["artifact_sha"],
            format_constraints=constraints_envelope["payload"])

        screenplay_ref = artifact_ref(screenplay_name, record["artifact_sha"])
        screenplay_selection_ref = artifact_ref(
            selection_name, str(self.workspace.get_sha(selection_name)))
        asset_graph = FakeAssetGraphAdapter().run(
            screenplay, screenplay_sha=record["artifact_sha"])
        validate_production_asset_graph(
            asset_graph, screenplay=screenplay,
            screenplay_sha=record["artifact_sha"])
        asset_name = "creative:asset_graph"
        asset_ref = self._write_envelope(
            asset_name, artifact_type="asset_graph",
            payload_schema_version="asset_graph_v1", payload=asset_graph,
            dependencies=[screenplay_name, selection_name],
            parent_refs=[screenplay_ref, screenplay_selection_ref],
            producer=self._runtime_producer("fake_asset_graph_adapter"),
            selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)

        shot_plan = FakeShotPlanAdapter().run(
            screenplay, asset_graph, screenplay_sha=record["artifact_sha"],
            asset_graph_sha=asset_ref["sha"])
        validate_production_shot_plan(
            shot_plan, screenplay=screenplay,
            screenplay_sha=record["artifact_sha"], asset_graph=asset_graph,
            asset_graph_sha=asset_ref["sha"])
        shot_name = "creative:shot_plan"
        shot_ref = self._write_envelope(
            shot_name, artifact_type="shot_plan",
            payload_schema_version="shot_plan_v1", payload=shot_plan,
            dependencies=[screenplay_name, asset_name],
            parent_refs=[screenplay_ref,
                         artifact_ref(asset_name, asset_ref["sha"])],
            producer=self._runtime_producer("fake_shot_plan_adapter"),
            selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)

        storyboard = FakeStoryboardAdapter().run(
            shot_plan, shot_plan_sha=shot_ref["sha"],
            asset_graph_sha=asset_ref["sha"])
        validate_storyboard(
            storyboard, screenplay=screenplay,
            screenplay_sha=record["artifact_sha"], asset_graph=asset_graph,
            asset_graph_sha=asset_ref["sha"], shot_plan=shot_plan,
            shot_plan_sha=shot_ref["sha"])
        storyboard_name = "creative:storyboard"
        storyboard_ref = self._write_envelope(
            storyboard_name, artifact_type="storyboard",
            payload_schema_version="storyboard_v1", payload=storyboard,
            dependencies=[shot_name, asset_name],
            parent_refs=[artifact_ref(shot_name, shot_ref["sha"]),
                         artifact_ref(asset_name, asset_ref["sha"])],
            producer=self._runtime_producer("fake_storyboard_adapter"),
            selection_policy_version=WAVE4_PRODUCTION_POLICY_VERSION)

        for step, (adapter, name, ref, validator) in enumerate((
                ("fake_asset_graph_adapter", asset_name, asset_ref,
                 "validate_production_asset_graph"),
                ("fake_shot_plan_adapter", shot_name, shot_ref,
                 "validate_production_shot_plan"),
                ("fake_storyboard_adapter", storyboard_name, storyboard_ref,
                 "validate_storyboard")), start=6):
            self.workspace.record_trace(
                step, {"adapter": adapter},
                {"artifact": name, "sha": ref["sha"]},
                {"passed": True, "validators_run": [validator],
                 "model_calls": 0, "media_generated": False})
        return {
            "schema_version": "r2_d_wave4_result_v1", "status": "PASS",
            "fixture_only": True, "model_calls": 0,
            "media_generated": False,
            "asset_graph_ref": artifact_ref(asset_name, asset_ref["sha"]),
            "shot_plan_ref": artifact_ref(shot_name, shot_ref["sha"]),
            "storyboard_ref": artifact_ref(
                storyboard_name, storyboard_ref["sha"]),
            "lineage_report": build_lineage_report(self.workspace),
        }

    def run_wave2(self, creative_structure_spec: dict[str, Any], *,
                  user_brief: dict[str, Any] | None = None) -> dict[str, Any]:
        """Materialize the frozen 20-to-5-to-25-to-3 planning fixture."""
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
            validator_id="validate_theme_structure",
            validator=lambda value: validate_theme_structure(
                value, structure_sha=creative_structure_spec["artifact_sha"]),
            parent_refs=[public_structure_ref],
            dependency_names=["creative:structure_spec"],
            selection_count=THEME_SELECT_K,
            selection_policy_version=WAVE2_SELECTION_POLICY_VERSION,
            decision_origin="deterministic_wave2_fixture",
        )

        theme_origins: dict[str, dict[str, Any]] = {}
        story_skill: SkillSpec | None = None
        story_candidates: list[dict[str, Any]] = []
        for selected in theme["selections"]:
            record = selected["selected_record"]
            envelope = self.workspace.read_artifact(record.artifact_id)
            payload = envelope["payload"]
            theme_origins[payload["theme_id"]] = selected
            story_skill, story_batch = self._execute_skill(
                "fake_story",
                creative_structure_spec=creative_structure_spec,
                selected_theme=payload,
            )
            story_candidates.extend(story_batch["candidates"])
        if story_skill is None:
            raise PipelineBlocked("theme_selection_empty")

        def story_parent_refs(payload: dict[str, Any]
                              ) -> list[dict[str, str]]:
            origin = theme_origins[payload["theme_id"]]
            record = origin["selected_record"]
            return [
                public_structure_ref,
                artifact_ref(record.artifact_id, record.artifact_sha),
                artifact_ref(origin["selection_name"],
                             origin["selection_ref"]["sha"]),
            ]

        def story_dependencies(payload: dict[str, Any]) -> list[str]:
            origin = theme_origins[payload["theme_id"]]
            return [
                "creative:structure_spec",
                origin["selected_record"].artifact_id,
                origin["selection_name"],
            ]

        def validate_selected_story(payload: dict[str, Any]) -> None:
            theme_id = payload.get("theme_id")
            if theme_id not in theme_origins:
                raise ContractError("story_theme_not_selected", theme_id)
            validate_story_structure(
                payload,
                structure_sha=creative_structure_spec["artifact_sha"],
                theme_id=theme_id,
            )

        story_pool_parents = [public_structure_ref]
        for selected in theme["selections"]:
            story_pool_parents.extend([
                artifact_ref(selected["selected_record"].artifact_id,
                             selected["selected_record"].artifact_sha),
                artifact_ref(selected["selection_name"],
                             selected["selection_ref"]["sha"]),
            ])
        story = self._materialize_stage(
            stage="story_blueprint",
            skill=story_skill,
            raw_candidates=story_candidates,
            candidate_type="story_blueprint",
            candidate_schema="story_blueprint_v1",
            validator_id="validate_story_structure",
            validator=validate_selected_story,
            parent_refs=story_pool_parents,
            dependency_names=["creative:structure_spec"],
            selection_count=STORY_SELECT_K,
            selection_policy_version=WAVE2_SELECTION_POLICY_VERSION,
            decision_origin="deterministic_wave2_fixture",
            selection_key=lambda record: (
                int(record.candidate_id.rsplit("_", 1)[-1]),
                record.candidate_id,
            ),
            candidate_parent_resolver=story_parent_refs,
            candidate_dependency_resolver=story_dependencies,
        )

        self.workspace.record_trace(
            1,
            {"skill": "fake_theme"},
            {"artifact": "creative:theme_pool",
             "sha": theme["pool_ref"]["sha"]},
            {"passed": True,
             "validators_run": ["validate_theme_structure"]},
        )
        self.workspace.record_trace(
            2,
            {"skill": "fake_story", "calls": THEME_SELECT_K},
            {"artifact": "creative:story_blueprint_pool",
             "sha": story["pool_ref"]["sha"]},
            {"passed": True,
             "validators_run": ["validate_story_structure"]},
        )
        return {
            "schema_version": "r2_d_wave2_result_v1",
            "status": "PASS",
            "model_calls": 0,
            "structure_public_sha": creative_structure_spec["artifact_sha"],
            "structure_workspace_ref": structure_workspace_ref,
            "theme_selections": [row["selection"]
                                 for row in theme["selections"]],
            "story_blueprint_selections": [row["selection"]
                                             for row in story["selections"]],
            "call_counts": dict(self.call_counts),
            "lineage_report": build_lineage_report(self.workspace),
        }
