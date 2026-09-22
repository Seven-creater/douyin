"""Real model adapters for controlled, single-shot generation acceptance."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from src.agentic_video.creative_pipeline.contracts import (
    ContractError, artifact_ref,
)
from src.agentic_video.creative_pipeline.generation.adapter import (
    validate_shot_contract_schema,
)
from src.agentic_video.creative_pipeline.generation.image import (
    validate_image_candidate_schema,
)
from src.agentic_video.creative_pipeline.generation.video import (
    validate_video_candidate_schema,
)
from src.agentic_video.generation_v9g import build_h3_request
from src.agentic_video.manifest import json_hash


REAL_REQUEST_BASE_FIELDS = {
    "schema_version", "request_id", "media_kind", "candidate_count",
    "shot_contract_ref", "asset_graph_ref", "shot_contract",
}
REAL_VIDEO_EXTRA_FIELDS = {"source_image_ref"}
GENERATED_MEDIA_FIELDS = {
    "schema_version", "media_id", "candidate_id", "media_kind", "path",
    "content_sha", "byte_size", "model_job_id",
}
EXECUTION_FIELDS = {
    "model_id", "model_version", "model_sha", "config_sha", "prompt_sha",
    "seed", "generation_time_ms",
}


class RealImageClient(Protocol):
    def generate(self, *, prompt: str, seed: int, output_path: Path,
                 config: dict[str, Any]) -> dict[str, Any]:
        ...


class RealVideoClient(Protocol):
    def generate(self, request: dict[str, Any],
                 output_path: Path) -> dict[str, Any]:
        ...


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validate_ref(value: object, reason: str) -> None:
    try:
        if not isinstance(value, dict):
            raise ContractError(reason)
        artifact_ref(value.get("artifact_id"), value.get("sha"))
    except ContractError:
        raise ContractError(reason) from None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_real_generation_request(
        *, media_kind: str, shot_contract: dict[str, Any],
        shot_contract_ref: dict[str, str], asset_graph_ref: dict[str, str],
        candidate_count: int = 3,
        source_image_ref: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a one-shot request without changing the frozen shot contract."""
    value = {
        "schema_version": "real_generation_request_v1",
        "request_id": f"REAL_{media_kind.upper()}_{shot_contract['shot_id']}",
        "media_kind": media_kind,
        "candidate_count": candidate_count,
        "shot_contract_ref": dict(shot_contract_ref),
        "asset_graph_ref": dict(asset_graph_ref),
        "shot_contract": deepcopy(shot_contract),
    }
    if media_kind == "video":
        value["source_image_ref"] = dict(source_image_ref or {})
    validate_real_generation_request(value)
    return value


def validate_real_generation_request(value: dict[str, Any]) -> None:
    _require(isinstance(value, dict), "real_generation_request_invalid")
    kind = value.get("media_kind")
    expected = (REAL_REQUEST_BASE_FIELDS | REAL_VIDEO_EXTRA_FIELDS
                if kind == "video" else REAL_REQUEST_BASE_FIELDS)
    _require(kind in {"image", "video"} and set(value) == expected,
             "real_generation_request_keys_invalid")
    _require(value["schema_version"] == "real_generation_request_v1",
             "real_generation_request_schema_invalid")
    _require(isinstance(value["request_id"], str)
             and bool(value["request_id"].strip()),
             "real_generation_request_id_invalid")
    _require(isinstance(value["candidate_count"], int)
             and value["candidate_count"] == 3,
             "real_generation_candidate_count_invalid")
    _validate_ref(value["shot_contract_ref"],
                  "real_generation_shot_ref_invalid")
    _validate_ref(value["asset_graph_ref"],
                  "real_generation_asset_ref_invalid")
    validate_shot_contract_schema(value["shot_contract"])
    _require(value["shot_contract_ref"]["artifact_id"]
             == ("creative:shot_contract:"
                 f"{value['shot_contract']['shot_id']}"),
             "real_generation_shot_artifact_invalid")
    _require(value["asset_graph_ref"]["sha"]
             == value["shot_contract"]["asset_graph_sha"],
             "real_generation_asset_sha_mismatch")
    if kind == "video":
        _validate_ref(value["source_image_ref"],
                      "real_generation_source_ref_invalid")


def real_request_parent_refs(value: dict[str, Any]) -> list[dict[str, str]]:
    validate_real_generation_request(value)
    refs = [dict(value["shot_contract_ref"]), dict(value["asset_graph_ref"])]
    if value["media_kind"] == "video":
        refs.append(dict(value["source_image_ref"]))
    return refs


def validate_generated_media(value: dict[str, Any], *, verify_file: bool) -> None:
    _require(isinstance(value, dict) and set(value) == GENERATED_MEDIA_FIELDS,
             "generated_media_keys_invalid")
    _require(value["schema_version"] == "generated_media_v1"
             and value["media_kind"] in {"image", "video"},
             "generated_media_schema_invalid")
    for key in ("media_id", "candidate_id", "path", "model_job_id"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 f"generated_media_{key}_invalid")
    _require(_sha(value["content_sha"]), "generated_media_sha_invalid")
    _require(isinstance(value["byte_size"], int) and value["byte_size"] > 0,
             "generated_media_size_invalid")
    if verify_file:
        path = Path(value["path"])
        _require(path.is_file(), "generated_media_file_missing")
        _require(path.stat().st_size == value["byte_size"],
                 "generated_media_size_mismatch")
        _require(file_sha256(path) == value["content_sha"],
                 "generated_media_sha_mismatch")


def validate_real_candidate(candidate: dict[str, Any],
                            request: dict[str, Any]) -> None:
    validate_real_generation_request(request)
    kind = request["media_kind"]
    if kind == "image":
        validate_image_candidate_schema(candidate)
    else:
        validate_video_candidate_schema(candidate)
    contract = request["shot_contract"]
    _require(candidate["media_kind"] == kind,
             "real_candidate_media_kind_mismatch")
    _require(candidate["request_sha"] == json_hash(request)
             and candidate["shot_id"] == contract["shot_id"]
             and candidate["shot_contract_sha"]
             == request["shot_contract_ref"]["sha"],
             "real_candidate_contract_mismatch")
    _require(candidate["asset_graph_sha"]
             == request["asset_graph_ref"]["sha"]
             and candidate["asset_refs"] == contract["asset_refs"],
             "real_candidate_asset_mismatch")
    _require(candidate["parent_refs"] == real_request_parent_refs(request),
             "real_candidate_parent_mismatch")
    _require(candidate["structure_trace"] == contract["structure_trace"],
             "real_candidate_trace_mismatch")
    if kind == "video":
        _require(candidate["source_image_ref"] == request["source_image_ref"],
                 "real_candidate_source_mismatch")
        _require(contract["duration_budget_s"][0]
                 <= candidate["duration_s"]
                 <= contract["duration_budget_s"][1],
                 "real_candidate_duration_out_of_contract")


def _asset_descriptions(shot_contract: dict[str, Any],
                        asset_graph: dict[str, Any]) -> list[str]:
    by_id = {row["asset_id"]: row for row in asset_graph["assets"]}
    descriptions = []
    for ref in shot_contract["asset_refs"]:
        asset = by_id.get(ref["asset_id"])
        _require(asset is not None, "real_adapter_asset_missing")
        _require(json_hash(asset) == ref["asset_sha"],
                 "real_adapter_asset_sha_mismatch")
        descriptions.append(
            f"{asset['type']} {asset['asset_id']}: {asset['description']}")
    return descriptions


def compile_real_prompt(shot_contract: dict[str, Any],
                        asset_graph: dict[str, Any]) -> str:
    """Compile only committed contract content; do not invent story roles."""
    assets = "; ".join(_asset_descriptions(shot_contract, asset_graph))
    camera = shot_contract["camera"]
    return (
        f"Create one self-contained vertical cinematic shot. "
        f"Visible action: {'; '.join(shot_contract['actions'])}. "
        f"Frame: {shot_contract['frame_description']}. "
        f"Environment: {shot_contract['environment']}. "
        f"Canonical assets: {assets}. "
        f"Camera: {camera.get('shot_size', 'medium')}, "
        f"{camera.get('angle', 'eye_level')}, "
        f"{camera.get('movement', 'static')}. "
        f"Lighting: {shot_contract['lighting']}. "
        f"Composition: {shot_contract['composition']}. "
        "Preserve canonical identities and show only the contracted action."
    )


class RealImageAdapter:
    """Injectable real image adapter; provider choice remains outside contracts."""

    adapter_version = "1.0.0"

    def __init__(self, client: RealImageClient, *, model_id: str,
                 model_version: str, model_sha: str,
                 seeds: tuple[int, int, int] = (101, 102, 103),
                 config: dict[str, Any] | None = None) -> None:
        _require(bool(model_id.strip()) and bool(model_version.strip()),
                 "real_image_model_identity_invalid")
        _require(_sha(model_sha), "real_image_model_sha_invalid")
        self.client = client
        self.model_id = model_id
        self.model_version = model_version
        self.model_sha = model_sha
        self.seeds = seeds
        self.config = deepcopy(config or {})

    def generate(self, request: dict[str, Any], *, asset_graph: dict[str, Any],
                 output_dir: Path) -> list[dict[str, Any]]:
        validate_real_generation_request(request)
        _require(request["media_kind"] == "image",
                 "real_image_request_kind_invalid")
        _require(len(self.seeds) == request["candidate_count"],
                 "real_image_seed_count_invalid")
        prompt = compile_real_prompt(request["shot_contract"], asset_graph)
        return [self._generate_one(request, prompt, output_dir, index, seed)
                for index, seed in enumerate(self.seeds, 1)]

    def _generate_one(self, request: dict[str, Any], prompt: str,
                      output_dir: Path, index: int,
                      seed: int) -> dict[str, Any]:
        candidate_id = f"RIMG_{request['shot_contract']['shot_id']}_{index:02d}"
        output_path = Path(output_dir) / f"{candidate_id}.png"
        started = time.monotonic()
        raw = self.client.generate(
            prompt=prompt, seed=seed, output_path=output_path,
            config=deepcopy(self.config))
        elapsed = max(0, round((time.monotonic() - started) * 1000))
        return _build_real_result(
            request=request, candidate_id=candidate_id, seed=seed,
            output_path=Path(raw.get("output_path") or output_path),
            model_job_id=str(raw.get("id") or candidate_id),
            model_id=self.model_id, model_version=self.model_version,
            model_sha=self.model_sha, adapter_id="real_image_adapter",
            adapter_version=self.adapter_version, prompt=prompt,
            config=self.config, generation_time_ms=elapsed)


class RealVideoAdapter:
    """Run exactly three candidates for one shot across injected H3 clients."""

    adapter_version = "1.0.0"

    def __init__(self, clients: list[RealVideoClient], *, model_id: str,
                 model_version: str, model_sha: str,
                 aspect_ratio: str = "9:16", inference_steps: int = 50,
                 seeds: tuple[int, int, int] = (201, 202, 203)) -> None:
        _require(bool(model_id.strip()) and bool(model_version.strip()),
                 "real_video_model_identity_invalid")
        _require(_sha(model_sha), "real_video_model_sha_invalid")
        _require(len(clients) == len(seeds) == 3,
                 "real_video_parallelism_invalid")
        self.clients = clients
        self.model_id = model_id
        self.model_version = model_version
        self.model_sha = model_sha
        self.aspect_ratio = aspect_ratio
        self.inference_steps = inference_steps
        self.seeds = seeds

    def generate(self, request: dict[str, Any], *, asset_graph: dict[str, Any],
                 output_dir: Path) -> list[dict[str, Any]]:
        validate_real_generation_request(request)
        _require(request["media_kind"] == "video",
                 "real_video_request_kind_invalid")
        prompt = compile_real_prompt(request["shot_contract"], asset_graph)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(
                self._generate_one, request, prompt, output_dir, index,
                seed, client)
                for index, (seed, client) in enumerate(
                    zip(self.seeds, self.clients), 1)]
            rows = [future.result() for future in futures]
        return sorted(rows, key=lambda row: row["candidate"]["candidate_id"])

    def _generate_one(self, request: dict[str, Any], prompt: str,
                      output_dir: Path, index: int, seed: int,
                      client: RealVideoClient) -> dict[str, Any]:
        contract = request["shot_contract"]
        duration = sum(contract["duration_budget_s"]) / 2
        candidate_id = f"RVID_{contract['shot_id']}_{index:02d}"
        output_path = Path(output_dir) / f"{candidate_id}.mp4"
        config = {
            "aspect_ratio": self.aspect_ratio,
            "duration_s": duration,
            "inference_steps": self.inference_steps,
            "task": "t2va",
        }
        h3_request = build_h3_request(
            prompt=prompt, duration_s=duration, references=None,
            first_frame=None, last_frame=None, capabilities={"t2va": True},
            seed=seed, aspect_ratio=self.aspect_ratio,
            inference_steps=self.inference_steps)
        started = time.monotonic()
        raw = client.generate(h3_request, output_path)
        elapsed = max(0, round((time.monotonic() - started) * 1000))
        return _build_real_result(
            request=request, candidate_id=candidate_id, seed=seed,
            output_path=Path(raw.get("output_path") or output_path),
            model_job_id=str(raw.get("id") or candidate_id),
            model_id=self.model_id, model_version=self.model_version,
            model_sha=self.model_sha, adapter_id="real_video_adapter",
            adapter_version=self.adapter_version, prompt=prompt,
            config=config, generation_time_ms=elapsed)


def _build_real_result(
        *, request: dict[str, Any], candidate_id: str, seed: int,
        output_path: Path, model_job_id: str, model_id: str,
        model_version: str, model_sha: str, adapter_id: str,
        adapter_version: str, prompt: str, config: dict[str, Any],
        generation_time_ms: int) -> dict[str, Any]:
    _require(output_path.is_file(), "real_adapter_output_missing")
    content_sha = file_sha256(output_path)
    kind = request["media_kind"]
    contract = request["shot_contract"]
    parent_refs = real_request_parent_refs(request)
    candidate = {
        "schema_version": f"{kind}_candidate_v1",
        "candidate_id": candidate_id,
        "media_kind": kind,
        "shot_id": contract["shot_id"],
        "request_sha": json_hash(request),
        "shot_contract_sha": request["shot_contract_ref"]["sha"],
        "asset_graph_sha": request["asset_graph_ref"]["sha"],
        "parent_refs": parent_refs,
        "asset_refs": deepcopy(contract["asset_refs"]),
        "backend": {
            "adapter_id": adapter_id, "adapter_version": adapter_version,
            "model_id": model_id, "seed": seed,
        },
        "placeholder": {
            "media_type": kind,
            "materialization": (
                f"artifact:creative:generated_media:{candidate_id}"),
            "generated": True, "content_sha": content_sha,
        },
        "structure_trace": deepcopy(contract["structure_trace"]),
    }
    if kind == "video":
        candidate["source_image_ref"] = dict(request["source_image_ref"])
        candidate["duration_s"] = sum(contract["duration_budget_s"]) / 2
    media = {
        "schema_version": "generated_media_v1",
        "media_id": f"MEDIA_{candidate_id}",
        "candidate_id": candidate_id,
        "media_kind": kind,
        "path": str(output_path.resolve()),
        "content_sha": content_sha,
        "byte_size": output_path.stat().st_size,
        "model_job_id": model_job_id,
    }
    execution = {
        "model_id": model_id, "model_version": model_version,
        "model_sha": model_sha, "config_sha": json_hash(config),
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": seed, "generation_time_ms": generation_time_ms,
    }
    validate_real_candidate(candidate, request)
    validate_generated_media(media, verify_file=True)
    _require(set(execution) == EXECUTION_FIELDS,
             "real_adapter_execution_keys_invalid")
    return {"candidate": candidate, "media": media,
            "execution": execution}
