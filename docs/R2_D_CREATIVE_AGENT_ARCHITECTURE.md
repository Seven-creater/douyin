# R2-D Reference-driven Creative Agent Architecture Freeze

Status: `DESIGN FREEZE — IMPLEMENTATION NOT STARTED`

Date: 2026-09-22
Upstream freeze: `R2-C.5`
Creative Structure Spec: `CREATIVE-STRUCTURE-V1`
Creative Structure Spec SHA: `5a22fb5af29588b6220fbe9ccda751c922abe617fca1716b2f97f2942218a967`

This document freezes the target architecture from the published Creative
Structure Spec through final video evaluation. It is an implementation
contract, not authorization to run models, install third-party skills, or
start media generation.

## 1. Scope and decisions

### 1.1 Goal

Build a reference-driven creative system in which the reference is used only
to publish a small transferable control artifact. Creative skills receive that
artifact and a user brief, but never receive the reference video, reference
claims, reference narrative, extraction traces, or audit annex.

The frozen top-level flow is:

```text
Reference Zone                              Creative/Production Zones

reference media
  -> verified reference representation
  -> creative_structure_spec_v1  ---------> theme candidate pool
                                            -> selected theme
                                            -> story-blueprint candidate pool
                                            -> selected story blueprint
                                            -> screenplay candidate pool
                                            -> selected screenplay
                                            -> asset graph and visual bibles
                                            -> shot plan
                                            -> storyboard / shot contracts
                                            -> image and video candidates
                                            -> assembly
                                            -> evaluation and bounded repair
                                            -> approved master
```

The explicit Theme -> Story Blueprint -> Screenplay separation follows the
planning-before-surface-generation principle demonstrated by Plan-and-Write.
It is not evidence that this project's implementation will work without its
own evaluation.

### 1.2 Frozen architectural decisions

1. `creative_structure_spec_v1` is the only reference-derived artifact allowed
   to cross into creative generation.
2. Creative planning is mandatory. No supported path goes directly from the
   structure spec to a screenplay.
3. Every stage writes immutable, schema-versioned artifacts before the next
   stage runs.
4. Best-of-N is a common candidate-pool mechanism, not a separate late-stage
   subsystem.
5. Deterministic validation runs before model-based judging.
6. Candidate generation, candidate evaluation, and candidate selection are
   separate activities with separate traces.
7. External skills are optional adapters. The core pipeline cannot depend on
   a specific GitHub repository or on prose-only output.
8. Existing runtime infrastructure is extended, not duplicated.
9. Repair targets the smallest invalid artifact and has a fixed attempt and
   cost budget.
10. Expensive image/video generation cannot start from an uncommitted
    screenplay, asset graph, or shot plan.

### 1.3 Non-goals

R2-D does not redesign reference understanding, reopen Creative Structure
minimality, define a universal theory of stories, or require every story to
contain conflict, competition, capability proof, social prejudice, a reversal,
or a three-act structure. It does not install `screenwriting-skills`, choose a
new image/video model, or execute Best-of-N during architecture freeze.

## 2. Reuse map: current repository versus new work

The repository already has production infrastructure that the conceptual
proposal did not account for. R2-D must reuse it.

| Concern | Existing implementation | R2-D decision |
|---|---|---|
| Skill metadata and runnable checks | `src/agentic_video/skills/registry.py` (`SkillSpec`, `SkillRegistry`) | Extend additively with schema, permission, version, and call-budget metadata. Do not create a second registry. |
| Agent routing/execution loop | `src/agentic_video/agent_v4.py` | Retain as the execution harness. Creative orchestration supplies an allowlisted stage graph; the model may not invent stages. |
| Immutable artifact versions and dependency invalidation | `src/agentic_video/workspace.py` | Reuse. Candidate selection remains a separate artifact rather than adding ad-hoc workspace status values. |
| SHA lineage, approvals, revocation | `src/agentic_video/provenance.py` | Reuse and extend parent bindings to candidate pools, selections, skill package hashes, and policy versions. |
| Existing screenplay prompt | `src/agentic_video/skills/__init__.py` | Do not reuse as-is: it reads old Creative DNA and violates the new reference boundary. Runtime code may be reused; the prompt contract may not. |
| Existing reference-oriented story planner | `src/agentic_video/story_planner.py` | Keep as legacy. Its fixed arc-role assumptions are not the R2-D Story Blueprint contract. |
| Asset graph | `src/agentic_video/skills/asset_schema.py` | Reuse `asset_graph_v1` and its deterministic completeness checks. |
| Asset generation and acceptance | `src/agentic_video/asset_studio/` | Reuse through an adapter after screenplay and asset graph commitment. |
| Shot plan and storyboard | `src/agentic_video/storyboard/` | Reuse `shot_plan_v1`, validators, and backends. Shot planning requires both screenplay and asset graph. |
| Video generation and observation/repair | `src/agentic_video/generation_v9g.py` and later production modules | Reuse backend capabilities only behind a new shot-contract adapter and current-source preflight. Do not expose reference inputs. |
| Frozen control artifact | `experiments/creative_structure_minimality/r2_c5/creative_structure_spec_v1.json` | This exact SHA is the R2-D upstream root. |

The proposed new namespace is deliberately small:

```text
src/agentic_video/creative_pipeline/
  __init__.py
  contracts.py              # common envelopes and candidate-pool schemas
  orchestrator.py           # frozen stage DAG and budget enforcement
  planning/
    theme.py                 # theme_candidate_v1 producer contract
    story.py                 # story_blueprint_v1 producer contract
  writing/
    screenplay_adapter.py    # internal/external writer adapter
  evaluation/
    structure.py             # deterministic bindings and semantic audit input
    ranking.py               # scorecard and selection policy
    feasibility.py           # production cost/asset feasibility
  adapters/
    legacy_assets.py
    legacy_storyboard.py
    legacy_generation.py
```

No `creative_agent/skills/registry.py`, new workspace, or second provenance
engine is authorized.

## 3. Skill Runtime

### 3.1 Skill definition

A runtime skill is an executable, versioned capability with a typed contract.
An Agent Skills `SKILL.md` package is one possible source of procedural
instructions; it is not itself trusted executable authority.

`SkillSpec` will be extended, or wrapped compatibly, with:

```text
skill_id
skill_version
input_artifact_types[]
input_schema_versions[]
output_artifact_type
output_schema_version
entrypoint
validator_ids[]
permission_profile
cost_class
max_calls
max_repair_attempts
deterministic
idempotent
destructive
package_sha
prompt_or_instruction_sha
```

The existing `inputs`, `outputs`, `preconditions`, `validators`, `cost_class`,
`destructive`, and `idempotent` fields remain compatible.

### 3.2 Runtime control flow

```text
stage graph authorizes skill
  -> registry resolves an exact version
  -> input envelope and parent SHAs validated
  -> permission profile enforced
  -> request payload built from an allowlist
  -> skill executes once
  -> raw response/trace stored outside public artifact
  -> output schema validated
  -> deterministic gates run
  -> artifact written as draft
  -> evaluation artifacts written
  -> selection/commit or bounded repair
```

The model may choose only among skills the deterministic stage graph marks as
runnable. It cannot change budgets, permissions, parent SHAs, validators, or
the next artifact type.

### 3.3 External skill adapters

An adapter translates the system artifact into a skill-specific request and
translates the response back into the frozen output schema. The adapter, not
the external skill, owns:

- input field allowlisting;
- output parsing and schema validation;
- stable-ID assignment;
- trace capture;
- timeout/call budget;
- permission enforcement;
- removal of prose or fields outside the contract.

The first screenwriting adapter may target `screenwriting-skills`, but R2-D
must also support a fake deterministic adapter and an internal-model adapter.
The pipeline is therefore testable without installing the GitHub project.

## 4. Artifact Protocol

### 4.1 Common artifact envelope

Every creative and production artifact is stored immutably with this logical
envelope. `Workspace` may store provenance fields alongside the payload rather
than duplicating them inside every payload file.

```json
{
  "artifact_id": "...",
  "artifact_type": "story_blueprint",
  "schema_version": "story_blueprint_v1",
  "version": "v1",
  "content_sha": "...",
  "status": "draft",
  "producer": {
    "skill_id": "...",
    "skill_version": "...",
    "package_sha": "...",
    "prompt_or_instruction_sha": "...",
    "run_id": "..."
  },
  "derived_from": [
    {"artifact_id": "...", "sha": "..."}
  ],
  "policy_versions": {
    "validation": "...",
    "selection": "..."
  },
  "payload": {}
}
```

Mutable aliases such as `latest` are UI conveniences only. Dependencies and
approvals bind exact SHAs.

### 4.2 Core schemas

#### `theme_candidate_v1`

```text
theme_id
parent_structure_sha
domain
premise
audience_promise
tone
binding_slots
structure_bindings
```

`structure_bindings` must explicitly bind all four frozen IDs:

```text
I0_PRIOR_INTERPRETATION
E1_NEW_INFORMATION
I1_UPDATED_INTERPRETATION
R1_INFORMATION_UPDATE
```

The relation binding names the three element bindings; it may not merely state
that an update exists.

#### `story_blueprint_v1`

```text
blueprint_id
theme_id
logline
characters[]
setting
goal
stakes
events[]
event_relations[]
structure_bindings
production_assumptions[]
```

Every structure binding references stable event or state IDs. Conflict is
allowed but not required by the frozen Creative Structure Spec.

#### `screenplay_v1`

```text
screenplay_id
blueprint_id
target_duration_s
characters[]
scenes[]
beats[]
dialogue_or_text_cues[]
structure_trace
production_requirements
```

`structure_trace` maps the four frozen structure IDs to screenplay scene/beat
IDs. It is provenance, not a request to mention abstract IDs in the story.

Existing `asset_graph_v1` and `shot_plan_v1` remain authoritative downstream
schemas.

### 4.3 Artifact DAG

```text
creative_structure_spec
  -> theme_candidate_pool
  -> theme_selection
  -> story_blueprint_candidate_pool
  -> story_blueprint_selection
  -> screenplay_candidate_pool
  -> screenplay_selection / committed screenplay
     -> asset_graph
        -> approved character/environment/prop assets
     -> shot_plan (also depends on asset_graph)
        -> storyboard frames / shot contracts
        -> image/video candidate pools
        -> selected shots
        -> assembled master
        -> approval
```

Any changed or revoked parent recursively makes selections, approvals, and
downstream artifacts stale through the existing workspace dependency graph.

## 5. Candidate Pool and Best-of-N

### 5.1 Candidate record

Candidate content is an immutable artifact; the pool stores references rather
than copying the content repeatedly.

```json
{
  "candidate_id": "...",
  "artifact_type": "theme_candidate",
  "artifact_sha": "...",
  "parent_shas": ["..."],
  "generation_config_sha": "...",
  "producer_skill": {"id": "...", "version": "..."},
  "status": "eligible",
  "scorecard_ids": ["..."],
  "selected": false
}
```

Candidate status is separate from workspace artifact status. The minimal pool
states are:

```text
generated -> schema_valid -> eligible -> scored -> selected
                              |            |
                              v            v
                           rejected     superseded
```

### 5.2 Default search policy

The first implementation uses the requested bounded policy:

| Level | Generate | Keep | Expensive side effects allowed |
|---|---:|---:|---|
| Theme | 20 | 5 | No |
| Story blueprint | 5 per selected theme, max 25 | 3 overall | No |
| Screenplay | 1 per selected blueprint, max 3 | 1 | No |
| Storyboard/shot realization | Configured per selected screenplay | Stage policy | Images only after contracts commit |
| Video | Hardware/budget policy | Final selection | Yes, after all upstream gates |

Counts are configuration values under a versioned selection policy; changing
them creates a new experiment/policy version, not a silent runtime tweak.

### 5.3 Scorecards and selection

Scores remain multidimensional:

```text
structure_compliance
originality
coherence
emotional_effect
visual_potential
production_feasibility
cost_risk
```

There is no universal scalar score stored by the candidate. A versioned
selection policy decides eligibility, hard failures, weights, tie-breaks, and
whether human approval is required. The ranker never receives the reference
video or reference representation.

## 6. Creative pipeline

### 6.1 Theme generation

Input allowlist:

- exact `creative_structure_spec_v1` public fields;
- user creative brief;
- target format, duration, audience, language, safety constraints;
- explicit production envelope.

Output: `theme_candidate_pool_v1`.

Deterministic gates:

- schema and stable IDs;
- `parent_structure_sha` equality;
- all binding slots present;
- all four structure IDs bound;
- no reference context fields or source IDs;
- candidate count and token/call budget.

Semantic evaluation checks whether the written premise actually entails its
declared bindings. It receives only the spec, candidate, and rubric.

### 6.2 Story planning

Input: one selected theme plus the structure spec.
Output: `story_blueprint_candidate_pool_v1`.

The planner creates characters, goals, stakes, events, and causal/temporal
relations. The structure spec constrains the information update but does not
dictate genre, character identity, conflict type, ending tone, or visual form.

The compliance validator verifies:

1. I0, E1, and I1 bind to distinct explicit states/evidence;
2. R1 connects those exact bindings;
3. event IDs resolve and ordering is possible;
4. the update is not merely a new action with no interpretation change;
5. no extra reference-derived invariant is claimed.

### 6.3 Screenwriting

Input: one selected blueprint, the public structure spec, format constraints,
and production envelope.
Output: `screenplay_candidate_pool_v1`.

The writer cannot access reference-zone artifacts. An external screenwriting
skill may add craft reasoning for scene structure, character, dialogue, and
revision, but its response must compile to `screenplay_v1`.

The screenplay gate checks schema, blueprint coverage, structure trace,
duration envelope, dialogue/text feasibility, and required assets. A separate
blind creative-quality evaluator may judge coherence and craft.

### 6.4 Production preparation

The current repository imposes this dependency order:

```text
committed screenplay
  -> asset_graph_v1
  -> approved visual assets / bibles
  -> shot_plan_v1 (screenplay + asset_graph)
  -> storyboard frames and shot contracts
```

Storyboard does not rewrite story structure. A proposed story change returns
to the screenplay or blueprint stage and invalidates all dependent artifacts.

### 6.5 Generation and assembly

Generation adapters accept committed shot contracts and approved assets only.
They produce immutable candidate sets with model/config/seed/input hashes.
Neutral observation is separated from repair instructions. Assembly accepts
only selected shots whose parent contracts and asset SHAs remain current.

## 7. Evaluation and repair loop

### 7.1 Evaluation layers

| Layer | Method | Purpose |
|---|---|---|
| Schema/lineage | Deterministic | Parseability, IDs, SHA parents, version and policy match |
| Structure compliance | Deterministic mapping checks plus one bounded semantic audit where entailment is required | Preserve I0/E1/I1/R1 without reference access |
| Creative quality | Blind model judge and/or human scorecard | Originality, coherence, emotional effect |
| Production feasibility | Deterministic limits plus specialist review | Duration, asset count, motion complexity, cost risk |
| Visual/video quality | Measurements plus isolated visual reviewer | Identity, continuity, legibility, action, artifacts |
| Final approval | Human or explicit configured policy | Authorize release candidate SHA and parent SHAs |

A deterministic validator never upgrades a semantic claim merely because IDs
resolve. A model judge never overrides schema, lineage, revocation, budget, or
permission failures.

### 7.2 Repair routing

```text
failure classified
  -> find earliest invalid artifact
  -> preserve valid parents and siblings
  -> create a new artifact version
  -> rerun only dependent validators and selections
  -> recursively stale downstream artifacts
```

Examples:

- missing R1 in a theme: regenerate/repair that theme only;
- blueprint relation unsupported: repair blueprint, not screenplay wording;
- screenplay dialogue issue: repair screenplay, then invalidate production;
- asset identity issue: repair that asset and dependent shots;
- local video artifact: repair that shot candidate, not the story.

Every skill has `max_calls` and `max_repair_attempts`. Exhaustion produces a
blocked artifact and explicit user decision; it never starts an unbounded
self-critique loop.

## 8. Lineage, selection, and approval

### 8.1 Required provenance

Each artifact records:

```text
artifact ID/type/version/SHA
exact parent IDs and SHAs
producer run and skill version
skill package and prompt/instruction SHA
model/config/seed where applicable
schema and policy versions
validation artifacts
selection artifact
approval binding
```

Selection is an artifact:

```text
selection_id
pool_sha
selected_candidate_id
selected_candidate_sha
scorecard_shas[]
selection_policy_version
decision_origin
```

Approval continues to bind candidate SHA, parent SHAs, policy version, and
reviewer. A new candidate, new parent, new skill package, or changed policy
cannot reuse an old approval.

### 8.2 Freeze boundaries

- The R2-C.5 spec SHA is immutable for R2-D v1.
- A changed structure creates `creative_structure_spec_v2`; it does not edit
  v1 in place.
- Third-party skill updates require a new pinned package SHA and revalidation.
- Selection-policy changes invalidate score-derived selections, not source
  candidates.
- Repair never overwrites failed outputs or traces.

## 9. Security and isolation boundary

### 9.1 Data zones

| Zone | May read | May write | Forbidden |
|---|---|---|---|
| Reference | Reference media and verified R2 artifacts | Reference representations and audits | Creative candidate generation |
| Publish boundary | Validated structure/audit inputs | Public structure spec | Passing source text, paths, claim IDs, raw media, audit annex |
| Creative | Public structure spec, user brief, production envelope, current candidate parents | Theme/blueprint/screenplay candidates | All reference-zone artifacts and hidden evaluation labels |
| Production | Committed screenplay, asset graph, shot plan, approved assets | Media candidates and assemblies | Reopening reference interpretation or story structure implicitly |
| Evaluation | Candidate under review and stage rubric | Scorecard/audit | Mutating candidate, changing policy, revealing sealed labels to generator |

### 9.2 Request isolation

Creative request builders are whitelist constructors. They must reject unknown
fields rather than forward whole dictionaries. Logs and error feedback use
field paths and rule codes; they do not echo blocked reference content.

### 9.3 Third-party skill policy

Before an external skill is enabled:

1. pin repository commit/package hash and record license;
2. inventory every `SKILL.md`, script, reference, and asset that can load;
3. review for filesystem, shell, network, credential, and prompt-injection risk;
4. assign a permission profile;
5. run contract tests with fake artifacts and no secrets;
6. permit only the minimum input payload and output directory;
7. prohibit direct workspace mutation and committed-artifact overwrite;
8. store all raw output as untrusted until validation succeeds.

Default permissions for a screenwriting skill are: no network, no shell, no
reference-zone paths, no credentials, read-only request payload, and write-only
temporary output. Installation is a separate explicit action.

## 10. Failure model and observability

Failure classes are stable reason codes:

```text
contract_invalid
reference_boundary_violation
lineage_stale
skill_untrusted_or_unavailable
generation_failed
structure_noncompliant
quality_below_policy
production_infeasible
budget_exhausted
approval_required
```

Every call trace records stage, skill/package hashes, input artifact SHAs,
sanitized request hash, raw-response artifact path, token/time/cost data,
validator results, retry count, and output SHA. Trace content remains outside
the public downstream artifact.

## 11. Frozen stage graph and implementation plan

The end-state architecture is frozen now; implementation may be delivered in
dependency order without redesigning later stages.

### Wave 1 — Contracts and runtime extension

- add common envelopes and candidate-pool schemas;
- extend existing `SkillSpec` metadata without breaking legacy skills;
- add permission and budget enforcement;
- add deterministic fake skills;
- prove workspace lineage and invalidation for the new artifact DAG.

### Wave 2 — Creative planning

- implement Theme and Story Blueprint adapters;
- implement deterministic structure-binding validators;
- implement candidate pools, scorecards, and selection artifacts;
- run local golden-path and negative-control tests;
- perform one bounded model acceptance only after static review.

### Wave 3 — Screenwriting integration

- security-review and pin the selected external skill package, or use the
  internal adapter;
- compile output to `screenplay_v1`;
- validate blueprint and structure trace;
- compare against the fake/internal adapter using the same fixtures.

### Wave 4 — Existing production adapters

- adapt selected screenplay to `asset_graph_v1`;
- reuse asset-studio acceptance;
- adapt screenplay plus asset graph to `shot_plan_v1`;
- reuse storyboard validators and backends.

### Wave 5 — Media search, evaluation, and repair

- add image/video candidate pools;
- wrap current generation backends behind shot contracts;
- add selection and approval binding;
- add scoped repair and assembly;
- run a limited end-to-end real-model acceptance.

No wave may silently change an upstream schema or selection policy. Required
changes produce a versioned design amendment and regression plan.

## 12. Acceptance criteria for the eventual implementation

R2-D through the final-video loop is accepted only when:

1. a fake-skill end-to-end run reaches a committed master with no model calls;
2. every artifact has exact parent SHA lineage and stale propagation works;
3. creative requests contain no reference media, path, claim/event ID, source
   wording, audit annex, old DNA, or hidden evaluation label;
4. surface-similar stories without R1 fail structure compliance;
5. cross-domain stories with the same four-component mechanism pass;
6. changing the structure spec SHA invalidates every dependent selection;
7. Best-of-N generation is reproducible from pool/policy/config hashes;
8. generators and judges are isolated and cannot edit one another's artifacts;
9. external skills run under the declared permission profile and exact package
   SHA;
10. local full tests pass before any real-model call;
11. expensive media generation cannot start without committed upstream gates;
12. final approval binds the master SHA and all required parent SHAs.

## 13. Deferred decisions

The architecture deliberately does not freeze:

- which screenwriting skill implementation wins;
- the model used for theme/story/screenplay generation or judging;
- creative-quality weights and human-review policy;
- final per-stage N/K values after the baseline experiment;
- image/video model selection and hardware routing;
- whether R2-C.4 is later rerun as a genuine human study.

These are versioned configuration or evaluation decisions, not reasons to
change the artifact DAG or security boundary.

## 14. Research and implementation basis

- [Plan-and-Write: Towards Better Automatic Storytelling](https://ojs.aaai.org/index.php/AAAI/article/view/4726) supports explicit storyline planning before surface story generation.
- [Agent Skills specification](https://github.com/agentskills/agentskills/blob/main/docs/specification.mdx) defines a skill package as `SKILL.md` plus optional scripts, references, and assets; R2-D treats that package as versioned input to a controlled runtime, not as trusted authority.
- [screenwriting-skills](https://github.com/jtydhr88/screenwriting-skills) is a candidate external craft library with multiple specialist skills. It is not yet installed or selected as the production writer.
- [Story-Film-Skills](https://github.com/badgids/Story-Film-Skills) provides a useful implementation comparison for durable project files, stable IDs, deterministic validators, and recoverable workflows. R2-D reuses the repository's own existing equivalents instead of importing its entire pipeline.
- [SoK: Agentic Skills — Beyond Tool Use in LLM Agents](https://arxiv.org/abs/2602.20867) and [Agent Skills for Large Language Models](https://arxiv.org/abs/2602.12430) motivate treating third-party skills as governed supply-chain inputs with explicit trust and permission boundaries. These recent surveys inform the threat model; they do not validate this architecture experimentally.

---

Freeze rule: implementation may refine internal function names and file
placement, but changing the public artifact DAG, reference boundary, skill
permission model, evaluation separation, or lineage semantics requires a
versioned architecture amendment.
