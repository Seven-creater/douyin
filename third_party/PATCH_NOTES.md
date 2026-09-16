# V9-G upstream patch notes

## CineCrew

Pinned at `9a00efa7db65ba028c1523907f6cee844776123a`. The original trace helper
depends on CineCrew configuration and Markdown span rendering. V9-G adapts only
its per-run trace-directory and atomic artifact-writing behavior to a JSONL
boundary. No CineCrew internal schema class is imported by first-party code.

## NEWTON

Pinned at `072bda240a5991bc4da6ff8b6cd5f6f33fcfc058`. The loop-memory data model
and serialized attempt trace are retained. V9-G wraps them so upstream class
names are not part of the public V9-G API. Planner, executor and verifier logic
remain first-party and contract-specific.
