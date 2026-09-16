# V9-G service order

The first real run intentionally uses sequential GPU ownership:

1. `bash scripts/v9g/serve_h3.sh fl2va 30010`
2. run the FL2VA capability cases, stop the service;
3. `bash scripts/v9g/serve_h3.sh ref2va 30011`
4. run the Ref2VA and hybrid cases, stop the service;
5. only then start the Omni pool for observation and verification.

The server command follows the official SGLang contract: the model path is the
root `MiniMaxAI/MiniMax-H3`; `--model-variant` selects `fl2va` or `ref2va`.
Never point `--model-path` at a downloaded `FL2VA/` or `Ref2VA/` subdirectory.
