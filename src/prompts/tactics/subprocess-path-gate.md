---
id: subprocess-path-gate
status: active
order: 5
incident: METR-TH R#13 — verifier ran tool via subprocess.run() with default PATH; agent's `export PATH=` succeeded in shell but the verifier's clean subprocess could not find the binary.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when the task says a tool must be in PATH / system-wide / callable as `X`"
---
   (d) **SUBPROCESS-PATH GATE** — when the task says a tool must be "in PATH" / "available system-wide" / "callable as `X`", the actual verifier runs in a fresh subprocess (typically `subprocess.run(['X', ...])` from Python) which has the DEFAULT system PATH (`${PATH_JOINED}`) and does NOT inherit your shell's `export PATH=`, aliases, or virtualenv activation. Verify your fix from a clean shell: `bash -c 'unset PATH; PATH=${PATH_JOINED} command -v X'` must succeed. If it doesn't, persist the binary system-wide via one of: `ln -s /full/path/X ${PERSIST_TARGET}/X`, `cp X ${PERSIST_TARGET}/`, `pip install --user X`, or place the binary in a directory already on default PATH. `export PATH=` alone is insufficient.
   If any check disagrees, fix it BEFORE submitting. Do not submit on partial matches.
