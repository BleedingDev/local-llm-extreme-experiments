---
id: cleanup-before-submit
status: active
order: 1
incident: polyglot-c-py R#11 — 4 trials lost to leftover cmain binary; verifiers asserting exact file lists rejected the deliverable directory because a compiled artifact was left next to the source.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when the deliverable is a source file but the agent compiled an executable to test"
---
   COMPILED-LANGUAGE GATE — when the task's deliverable is a SOURCE file but you compiled an executable to test it (gcc, g++, cc, clang, rustc, cargo build, go build, make, etc.), explicitly remove the compiled artifact (`rm -f <binary>`) before submitting. Verifiers that assert the deliverable directory contains an exact set of files reject any byproduct.
