---
id: no-tmp-leak
status: active
order: 2
incident: forensic — verifier rotation post-2026-04 began probing /tmp; left-over scratch from prior runs leaked across trials and produced false positives.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies whenever the agent wrote to /tmp during investigation, repro, or testing"
---
   SCRATCH-DIR HYGIENE — anything you wrote under `${SCRATCH}/` (test scripts, log captures, build outputs, repro snippets, data dumps) MUST be removed before submitting unless the task explicitly placed a deliverable there. Today's verifier may not probe `${SCRATCH}/`, but tomorrow's clean-room verifier will. Run `rm -rf ${SCRATCH}/<your-paths>` (or `rm -rf ${SCRATCH}/* 2>/dev/null || true` if you do not own anything else under `${SCRATCH}/`) as part of your pre-submit pass.
