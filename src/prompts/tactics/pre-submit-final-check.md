---
id: pre-submit-final-check
status: active
order: 3
incident: aider-polyglot R#7 — multiple submissions on partial matches; agent saw "service running" and skipped the literal verification command from the task description.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "always-on; runs immediately before echo ${SUBMIT_SENTINEL}"
---
8. **CRITICAL — pre-submit final-check pass:** before `echo ${SUBMIT_SENTINEL}`, do these checks in one bash call (chained with `&&` or in a heredoc):
   (a) Re-read the original task instruction line by line. Watch for plurals ("print **all** moves", "for **each** input"), edge cases ("if there are multiple X"), and END-TO-END flows ("then `curl http://...` should return Y").
   (b) For end-to-end flows, **literally run the verification command from the task description** (e.g. `curl -s http://localhost:8080/hello.html` and inspect output, not just that the service is up).
