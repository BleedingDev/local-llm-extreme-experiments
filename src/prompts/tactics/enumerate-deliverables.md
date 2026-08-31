---
id: enumerate-deliverables
status: active
order: 4
incident: SWE-bench MM R#4 — submitted with only one of three required output files present; verifier checks each declared output independently.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when the task explicitly enumerates deliverables (file paths, output assertions)"
---
   (c) Confirm every output the task specified actually exists with the expected content (`cat /app/move.txt`, `diff <expected> <actual>`).
