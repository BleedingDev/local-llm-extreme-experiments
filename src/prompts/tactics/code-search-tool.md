---
id: code-search-tool
status: deprecated
order: 91
incident: covered by the always-on tool selection guide in principles.md; kept as a stub so we can re-promote when the code_search A/B harness shows clear over/under-use.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when the task is large and the agent is unsure where the relevant code lives"
merged_into: principles
---
code_search guidance lives in principles.md (tool selection guide).

If forensic evidence shows agents systematically reach for `bash rg` on conceptual questions ("where is auth", "where is rate limiting"), re-activate this tactic with a body that explicitly directs the agent to call `code_search` first for conceptual queries and `bash rg` only for exact-token searches.
