---
id: image-input-tool
status: deprecated
order: 90
incident: covered by the always-on tool selection guide in principles.md; kept as a stub so we can re-promote when verifier evidence shows agents misuse view_image.
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when the task references a screenshot or image deliverable"
merged_into: principles
---
view_image guidance lives in principles.md (tool selection guide).

If forensic evidence shows agents fail to load images (or load images they don't need), re-activate this tactic with a body like:
- Always call `view_image(path)` BEFORE asserting on image contents.
- Never call `view_image` to "see what's in the workspace"; use `bash ls`.
