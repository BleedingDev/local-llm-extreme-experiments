import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import {
  applyEdit,
  editApplySupportedFamilies,
  parseEditApplyErrorCode,
  type EditApplyWorkspace,
} from "../src/edit-strategy/apply-layer";

const hash = (content: string): string => `sha256:${createHash("sha256").update(content).digest("hex")}`;

const workspace = (files: Record<string, string>, protectedPaths: string[] = []): EditApplyWorkspace => ({
  files: Object.entries(files).map(([path, content]) => ({ path, content })),
  protectedPaths,
});

describe("edit strategy apply layer", () => {
  test("applies whole-file edits without writing protected paths", () => {
    const result = applyEdit(workspace({ "src/a.txt": "old\n", "package.json": "{}\n" }, ["package.json"]), {
      strategyFamily: "whole_file",
      payload: {
        path: "src/a.txt",
        content: "new\n",
        baseContentHash: hash("old\n"),
      },
    });

    expect(result.status).toBe("applied");
    expect(result.changedFiles[0]).toMatchObject({
      path: "src/a.txt",
      beforeContent: "old\n",
      afterContent: "new\n",
    });
    expect(result.previewDiff).toContain("after-sha256");

    const protectedResult = applyEdit(workspace({ "package.json": "{}\n" }, ["package.json"]), {
      strategyFamily: "whole_file",
      payload: { path: "package.json", content: "{\"name\":\"bad\"}\n" },
    });
    expect(protectedResult.status).toBe("failed");
    expect(protectedResult.errorCode).toBe("protected_path_violation");
    expect(protectedResult.changedFiles).toEqual([]);
  });

  test("applies exact replacement and reports stale, missing, and ambiguous matches", () => {
    const ok = applyEdit(workspace({ "src/a.txt": "alpha\nbeta\n" }), {
      strategyFamily: "exact_replace",
      payload: {
        path: "src/a.txt",
        search: "beta",
        replace: "gamma",
        expectedContentHash: hash("alpha\nbeta\n"),
      },
    });
    expect(ok.status).toBe("applied");
    expect(ok.changedFiles[0]?.afterContent).toBe("alpha\ngamma\n");

    const stale = applyEdit(workspace({ "src/a.txt": "alpha\nbeta\n" }), {
      strategyFamily: "exact_replace",
      payload: {
        path: "src/a.txt",
        search: "beta",
        replace: "gamma",
        expectedContentHash: "sha256:stale",
      },
    });
    expect(stale.errorCode).toBe("hash_mismatch");

    const missing = applyEdit(workspace({ "src/a.txt": "alpha\n" }), {
      strategyFamily: "exact_replace",
      payload: { path: "src/a.txt", search: "beta", replace: "gamma" },
    });
    expect(missing.errorCode).toBe("exact_match_not_found");

    const ambiguous = applyEdit(workspace({ "src/a.txt": "same\nsame\n" }), {
      strategyFamily: "exact_replace",
      payload: { path: "src/a.txt", search: "same", replace: "once" },
    });
    expect(ambiguous.errorCode).toBe("exact_match_ambiguous");
  });

  test("applies hash-range edits and rejects stale or out-of-bounds ranges", () => {
    const ok = applyEdit(workspace({ "src/a.txt": "one\ntwo\nthree\n" }), {
      strategyFamily: "hash_range",
      payload: {
        operations: [
          {
            path: "src/a.txt",
            startLine: 2,
            endLine: 2,
            expectedContentHash: hash("one\ntwo\nthree\n"),
            replacement: "TWO\n",
          },
        ],
      },
    });
    expect(ok.status).toBe("applied");
    expect(ok.changedFiles[0]?.afterContent).toBe("one\nTWO\nthree\n");

    const stale = applyEdit(workspace({ "src/a.txt": "one\ntwo\nthree\n" }), {
      strategyFamily: "hash_range",
      payload: {
        operations: [{ path: "src/a.txt", startLine: 2, endLine: 2, expectedContentHash: "sha256:old", replacement: "TWO\n" }],
      },
    });
    expect(stale.errorCode).toBe("hash_mismatch");

    const bounds = applyEdit(workspace({ "src/a.txt": "one\n" }), {
      strategyFamily: "hash_range",
      payload: {
        operations: [{ path: "src/a.txt", startLine: 3, endLine: 3, replacement: "three\n" }],
      },
    });
    expect(bounds.errorCode).toBe("range_out_of_bounds");
  });

  test("applies unified diffs and reports parse/context failures", () => {
    const ok = applyEdit(workspace({ "src/a.txt": "value=old\n" }), {
      strategyFamily: "unified_diff",
      payload: {
        patch: "--- a/src/a.txt\n+++ b/src/a.txt\n@@\n-value=old\n+value=new\n",
      },
    });
    expect(ok.status).toBe("applied");
    expect(ok.changedFiles[0]?.afterContent).toBe("value=new\n");

    const malformed = applyEdit(workspace({ "src/a.txt": "value=old\n" }), {
      strategyFamily: "unified_diff",
      payload: { patch: "not a diff" },
    });
    expect(malformed.errorCode).toBe("parse_error");

    const mismatch = applyEdit(workspace({ "src/a.txt": "value=current\n" }), {
      strategyFamily: "unified_diff",
      payload: {
        patch: "--- a/src/a.txt\n+++ b/src/a.txt\n@@\n-value=old\n+value=new\n",
      },
    });
    expect(mismatch.errorCode).toBe("hunk_context_mismatch");
  });

  test("applies structured apply_patch updates and fails closed on unsupported sections", () => {
    const ok = applyEdit(workspace({ "src/a.txt": "value=old\n" }), {
      strategyFamily: "apply_patch",
      payload: {
        patch: "*** Begin Patch\n*** Update File: src/a.txt\n@@\n-value=old\n+value=new\n*** End Patch\n",
      },
    });
    expect(ok.status).toBe("applied");
    expect(ok.changedFiles[0]?.afterContent).toBe("value=new\n");

    const unsupported = applyEdit(workspace({ "src/a.txt": "value=old\n" }), {
      strategyFamily: "apply_patch",
      payload: {
        patch: "*** Begin Patch\n*** Delete File: src/a.txt\n*** End Patch\n",
      },
    });
    expect(unsupported.status).toBe("failed");
    expect(unsupported.errorCode).toBe("parse_error");
    expect(unsupported.changedFiles).toEqual([]);
  });

  test("keeps supported families explicit and parses stable error codes", () => {
    expect(editApplySupportedFamilies()).toEqual([
      "whole_file",
      "exact_replace",
      "unified_diff",
      "apply_patch",
      "hash_range",
    ]);
    expect(parseEditApplyErrorCode("hunk_context_mismatch")).toBe("hunk_context_mismatch");
  });
});
