import { describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();

describe("DFlash MLX patch compatibility", () => {
  test("keeps baseline cache kwargs aligned with the installed mlx-lm generate_step signature when available", () => {
    const patch = readFileSync(join(root, "patches", "dflash-triattention-mlx.patch"), "utf8");
    expect(patch).toContain('baseline_kwargs: dict[str, Any] = {"sampler": sampler}');
    expect(patch).toContain('baseline_kwargs["max_kv_size"] = args.max_kv_size');
    expect(patch).toContain("kv_bits=args.kv_bits");
    expect(patch).toContain("kv_group_size=args.kv_group_size");
    expect(patch).toContain("quantized_kv_start=args.quantized_kv_start");

    const python = existsSync(join(root, ".venv", "bin", "python")) ? join(root, ".venv", "bin", "python") : undefined;
    const vendorDflash = join(root, "vendor", "dflash");
    if (python === undefined || !existsSync(vendorDflash)) {
      return;
    }

    const output = execFileSync(
      python,
      [
        "-c",
        [
          "import inspect, json",
          "from mlx_lm.generate import generate_step",
          "from dflash.model_mlx import stream_generate",
          "print(json.dumps({",
          "  'generate_step': list(inspect.signature(generate_step).parameters),",
          "  'dflash_stream_generate': list(inspect.signature(stream_generate).parameters),",
          "}))",
        ].join("\n"),
      ],
      {
        cwd: root,
        encoding: "utf8",
        env: {
          ...process.env,
          PYTHONPATH: [vendorDflash, process.env.PYTHONPATH].filter(Boolean).join(":"),
        },
      },
    );
    const signatures = JSON.parse(output) as {
      generate_step: string[];
      dflash_stream_generate: string[];
    };

    expect(signatures.generate_step).toEqual(expect.arrayContaining([
      "max_kv_size",
      "kv_bits",
      "kv_group_size",
      "quantized_kv_start",
    ]));
    expect(signatures.dflash_stream_generate).toEqual(expect.arrayContaining([
      "cache_optimization",
      "turboquant_strategy",
      "triattention_enable",
      "triattention_kv_budget",
    ]));
  });
});
