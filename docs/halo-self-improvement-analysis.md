# HALO Self-Improvement Analysis For BleedingAgent

Source: https://github.com/context-labs/halo

## What HALO Actually Does

HALO is a trace-driven optimizer for agent harnesses. The core loop is:

1. Collect detailed execution traces from the agent harness.
2. Store traces as OTLP/OpenInference-shaped JSONL spans.
3. Build an index over trace IDs, spans, errors, models, token counts, and services.
4. Let a specialized trace-analysis agent inspect systemic failures across many runs.
5. Produce a report describing recurring harness failure modes.
6. Feed that report to a coding agent to patch prompts, tools, schemas, or control flow.
7. Re-run evals and repeat.

The important design point is that HALO optimizes the harness, not just a single answer. It looks
for repeated failure modes across traces: hallucinated tool calls, invalid tool arguments, refusal
loops, redundant arguments, slow/oversized trace regions, semantic task failures, and brittle
handoffs.

## Useful HALO Internals

HALO trace records use a canonical span shape:

- `trace_id`, `span_id`, `parent_span_id`
- `name`, `kind`, `start_time`, `end_time`
- `status.code`, `status.message`
- `resource.attributes`
- `scope.name`, `scope.version`
- `attributes` carrying OpenInference-style keys such as `llm.model_name`, `tool.name`,
  `input.value`, `output.value`, token counts, agent names, and project IDs.

HALO then builds a sidecar trace index with:

- trace IDs and byte offsets into the JSONL file;
- span counts and start/end timestamps;
- error presence;
- service names, model names, agent names;
- total input/output tokens;
- project ID.

HALO exposes trace tools:

- dataset overview;
- query traces;
- count traces;
- view one trace;
- view selected spans;
- search inside one trace.

The tool design is intentionally bounded. Large traces are not dumped blindly; the engine searches
or fetches selected spans and caps payload sizes.

## What We Applied Now

BleedingAgent now keeps the existing simple metrics, but also writes HALO-style spans:

- `.bag/telemetry/events.jsonl`: event stream for operator/debug use.
- `.bag/telemetry/metrics.json`: aggregate run metrics.
- `.bag/telemetry/spans.jsonl`: OTLP/OpenInference-shaped span stream.

Each run writes:

- `agent.run` root span;
- `step.*` spans for deterministic/master/local phases;
- `llm.<role>.<model>` spans with model, endpoint, HTTP status, duration, and token counts;
- `tool.<namespace>.<tool>` spans with tool name, description version, retry count, input hash,
  capped input preview, output size/kind, capped output preview, and error details.

Self-optimization now reads both metrics and spans. It can detect:

- failed tool/LLM/step clusters;
- repeated failing tool input hashes;
- repeated error messages;
- trace counts and error span counts;
- p50/p95 latency clusters;
- observation-kind breakdowns.

`bag self-optimize` includes this HALO-style analysis in the generated candidate and markdown
report, then can apply only safe local changes: config patches and `.bag/tool-guidance.md`.

## What Is Implemented Properly Now

BleedingAgent now has the missing HALO-style trace-store layer:

- builds `.bag/telemetry/spans.jsonl.index.jsonl`;
- builds `.bag/telemetry/spans.jsonl.index.meta.json`;
- groups spans by `trace_id`;
- records byte offsets and byte lengths into the original JSONL trace file;
- tracks span counts, error span counts, services, models, agents, observation kinds, span names,
  token totals, project IDs, and sample trace IDs;
- supports dataset overview, trace query/count, full trace view, selected span view, and substring
  search inside a trace;
- caps string attributes to bounded budgets and returns oversized summaries instead of dumping huge
  traces into the model context.

The ACP provider exposes this through `/traces`, so users in any compatible ACP client can inspect
trace health without using a separate interactive CLI.

Self-optimization now produces eval-gated improvement proposals, not only passive findings:

- target: tool guidance, prompt policy, runtime policy, or eval suite;
- rationale from metric and trace evidence;
- patch sketch describing the intended improvement;
- eval gate describing the before/after check required before trusting the change.

Applying a candidate writes:

- `.bag/tool-guidance.md`;
- `.bag/self-improvement-plan.md`;
- optional `bag.config.json` policy patch.

It still avoids mutating project source files directly. That boundary is intentional: the ACP coding
agent can edit source when the user asks it to, but self-improvement artifacts must pass eval gates
before becoming runtime policy.

## What We Still Do Not Have

This is not full HALO yet. Missing layers:

- no dedicated RLM/DSPy optimization agent over trace tools;
- no automatic prompt/tool-schema patch generation under eval gates;
- no A/B harness comparing before/after agent versions;
- no long-horizon eval suite for coding tasks.

That is intentional for this slice. The agent remains simple and ACP-first: ACP clients own the UI,
and BleedingAgent owns routing, coding actions, traces, evals, and optimization artifacts.

## Next Practical Slice

The next high-value implementation is the RLM/model optimizer over the trace store:

1. Let the master model inspect trace clusters through the internal trace tools.
2. Generate proposed prompt/tool-schema patches as artifacts, not direct source edits.
3. Run a fixed coding eval set against old and new guidance.
4. Apply only improvements that pass the eval delta.

That gives us the useful part of HALO without making BleedingAgent a heavy standalone CLI product.
