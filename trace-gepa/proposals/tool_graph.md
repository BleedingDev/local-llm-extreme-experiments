# Proposal R6-U: Tool-Transition Directed Graph

## TLDR
- **Hypothesis**: Pointwise classifiers ignore the Markov structure of agent traces; the conditional `P(tool_{t+1} | tool_t)` distribution carries strong, cheap priors (e.g., `git status` overwhelmingly precedes `git add`/`git diff`; `Read` precedes `Edit`). A weighted directed graph distilled from ~30K records exposes these priors as a reusable, inspectable artefact.
- **Artefact**: `data/tool_graph.json` (nodes=tool-keys, edges=transitions w/ count + smoothed prob), `data/tool_graph.dot` (Graphviz, edges thresholded at support>=5), `data/tool_graph_summary.md` (top-20 by support, top-20 by lift, dead-ends).
- **Three downstream uses**: (a) BAG planner-prompt suffix `top_k_followups(current_tool)`; (b) anomaly feature `-log P(observed | prev)` fed into R5 detector; (c) MCP server exposing `predict_next_tool(prev) -> [{tool, prob, support}]`.
- **Distinct from R1/R2**: not retrieval (R2 trace-RAG returns whole episodes), not new tool synthesis (R1 MCP-from-patterns mines repeated motifs into macros) — this is a compact 2nd-order statistical artefact.

## Construction
1. Walk every session in trace order. For consecutive `(a_i, a_{i+1})`, emit edge.
2. **Node key**: for `Bash`, use `Bash:<leading_token>` (`bun`, `git`, `rg`, `ls`, ...); for `Bash:git`, sub-tokenize to `Bash:git:<subcmd>` (`status`, `add`, `commit`). For other tools, just the tool name. This keeps the graph from collapsing into a Bash-megastar.
3. **Smoothing**: Laplace `α=0.5` over observed-targets-per-source. Prune edges with `count<3` for the .dot export.
4. **Session boundary**: insert virtual `<START>` / `<END>` nodes — useful for "what do users open with" priors.

## Use cases (concrete)
- **BAG runtime hint**: planner prompt gets appended `Hint: after {prev_tool} users typically run {top3 with %}`. Cost: 1 dict lookup, ~30 tokens.
- **Anomaly feature**: existing R5 detector gets new scalar `transition_surprisal`; merge via logistic head.
- **MCP**: 40-line server reads `tool_graph.json`, exposes one function. No retraining loop.

## Effort + ROI
- Effort: **0.5 dev-day** (pure stream-aggregation pass, no model). Json+dot output is trivial.
- ROI: low-risk shared substrate — the same artefact powers three independent consumers (BAG, R5, MCP). High inspection value (a human can eyeball the graph and find suspicious edges in minutes).

## Self-critique
First-order Markov ignores longer-range dependencies (e.g. `Read → Bash → Edit` where the `Bash` is incidental); a context-conditioned bigram or skip-gram extension would help, but adds tuning surface — keep v1 plain.

Path: `trace-gepa/proposals/tool_graph.md`
