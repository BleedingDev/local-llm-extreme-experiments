# Task Mutation: Semantic-Preserving Augmentation (175 -> 500+)

## TLDR
- **Mutate, don't synthesise:** paraphrase + token-substitute existing tasks while preserving `expected.primary_action` and `verifier_spec` -- gold answer remains valid, supervision signal multiplies 3x.
- **Reuses round-9 reverse_validator** to gate every mutation: if the original gold answer no longer passes the verifier on the mutated task, reject the mutation. Zero new verifier authoring cost.
- **Cheap and fast:** ~10K Opus prompt-tokens/task * 175 = ~$5-10 one-shot; output stored as `task_id_mutN` with full provenance metadata pointing back to parent.
- **Triple payoff:** (a) variance reduction on bench scores, (b) prompt-robustness signal via mutation-invariance scoring, (c) detector for fragile gold answers (tasks where mutations break verifier expose under-specified expected fields).

## Hypothesis
Bench thinness (n=175) inflates score variance and hides prompt-sensitivity. Controlled mutations preserving the verifier contract grow effective n without requiring new gold annotations -- because the mutation invariant *is* "verifier still passes the original primary_action."

## Pipeline
1. **Paraphrase user_request:** Opus call -- "Rephrase 3 ways, preserve technical intent, preserve any concrete identifiers tagged `<keep>...</keep>`." Identifiers tagged via cheap regex pre-pass over verifier_spec hits.
2. **Token substitution pool:** maintained `mutation_pool.json` with equivalence classes -- TS paths (`src/foo.ts <-> lib/bar.ts`), Python paths, repo names (`acme/api <-> acme/web`), person names. Substitute only where verifier_spec doesn't reference the literal.
3. **Validate every mutation:** run original `expected.primary_action` through reverse_validator against mutated `user_request`. If validator's posterior on gold drops below original task's posterior - 0.05, reject.
4. **Provenance:** `task_id`, `parent_task_id`, `mutation_kind` ('paraphrase' | 'substitute' | 'both'), `seed`, `validator_score`. Bench reports both raw and parent-deduplicated scores.

## Use Cases
- **Variance reduction:** report bench score with bootstrap CI over parent-clusters, not raw rows.
- **Robustness metric:** "mutation-invariance" = mean agreement of agent's primary_action across siblings. Low score => prompt-fragile.
- **Fragility audit:** tasks where >50% of mutations fail the validator gate flag under-specified expected fields -- feed back to round-3 verifier hardening.

## Self-Critique
Mutations cluster in semantic neighbourhoods of originals, so they reduce variance but don't expand bench coverage -- this is augmentation, not exploration, and pairs with (not replaces) round-4 synthetic generation.
