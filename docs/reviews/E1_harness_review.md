# E1 harness review

Reviewed `research/e1-localization-capture` at commit `c878026`, against `docs/preregistrations/E1_localization_capture.md`.

**Method and limits.** This review comes from reading the code only. Nothing was executed, so every "CONFIRMED" means confirmed by reading the code and working through the arithmetic. No source for transformers 5.22 or torch 2.11 was available, so API claims about those versions rest on memory (see section 4).

**Bottom line.** I found no wrong-token, off-by-one, padding, hook-leak, cache-key or calibration/test leak bug. The token-level core looks correct. The defects are in four places:
- the small-tier design
- what the resume guard does not check
- an unlogged concept-vector choice
- failure modes that turn into an ordinary-looking verdict

**Status labels.** CONFIRMED means the defect is in the code as read. PLAUSIBLE means the code path is real, but whether it hurts depends on model behaviour or library behaviour I could not observe.

---

## 1. Defects, worst first

### D1. Small tier: each concept only ever meets one parity of position, which can produce a false V2 failure (CONFIRMED)

**Where:** `src/mechinterp/design.py:209`, `experiments/e1_localization_capture/configs/small.json:12-14`, gate at `src/mechinterp/analysis.py:196`. The test that misses it is `tests/test_design.py:45-53`.

**Failure.** The start position is `concept_index * per_cell + cluster_index * max(1, 10 // n_clusters)`. With 10 concepts, 5 clusters and 1 position per cell, that is `c + 2k`. A concept therefore sees `{c, c+2, c+4, c+6, c+8}`. Even-indexed concepts get only even labels, and odd-indexed concepts get only odd labels. Position is balanced within each cluster, as the prereg says, but not within each concept.

**Effect on each statistic:**
- **`L` is mostly protected.** `m_clean` is subtracted per position, and within a balanced cluster the margin sums to zero over positions, so label priors and position-unspecific logit shifts cancel. The point estimate of `L` is unbiased if concept and position effects are additive. A concept-by-parity interaction cannot be separated from the concept effect.
- **`hit` is not protected, and gate V2 reads `hit`.** A concept-resampling replicate has an unequal mix of even and odd positions, so a label prior becomes fake between-concept variance.
  - Worked case: suppose the model mostly answers "7" (a likely digit favourite). Odd concepts then score 0.2 from the prior alone and even concepts score 0.
  - The point estimate stays at 0.10. The share of odd concepts in a replicate has SD 0.158 with 10 concepts, so hit has SD about 0.032 and a lower bound near 0.04.
  - A true accuracy around 0.15 could then miss the `> 0.10` bound in this prior-driven case.
  - A fully crossed design has no such term. So a real effect can be reported as `V2_fail`, which the prereg treats as a finding about the model.

**Fix.** Set `positions_per_cell_test: 2` in the small config. The existing formula then gives `2c + 2k`, and every concept sees all ten positions once while every cluster sees all ten positions twice. This doubles test trials to 800, so the ceiling at `tests/test_design.py:215` must rise. Two alternatives:
- 5 concepts x 5 clusters x 2 per cell, which keeps the budget and is also fully balanced, but has fewer concepts
- 10 x 10 x 1, which is also a Latin square

Add a test that positions are balanced within concepts. Log the change in the prereg, since no small-tier data exist yet.

### D2. Resume guard ignores code, model weights, dtype and environment, so a run can silently mix them (CONFIRMED)

**Where:** `src/mechinterp/runner.py:132-135`, all configs (`"revision": "main"`, line 5), `src/mechinterp/export.py:28` (only the last manifest is exported), `src/mechinterp/runner.py:249-255`.

**Failure.** The guard compares only `config_hash` and `data_sha256`. `L` subtracts a stored clean logit from a newly computed injected logit, and these can come from different sessions. The following can all change between sessions without detection:
- **Code.** Task text lives in `prompts.py`, not in the config. If a fix is pasted as a new bundle and the run resumes in the same `out` directory, old and new rows mix. This workflow is the likely one, because the code has never run and fixes are expected after the pilot.
- **Weights.** `revision: "main"` is not a pinned commit. The manifest records `commit` but never compares it.
- **Dtype and device.** A CPU float32 session and a GPU bf16 session are accepted into one store (see D8 for why that matters). `batch_size` is inside the config hash, but device and dtype are not.
- **Library versions on Colab.** Recorded per session, never compared. The bundle does not pin them, and only CI pins them.
- **Concept vectors.** `concept_vectors.pt` is lost with a Colab runtime and recomputed silently after `restore_checkpoint`, with no check that the new vectors match.

The export then shows only the latest manifest, so an earlier environment disappears from the report.

**Fix.**
- Pin `revision` to a commit hash in each config, as the prereg says identifiers are fixed in the config.
- Hash `src/mechinterp/*.py` into the manifest.
- On open, refuse a store whose code hash, resolved model commit, dtype or major library versions differ.
- Record a hash of the concept vectors and compare it after recomputing.
- Have `build_export` report every distinct environment.

### D3. "Final-token" concept vectors are read after the chat tail, an unlogged choice that may leave early-layer vectors nearly concept-free (deviation CONFIRMED; effect PLAUSIBLE)

**Where:** `src/mechinterp/vectors.py:19-20,53`, `src/mechinterp/prompts.py:94-96`, `concept_prompt_mode: "chat"` in all configs (line 7), prereg line 19.

**Failure.** The prereg says the "final-token residual for 'Tell me about {c}.'". In chat mode on Qwen3, the final token is the `\n\n` after `<think>\n\n</think>`, roughly ten tokens past the concept word and not part of that sentence. It is a different token on Llama and Gemma, so tiers are not strictly comparable.

At layers 0 and 3, which are in the small grid and where the prereg expects the strongest capture, little concept information may have reached that token. The difference is then normalized to unit length (`vectors.py:85`), so a weak signal looks exactly like a strong one. The only diagnostic is a mean raw norm.

Two consequences follow if the vectors are weak:
- The "concept" condition behaves like a structured random direction.
- That would shrink the concept-versus-random gap used for H3 and could weaken `L_inj`, which feeds V2 and the share `C_t`.

This choice is also missing from the prereg change log, which records comparable decisions.

**Fix.** Decide and log the extraction token before any data exist. Add a diagnostic per layer: pairwise cosine between concept vectors, and the raw difference norm relative to the activation norm at that token. Consider comparing `raw` mode (final token ".") on calibration concepts only.

### D4. A single NaN becomes an ordinary "no effect" verdict (PLAUSIBLE; mainly on GPU)

**Where:** `src/mechinterp/runner.py:423`, `src/mechinterp/analysis.py:54`, `src/mechinterp/stats.py:37-39`, `src/mechinterp/runner.py:521-526,536`.

**Failure.** Nothing checks that logits are finite. JSON accepts `NaN`, and one NaN makes its cell NaN. Then `0 * NaN` in `weighted_cell_mean` makes every replicate NaN, even those with zero weight on that cell. `percentile_ci` returns NaN, and `nan > x` is False, so the outcome is `V2_fail` or `mixed` rather than an error. In the impact search, NaN JS is treated as "too high" and quietly drives the bisection.

The risk is low on CPU float32. It is real for the float16 fallback at `modeling.py:32` with alpha up to 10.

**Fix.** Raise in `_run_clean_batch` and `_run_trial_batch` if logits are non-finite, and raise in `CellTable` on a non-finite value.

### D5. H3 can be reported as supported when impact matching failed, and small-tier search settings make failure likely (code CONFIRMED; trigger PLAUSIBLE)

**Where:** `src/mechinterp/analysis.py:177`, `src/mechinterp/runner.py:536-561`, `configs/small.json:26`, `configs/pilot.json:26`.

**Failure.** The failed match is not silent: it is recorded as `match_ok: false`, printed and added to warnings. But `supported` is still `impact_minus_norm.lo > 0`, so the boolean does not reflect it.

With `max_evals: 4` and a 10% tolerance, the search from the selected alpha typically goes `a, 2a, sqrt(2)a, 2^(1/4 or 3/4)a`. That final resolution is about 19% in alpha, and JS can change faster than alpha. If the target is more than one doubling away, there are even fewer bisection steps. The prereg defines H3 on impact-matched directions, so an unmatched run does not test it.

**Fix.** Return `supported: null` when `match_ok` is false, and raise `max_evals` for the small tier. The pilot's JS history can size this.

### D6. Injected-run answer mass is never checked, so `hit` and `L` may describe tokens the model would not emit (CONFIRMED that it is unchecked; effect PLAUSIBLE)

**Where:** `src/mechinterp/analysis.py:205-211` (clean rows only), `src/mechinterp/export.py:62` (the injected value is exported, then unused).

**Failure.** Scoring is restricted to label logits. At high alpha, or on tasks like T_fav, the injected distribution could move off the labels entirely, while `hit` still takes an argmax among the labels and `L` still moves. Capture could then be read from a distribution whose mass is elsewhere. The warning only looks at clean runs, and selection maximizes accuracy with no mass constraint.

**Fix.** Report mean injected `label_mass` per task and condition in the report, and warn below a threshold. Log whether this is purely descriptive so it cannot become a post-hoc filter.

### D7. Small-tier percentile intervals probably undercover, in the direction of finding effects (CONFIRMED by arithmetic for the simple case; implements the prereg as written)

**Where:** `configs/small.json:12-13`, `src/mechinterp/stats.py:35-41`, `tests/test_stats.py:41-42`.

**Failure.** The resampling is right, but percentile intervals with five clusters are short. When cluster variance dominates, the bootstrap SD of a mean carries a factor `sqrt((n-1)/n)`, and the correct reference is t with 4 df. That gives `1.96 * sqrt(0.8) = 1.75`, so true coverage is about 85% rather than 95%. With concept variance dominating (n = 10), coverage is about 90%.

This affects the lower bounds used by gate V2 and the capture rule. The prereg acknowledges a "thin bootstrap" but not this direction. The coverage test only exercises a dense 20 x 20 table, accepts anything from 0.85 to 1.0, and so says nothing about the shipped 10 x 5 design.

**Fix.** The data holds 10 test clusters, so use more of them if timing allows. Add a coverage simulation at the shipped dimensions, and state the measured coverage next to any small-tier result.

### D8. bf16 logits are quantized on the GPU tier (PLAUSIBLE)

**Where:** `src/mechinterp/modeling.py:30-33,105`.

**Failure.** bf16 has 8 bits of mantissa, so logits between 16 and 32 have a step of 0.125. `.float()` after the head does not restore that precision. Per-trial `L` gets rounding noise, which averages down, but exact ties become common, and `argmax` then favours the lower label index. The sham check cannot see this because it compares identical runs.

CPU tiers are unaffected.

**Fix.** Keep the body in bf16 and run the output head in float32, or run in float32 on the GPU tier.

### D9. Pilot runs on the same test concepts and clusters as later tiers, with estimates visible (CONFIRMED; low)

**Where:** `src/mechinterp/design.py:149,167-168,174-175`, `configs/pilot.json:4`, `src/mechinterp/analysis.py:247-263`.

**Failure.** All tiers share one seed and truncate one shuffle, so the pilot's test concepts and cluster are a prefix of the small tier's. `pilot_not_for_inference` withholds only verdicts, while `estimates` are still in the report. The model differs and no code path feeds these back into selection. But the prereg allows counts to change after the pilot, so a person can see test-split effects before changing the design.

**Fix.** Have the pilot draw only from the calibration split, or drop `estimates` from pilot reports and show only timing and token checks.

### D10. Unlogged or minor departures from the prereg (CONFIRMED; low)

- The change log says small tier runs 1 position per cell. `positions_per_cell_calibration` is actually 2 (`small.json:11`). This is harmless but unlogged.
- The change log says `C_t` is undefined when `|L_inj| < 0.05`. `stats.py:81,89` also treats any negative denominator as undefined. This is stricter, cannot change a verdict that passes V2, and should be logged.
- The selection tie-break at `design.py:364` (smallest alpha, then earliest layer) is not in the prereg. With 20 trials per grid point on the small tier, ties are likely. Log it.

---

## 2. Tests that claim more than they test (CONFIRMED)

| Test | Problem |
|---|---|
| `test_runner.py:112-115` `test_clean_and_injected_runs_use_identical_tokens` | Compares only length, and both sides come from the same cached prompt. Tokens are never compared. |
| `test_runner.py:55-61` `test_selection_is_reproducible_from_calibration_records_only` | Never re-runs selection. It only checks that the stored accuracy equals the maximum of the stored grid. |
| `test_design.py:45-53` | Checks balance within clusters only, which is why D1 passes. |
| `test_stats.py:41-42` | Dense 20 x 20 table and a wide acceptance band; not the shipped design (D7). |
| `test_analysis.py:154-157` `test_undefined_share_bounds_give_mixed_never_capture_or_gated` | The name says "never", but the body asserts `task_gated` and `capture` for one-sided undefined bounds. Those cases are almost unreachable, because `ratio_summary` voids both bounds at nearly the same point, but the name is wrong. |
| `test_runner.py:40-41` sham check | `alpha = 0` is a no-op wherever it is applied, so it cannot detect a wrong position. |
| `test_prompts.py:151-152` | `importorskip("jinja2")`, and CI does not install jinja2 explicitly. If it is not pulled in transitively, the only real-template test skips silently. |

**Two coverage gaps matter more than any single test above:**
1. No test connects a runner row to the token it injected. The hook and the prompt positions are tested separately, but `_run_trial_batch`'s mapping from trial to token to scored label is not. A small stub model whose logit for label `j` responds only to a perturbation at `positions[j]` would test this end to end, and should give high `L` and `hit` through the real runner.
2. No test uses a real tokenizer or chat template (CI is offline by design). Offsets and label ids after `</think>\n\n` are unverified until the pilot. `dry_run` prints the top tokens but asserts nothing. Add assertions on Colab before the pilot starts: the decoded word at each position, the decoded labels, and clean label mass per task, not only T_inj.

---

## 3. Areas that look correct

- **Token position.** Words are located by character span inside the list only, never in the T_fact instruction (`prompts.py:155-169`). The token must cover the span exactly and decode to the word. Trimmed and untrimmed offsets are both handled.
- **Layer index.** Injection and vector capture both hook the same module output (`self.layers[l]`). `hidden_states` is not used, so there is no off-by-one inside the harness. One thing to remember when comparing with papers: config layer `l` equals `hidden_states[l+1]`.
- **Label tokens.** Resolved in rendered context per task, must be identical across tasks, and raise rather than mis-score. Reasoning through the Qwen and Llama split patterns, `\n\n` followed by a digit gives a bare digit token.
- **Padding and batching.** Batches have one exact length, with no padding, and the logit is read at index -1. Rows stay aligned through grouping, because each record is built from its own trial.
- **Clean versus injected.** Both use the same `clean_key` and the same cached prompt. The clean store key includes task, cluster and query, and it correctly excludes concept.
- **Hooks.** Removed in `finally`. They clone rather than edit in place, and use each token's own norm.
- **`L` estimand.** Matches the prereg, using only the ten label logits. T_inj_none's eleventh entry is excluded from the margin and included in the answer argmax.
- **Task prompts.** Identical apart from the instruction. No control instruction mentions injection or anything unusual. Instruction precedes the list as designed.
- **Calibration and test separation.** Grid and impact search use calibration concepts and clusters only, enforced at `assert_trial_respects_split` and in `select_layer_alpha`. The test phase refuses to start before both records exist. Screening uses test concepts on calibration clusters and affects only the secondary analysis.
- **Impact search and test data.** It uses no test data. It replays deterministically on resume, because alphas are rounded to the key's precision.
- **Bootstrap.** Concepts and clusters are drawn independently as multinomial counts. `(concept_weights @ sums * cluster_weights)` correctly handles repeated draws as a pigeonhole bootstrap. The same weights across statistics keep differences and ratios paired.
- **Holm.** Correct step-down with monotonic adjustment, in separate families, matching the change log.
- **Capture share.** Ratio of means within each replicate. Undefined replicates count against both bounds, which is conservative.
- **Decision rules.** V1 is inclusive at 70% on clean T_fact across all ten queries in test clusters. V2 uses both bounds. The order is pilot, V1, V2, then rules. Capture, task-gated and mixed thresholds match. T_pick and T_fav disagreement gives mixed, and H2 uses the same rule on T_odd. Requiring Holm in addition to the bound is logged and only makes the rules stricter.
- **Interference.** `hit - clean_hit` on the same prompt with the same query matches the prereg.
- **Checkpoint mechanics.** I found no route to duplicate or dropped rows: keys are unique and fsynced, torn tails are truncated, and clean logits round-trip exactly. Different configs and data are refused. D2 covers what the guard does not compare.
- **Data lists.** By reading, I found no exact overlaps between concepts, baseline words and candidate words, or across splits, and `validate_data` enforces this at load. No candidate word relates to injection or oddity in a way that makes a control non-neutral. Some test concepts are semantically close to test clusters (`wine` and groceries, `mountains` and landscape). That adds heterogeneity rather than bias. Cross-split neighbours such as `silver` and `gold` are different concepts.
- **Bundle and CI.** Bundle contents are deterministic and contain code and data only. The workflow runs on push with pinned versions.

---

## 4. Runtime errors against torch 2.11, transformers 5.22, Python 3.13

I found no definite runtime error. Reading through, these all look sound:
- inference-tensor use outside `inference_mode` (only functional and view operations, cloned before saving)
- `weights_only=True` loading
- string seeding of `random.Random`
- `logits_to_keep` detection, which falls back safely
- tensor-or-tuple layer outputs

**Unverified from source, so check these first on Colab or in the first CI run:**
- `PreTrainedTokenizerFast(tokenizer_object=...)` in `tests/conftest.py:32`
- `decode(..., clean_up_tokenization_spaces=False)` in `prompts.py:125`
- `_commit_hash` still being set on the config (`runner.py:171`). If it is absent, `commit` is silently `None`, which makes D2 worse.

**One failure that would be loud, not silent:** the large-tier configs need all 10 test clusters to keep at least 10 single-token words. `bathroom` looks tight (`toothbrush`, `bathtub`, `faucet`, `bathrobe`, `shampoo`), and one short cluster raises `ValueError`. Adding spare words before the run would avoid this.
