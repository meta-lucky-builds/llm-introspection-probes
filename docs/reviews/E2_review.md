# E2 implementation review

Commit `59c4d73`, branch `research/e1-localization-capture`. Done by reading only: nothing was run except `bash -n`, which passes. Definitions were compared with `anthropics/jacobian-lens` at `581d398` (`lens.py`, `hf.py`, `hooks.py`, `fitting.py`), the lens's `config.yaml` and LFS pointer at `e166deb`, and 2607.15495 sections 2.1, 2.3, 2.5, 3.1, A.6, A.7 and A.8.

**Bottom line.** I found no convention error in the lens: the readout, J's direction and the layer index all match upstream and the paper. I found no bug that would crash or corrupt a run on reading. The worst problem is in the design: H2 is close to guaranteed by construction if J-share is small. The other problems are a decision-rule hole, a missing transposition guard, and several ways to lose billed time. Fixes to the gate's analysis must land before Stage 1, because `gate.json` is frozen once written.

## Defects, by harm

### 1. H1 and H2 are not a fair contest, because F_res is nearly F_full. PLAUSIBLE (the geometry is exact, the size is not yet measured)
`src/mechinterp/e2/families.py:54-60`, `src/mechinterp/e2/analysis.py:102-103`, prereg lines 47-49 and 61.

The NNLS residual is orthogonal to the component, and the vector is unit length. So exactly `cos(F_res, F_full) = sqrt(1 - j_share)` and `cos(F_j, F_full) = sqrt(j_share)`. The paper reports a median J-share of 6-7% for concept vectors at k = 16 (section 3.1). If Qwen is similar, cos(F_res, F_full) is about 0.97 and cos(F_j, F_full) about 0.25, and both are then scaled to the same JS.

Failure:
- `R(F_res) >= 0.5` then holds almost automatically whenever F_full has any effect.
- H1 requires F_j to beat what is nearly a copy of F_full, while H2 requires only that F_j does not. So "H2, residual route" is close to the default outcome, and it would be reported as "localization here does not need verbalizable content" (prereg line 61).
- F_res also keeps all overlap with J-lens vectors beyond the top 16, and the paper finds the residual's small effect on report is itself routed through the J-space, by clamping (section 3.1). F_res is not outside J-space in that sense.

Nothing in the code surfaces this. `residual_norm` is stored per concept and never summarized.

Fix, before data, logged in the prereg:
- Report `sqrt(1 - j_share)` per site and task in `stage2.json` and `summary.txt`.
- Add a control that gives H2 a null. For example, remove from F_full the component on 16 atoms drawn at random with the same removed share. Then ask whether F_res beats that control. Or state that R(F_res) is uninformative when J-share is below a preregistered value.

### 2. An untested rule reads as "mixed" and closes the line. CONFIRMED
`src/mechinterp/e2/analysis.py:100-112` and `:136-141`, locked in by `tests/test_e2_rules.py:159-165`.

Failure:
- If F_j or F_res misses the 10% tolerance at both sites, every cell is labelled `mixed`, and `line_closes_for_qwen3` is True. The prereg's stop rule closes on a finding of "H3 or mixed". Here the routes were never tested. F_j, at about 25% overlap with F_full, is the family most likely to need a large alpha, and the search is capped at 32x.
- In the other direction, an unmatched F_rand leaves H3 untested but the cell can still be `H1`, with `ablation_warranted` True, although H3 "takes precedence". Only `rules.h3_testable` shows this, and `format_summary` does not print it.

Fix: use a distinct label such as `routes_untestable`, and make `line_closes_for_qwen3` None when any cell lacks a tested route. Print `H1 (H3 untested)` and do not set `ablation_warranted` from such a cell.

### 3. No runtime check catches a transposed J, though the code says one does. CONFIRMED (a missing guard, not an actual error)
`src/mechinterp/e2/stage1.py:81` ("a transposed J should fail"), `src/mechinterp/e2/lens.py:302-353`, `tests/test_e2_pursuit_and_lens.py:297-304`.

Failure:
- `alignment_verdict` only asks whether 75% of layers prefer one nonzero offset among -1, 0 and +1. A transposed file gives low cosines at all three offsets, the best offset is noise, and it passes. So does any wrong file with offsets like `[0, 1, -1, 0]`.
- There is no floor on the cosine and no margin: a 1e-5 cosine difference counts as a "best offset".
- The known-concept check runs at layer 34, where the fit's own `final_identity_distance` is about 0.39, so J is near the identity and transposing it is close to invisible there.
- The transposition test only asserts a lower cosine and never calls the verdict.

So a wrong J would degrade the J-lens and fail the gate for the wrong reason, as `GATE_FAILED`, which is a permanent stop rule.

Off-by-one: `check_structure` catches only a wrong key set. A shift within keys 0..34 relies on this same weak verdict.

Fix, in `measure_alignment`:
- Also compute the cosine of each measured row with the stored column `J[:, dim]`, and fail if columns beat rows on most tested layers.
- Fail if the offset-0 cosine is below a floor set well above that of random rows.
- Require a margin before counting a nonzero best offset.

### 4. Any code fix discards every row, and this code has never run. CONFIRMED (the mechanism), PLAUSIBLE (the cost)
`src/mechinterp/e2/config.py:21-28`, `src/mechinterp/e2/session.py:39-42`, `src/mechinterp/e2/stage1.py:376-382`, `scripts/run_e2_gpu.sh:598`.

`e2_code_hash` covers all of `e2/` and all of E1's top-level modules, including `analysis.py`, `orchestrator.py`, `cli.py` and `gate_text.py`, none of which touch a row. A one-line fix to the final merge after tens of hours makes `_assert_same_identity` refuse all shards, with no override. A fix to `analysis.py` also makes the script refuse at step 4, before `write_results`. Only the manual `finalize` command then works.

Fix: split the hash. Rows and the gate check a hash of the modules that affect measurement (`lens`, `pursuit`, `families`, `session`, `design`, `worker`, `evalsets`, `stage1`, `gate`, plus E1's `code_hash`). Record the full hash without enforcing it.

### 5. NNLS non-convergence goes unnoticed, and the vectors stage has no heartbeat. PLAUSIBLE
`src/mechinterp/e2/pursuit.py:23-25` and `:53-61`, `src/mechinterp/e2/families.py:116-136`, `src/mechinterp/sharding.py:700-707`.

Failure:
- The paper says J-lens vectors collapse to a small subspace in early layers (Figure 28d), and an early `l_s` is the expected case here. Near-collinear atoms make cyclic coordinate descent slow.
- At the 5,000-sweep cap it returns silently. The comment says `component_dot_residual` "reports any shortfall", but nothing reads that value, so shares may not add to one and F_res may not be orthogonal to F_j.
- `compute_families` writes no heartbeat. Worst case is about 4 s per solve, up to 4 solves per concept, 60 concepts, 2 sites, which is about 30 minutes. That exceeds the 20-minute stall limit. The worker is killed, restarts from zero, and a second stall stops the run. The CPU dictionary fallback makes this more likely.

Fix: use an exact active-set NNLS on the at most 16 by 16 problem, raise or warn if `|component_dot_residual| > 1e-6`, and touch the heartbeat per concept.

### 6. The synonym sensitivity the prereg promises is not computed, and half of it cannot be. CONFIRMED
Prereg line 72, `src/mechinterp/e2/stage1.py:207-209` and `:353-357`.

The self-written synonyms enter the gate (55 of 552 items, half their intermediates). The prereg says the gate "can be recomputed without them". `ranks.jsonl` stores only the minimum over key and synonyms, so a keys-only gate cannot be recomputed. Dropping order-ops altogether is possible from the file, but `analyze_rows` does not do it. The only sensitivity it computes is for readout position.

Both lenses are scored on the same tokens, so this is not a direct bias towards either. But the gain on operation words may depend on our word list.

Fix, before Stage 1: store ranks per word and add `sensitivity_without_order_ops` and `sensitivity_keys_only`. Both gate nothing.

### 7. The gate validates a different functional from the dictionary. PLAUSIBLE
`src/mechinterp/e2/lens.py:13-15` and `:221-238`.

The gate reads `W_U norm(J h)`, which includes RMSNorm's per-dimension gain. The atoms are rows of `W_U J`, without it. This matches the paper's definition, and the README discloses it. But `norm_gain_report` only prints summary statistics, and no number says how far the dictionary being used is from the readout that passed the gate.

Fix: in Stage 2, report the mean cosine between rows of `W_U J` and `W_U diag(g) J`, and J-share under both. It decides nothing.

### Low

- **Time limit, CONFIRMED.** `scripts/run_e2_gpu.sh:65` allows 48 h, while E1's script allows 72 h for a smaller prompt count. `src/mechinterp/e2/worker.py:169` never compares its estimate with this limit. Hitting it gives FAILED, and `E2_ON_FINISH` may stop the machine. Fix: use 72 h and fail in preflight if the estimate exceeds the limit.
- **Late identity check, CONFIRMED.** `src/mechinterp/e2/orchestrator.py:321-323` compares E1's config, data and commit only after Stage 1, the dry run and all workers have loaded. A mismatch then needs a new state directory and a new Stage 1. Fix: check E1's manifest in `check_machine`, using standard library code. This reveals no results.
- **Unmatched families still run, PLAUSIBLE waste.** `src/mechinterp/e2/orchestrator.py:383-388` spends 12,000 test prompts per unmatched family and site, on rows the rules ignore. The prereg's "reported as unmatched" allows either reading. Choose one deliberately.
- **Stale failure reason, CONFIRMED.** `scripts/run_e2_gpu.sh:511-513`: on a retry, a Stage 1 crash that is not a `LensError` reports the old `failure_reason.txt` from an earlier failure. Fix: remove the file before Stage 1.
- **Skipped E1 vector check, CONFIRMED.** `src/mechinterp/e2/families.py:194-196`: if E1's `concept_vectors.pt` is missing, the check is skipped with a warning that goes to a worker log. It is recorded in `decomposition.json` but not in the summary. Fix: add it to `run_warnings`.
- **Dead config keys, CONFIRMED.** `analysis.family_alpha` and `analysis.min_denominator` in E2's config are never used, and `n_bootstrap` is used only for Stage 2. `src/mechinterp/e2/analysis.py:218-222` takes these from E1's manifest and a literal 0.05. The values agree today, but they are hashed as if they mattered.
- **Stale README.** `README.md:206-208` says the prereg is not edited and that entries still need logging. The prereg's change log already has them.

### Tests that pass without testing what they claim

- `tests/test_e2_pursuit_and_lens.py:297`: see defect 3.
- `tests/test_e2_pursuit_and_lens.py:281`: fits the "lens" on the same prompt it measures, so the cosine is 1.0. This says nothing about how the verdict behaves with one prompt against a corpus mean. `tests/test_e2_run.py:187` has the same problem.
- `tests/test_e2_rules.py:142-147`: the assertion sits inside an `if` and can pass vacuously.
- `tests/test_e2_data.py:100-105`: re-implements the glob and asserts two hashes differ, which is always true. `:108-112`, "changed nothing E1 reads", only checks a config string and one missing file.
- `compare_with_e1`'s comparison branch is never run: the factory writes no `concept_vectors.pt`, and the session passes None.
- `write_passing_gate` is handwritten, so no test goes from a real gate to Stage 3. No script path is tested beyond the refusals, and the README says as much.
- `check_identity_is_logit_lens` compares `read(None)` with `read(I)`, which can hardly fail.

## What looks right

**Lens**
- `h @ J.T` is upstream's `transport`, and it equals `J_l h`.
- Rows of J are `d z_i / d h`, so `W_U[t] @ J` is the paper's row of `W_U J_l`.
- Final RMSNorm comes before `W_U`, for both the J-lens and the logit lens.
- Lens key l is the output of block l, sources are `range(target)`, and the target is block 35. `config.yaml` has `target_layer: null`, and the file size implies 35 matrices at float16.
- `prompts_fitted: 479` and the LFS sha256 and size match the pins.
- dtype: float16 converted to float32, device moves are right, and Qwen has no BOS or softcap to mishandle.
- Readout rules match upstream's README for all six sets.

**Pursuit**
- It solves the paper's problem: nonnegative, at most k, k = 16 as in Figure 8. Unit atoms leave the cones unchanged.
- `component + residual` equals the target exactly in float64.
- Size: 151,643 atoms is 1.55 GB, about 4.7 GB at peak in `unit_atoms`. That is feasible beside 16 GB of weights, each selection is a single matrix-vector product, and there is a CPU fallback.

**Gate**
- Paired resampling over items, per layer.
- Maximal runs of at least 3 layers, a lower bound strictly above 0, and the largest mean gain gives `l_w`.
- Both lenses are scored on identical token sets, and ties are treated identically.

**Impact matching and leakage**
- Matching is per family and site, on calibration only, through E1's search. F_lens is compared like with like.
- F_full at `l_s` uses E1's alpha, and `l_w` uses E1's rule on calibration data.
- Exclusion is above 10%.
- Nothing derived from E1's test data sets an E2 parameter. `L_inj` and `L_pick` feed only Stage 2's description.

**Decision rules**
- H1 to H4 and undefined handling match the prereg, and H3 takes precedence.
- Holm runs over cells per comparison, across sites and tasks. The p-values are two-sided, and differences use paired replicates.

**Script**
- It refuses on E1's status, locks and live processes, and its `pgrep` pattern does not match E2's own module.
- Paths to E1's venv, `hf_home`, `.model_ready`, `results/results.jsonl` and `run/concept_vectors.pt` all match `run_e1_gpu.sh` and the runner.
- The gate-fail exit path, the frozen gate, and resume all work on reading. Clean rows are never foreign, so pairing holds with all workers reading other shards.

**E1 identity**
- `code_hash` globs `*.py` non-recursively, so `e2/` is excluded.
- `git diff bdcf034~1 HEAD` over `src/mechinterp` outside `e2/`, `run_e1_gpu.sh`, E1's experiment folder, `pyproject.toml` and the shared test helpers is empty.

## Not verifiable here

Nothing was executed. The real J-share, whether F_j can be impact matched, the RMSNorm gain, alignment cosines on the real file, and speed are all unknown until a run.
