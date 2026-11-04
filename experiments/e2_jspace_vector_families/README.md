# E2: J-space vector families

[Preregistration](../../docs/preregistrations/E2_jspace_vector_families.md) · [Review](../../docs/reviews/E2_review.md)

> **Paused, and not ready to run.** The review's items 1-7 and the prereg's design correction (F_sham, necessity and sufficiency, undetermined outcomes) are not implemented, so this code does not yet match the corrected design. One data-loading test is an expected failure (98 vs 99 items in one evaluation set, unresolved).

Commands run from the repo root.

E2 asks whether E1's injected-position advantage depends on the part of the vector inside J-space, or on the rest. The code has been tested only on tiny models in CI, and **has not run on a real model**.

On the machine that ran E1, once E1 is `DONE`:

```bash
bash mechinterp/scripts/run_e2_gpu.sh
```

It refuses to start while E1's status is anything but `DONE`, while E1 holds a lock, or while an E1 process is alive. That includes an `E1_ON_FINISH` hook, so start E2 by hand rather than from one. It reuses E1's venv and model cache, and installs nothing. State goes in `e2_qwen3_4b/` inside E1's state directory, and results in `results/` there, copied to `results/e2/qwen3_4b/` in the clone, with `SHA256SUMS`. Detaching, resume, `E2_RETRY=1`, `E2_ON_FINISH` (with `E2_FINAL_STATUS`), bounded waits and the status banner work as in E1.

**Stages**

1. **Lens and Stage 1.** Downloads the lens by pinned commit, checks its hash, and runs the lens checks on one GPU. Then it measures pass@10 per layer on all six evaluation sets. If the gate fails, the status is `GATE_FAILED`, the banner says so, and nothing else runs. A failed lens check is `FAILED` instead: a broken instrument says nothing about the model. `gate.json` is frozen once written, so changed code or config is refused rather than remeasured.
2. **Dry run.** E1's checks on E2's tasks, the number of single-token concepts, and a runtime estimate from measured speed.
3. **Stage 2.** Worker 0 builds the families once. `stage2.json` gives J-share, top lens tokens and Spearman intervals at `l_s`. It reads E1's results and gives a clear error if they are missing or unfinished. It decides nothing.
4. **Stage 3.** F_full on calibration at `l_w`'s alpha grid, then E1's impact search for each other family and site, then the full test cross. The merge checks exact keys and clean pairing, as in E1.

**Trial count.** The test stage is E1's full cross, with no subsampling: 40 concepts x 10 clusters x 10 positions x 3 tasks x 2 sites = 24,000 per family. That is 96,000 for F_full, F_j, F_res and F_rand, plus 600 for each single-token test concept under F_lens: 120,000 at most. Calibration adds 4,500 for F_full and at most 32,000 for the searches, usually far fewer. At most 156,500 prompts in all, which is about the size of E1's own run (roughly 146,000), so expect a similar runtime plus some minutes for Stage 1 and the vectors. The dry run gives the measured figure. If `l_w` equals `l_s` there is one site, and the count halves.

## Lens facts, checked by retrieval on 2026-11-04

| Item | Value |
|---|---|
| Code | `anthropics/jacobian-lens` at `581d398613e5602a5af361e1c34d3a92ea82ba8e`, read and not imported |
| Lens | `neuronpedia/jacobian-lens` at `e166deb5ad540932100ef803c8c9bcc2a5453694` |
| File | `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` |
| LFS pointer | 458,762,977 bytes, sha256 `9c94a43ac2db2debbab36acb74480f49a3ac411fb3d7fa8e5461377028e74606` |
| Fit | `Qwen/Qwen3-4B`, 2026-06-12, 479 prompts of raw WikiText, bf16, target layer null (the final block), transformers 5.11 |

**Stage 0.** `config.yaml` records no model revision. Qwen3-4B's newest commit is `1cfa9a7` (2025-07-26), which is the commit E1 pins and predates the fit, so the lens was fit on E1's checkpoint. No refit is needed, and that branch of Stage 0 is not implemented. E1 runs float32 where the fit ran bf16.

**`norm` is the model's final RMSNorm, not a unit normalization.** Upstream's `unembed` is `lm_head(final_norm(x))`, and the logit lens goes through the same norm. `J_l h` is `h @ J_l.T`.

**Layer indices.** Lens key `l` is the output of block `l`, which is the module E1 injects at, so there is no offset. HF's `output_hidden_states` would be off by one and is not used. The file is expected to hold blocks `0..34` of 36, with `35` as the target, which its size implies: 35 x 2560^2 x 2 bytes. This is asserted on load, not assumed.

**Checks at runtime, any of which stops the run:** size and hash; `weights_only`; integer keys, shape, dtype and finiteness; keys exactly `0..n-2`; final block through norm and head reproduces the model's logits; identity J reproduces the logit lens; the lens puts the model's own answer in its top 10 on known prompts at the last source layer; the per-prompt Jacobian, measured with upstream's estimator, matches the file's rows at offset 0 rather than one block off. That last one is paired but noisy, so only three quarters of tested layers agreeing on one nonzero offset fails it. Stage 1 prompts are raw text, not chat, as the sets are written and as the lens was fit. Concept vectors stay at E1's chat read site, so the lens is applied to chat-context vectors it was not fit on, which the paper does too.

## Choices beyond the paper and the prereg

Gradient pursuit (`pursuit.py`). The paper gives the objective and names the method.
- Atoms are unit-normalized for selection. Cones do not change, only the order of selection.
- Selection takes the largest positive inner product. Each selection is followed by one clipped gradient step with exact line search.
- A final exact nonnegative least squares on the chosen atoms, because the paper defines the component as the nearest point. This also makes the residual orthogonal to the component, so shares add to one. A coefficient that reaches zero frees its slot.
- The dictionary is the tokenizer's base vocabulary minus special tokens, unfiltered otherwise. W_U's unused and control rows are out.
- Rows of `W_U J_l` leave out RMSNorm's learned per-dimension gain, as the paper's definition does. `final_norm_gain` records how uneven that gain is.

Stage 1 (`gate.py`, `evalsets.py`).
- A band is a maximal run of consecutive layers, each with its own lower bound above 0. The largest mean gain wins, and an even band takes the lower middle layer.
- Recall pools items across the six sets, and an item is the mean over its intermediates that can be scored. Per-set tables and upstream's any-layer metric are reported too.
- A target matches as written, capitalized or lowered, each with and without a leading space, single-token forms only. Intermediates with no such form are left out and counted. Qwen tokenizes digits separately, so "12" counts only through "twelve", and only in order-ops.
- `before_target` tokenizes prompt plus target, since many prompts end in a space that merges into the target. The other reading is measured too, and reports whether the gate would differ.
- All upstream items are used: the files hold more than the paper's counts in most sets.

Stage 3 (`session.py`, `analysis.py`).
- `l_w`'s alpha is E1's rule on E1's alpha grid, at that layer, on calibration data.
- The search stops at E1's large-tier 5%, and a family counts as matched at the prereg's 10%. Matching uses T_inj prompts only, as E1's search does.
- F_lens targets F_full's JS on the same single-token concepts. A concept's token is `" {concept}"`, as it appears in the concept prompt.
- Holm covers each bound a rule must clear: the J-minus-residual gap, and `L(F) - 0.5 L(F_full)` for F_j, F_res and F_rand, over the cells where the rule can be evaluated. E1 left share bounds unadjusted, so `classification_without_holm` is reported beside the result.
- An unmatched F_rand leaves H3 untested, and the cell is still classified on H1 and H2, with `h3_testable` false. An unmatched F_j or F_res gives mixed, with `routes_testable` false.

## Entries for the prereg's change log

The prereg is not edited by this work. These need logging before data is collected:

1. Chinese targets: the evaluation sets hold none, so `chinese_targets.json` is our own table. Descriptive only.
2. Order-ops synonyms: upstream describes them but does not publish them, so `order_ops_synonyms.json` is ours. It affects the gate.
3. Tolerance: the prereg says both "E1's tolerance" (5% at the large tier) and "the 10% tolerance". Resolved as above.
4. The gate's band reading, and pooled recall.
5. Holm's scope, including the ratio bounds, and the treatment of unmatched families.
6. Readout position for prompts ending in a space.
7. "Line closes" read as: no task at any site is H1 or H2.
8. `l_s = l_w` collapses to one site.

## Unverified without a GPU

The lens file has never been opened: its keys, dtype and layers are inferred from its size and from upstream's code. Nothing has run on a real model. Not known until a run: whether `tokenizer.vocab_size` excludes Qwen's added tokens as expected; the number of single-token concepts; whether the dictionary fits beside the weights on a 24 GB card, with a fallback to CPU if not; whether the alignment check's backward pass fits, which is skipped with a warning if not; and the speed.
