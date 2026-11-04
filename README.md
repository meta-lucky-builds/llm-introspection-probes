# mechinterp

Preregistered experiments on what models report about their own internal state.

**E1** asks whether injection localization is a report or salience capture. Models can name which list position received an injected concept vector. Every published design asks about injection in the prompt. E1 checks whether the injected position also wins when the prompt asks something unrelated. See the [preregistration](docs/preregistrations/E1_localization_capture.md) and the [source verification log](docs/literature/source_verification.md).

## Status

The E1 code is written. **Nothing has been run yet, including the tests**, because this was developed on a machine where execution is not allowed. CI on push is the first real check. No results exist.

## Layout

| Path | Contents |
|---|---|
| `src/mechinterp/identity.py` | Run identity and code hash |
| `src/mechinterp/modeling.py` | Loading by pinned commit, layer discovery, last-token forward pass |
| `src/mechinterp/hooks.py` | `h <- h + alpha * \|\|h\|\| * v` at per-example positions; capture |
| `src/mechinterp/vectors.py` | Concept vectors, seeded random directions |
| `src/mechinterp/prompts.py` | Task text, chat rendering, token location, answer tokens |
| `src/mechinterp/scoring.py` | Margin, `L`, Jensen-Shannon divergence |
| `src/mechinterp/design.py` | Splits, trial plans, `(layer, alpha)` selection |
| `src/mechinterp/runner.py` | Checkpointed `Session` with phases and a dry run |
| `src/mechinterp/checkpoint.py` | JSONL store with fsync, restorable blob |
| `src/mechinterp/export.py` | Compact cell-mean export |
| `src/mechinterp/stats.py` | Two-way bootstrap, percentile intervals, Holm, ratio handling |
| `src/mechinterp/analysis.py` | Gates and decision rules |
| `experiments/e1_localization_capture/` | Word lists and tier configs |
| `src/mechinterp/sharding.py` | Multi-GPU orchestrator, barriers, merge and its checks. Never imports torch |
| `src/mechinterp/shard_worker.py` | One worker per GPU, and the go or no-go dry-run checks |
| `scripts/make_colab_bundle.py` | Packs the code into one Colab cell |
| `scripts/run_e1_gpu.sh` | One command for the large tier on a rented GPU machine |
| `src/mechinterp/e2/` | E2: lens, gradient pursuit, gate, families, session, workers, orchestrator, analysis |
| `experiments/e2_jspace_vector_families/` | E2's config, vendored evaluation sets and word tables |
| `scripts/run_e2_gpu.sh` | One command for E2, on the same machine, after E1 is `DONE` |

## Run on Colab (CPU)

The repo is private and no token goes into Colab, so code goes in by paste and results come out by printed output.

1. On your machine, run `python scripts/make_colab_bundle.py` and paste `dist/colab_cell.py` into a Colab cell. Run it.
2. Open a session and check timing. Colab already has torch, transformers and numpy.

```python
from mechinterp.runner import Session
session = Session.open("experiments/e1_localization_capture/configs/pilot.json", "/content/e1_pilot")
session.dry_run()
```

Check the printed `dropped_words`, `tokens` and `estimated_hours_at_most` before going on.

3. Run a segment. Repeat this cell until `complete` is true. Each segment stops itself after `runtime.segment_seconds`.

```python
session.run()
```

4. Print the results so they leave the runtime. The export holds per (task, vector condition, concept, cluster) cell means.

```python
print(session.export_text())
```

5. Analyze, on Colab or locally from the pasted export. Analysis needs only numpy.

```python
from mechinterp.analysis import analyze, format_report
from mechinterp.export import decode_export
print(format_report(analyze(decode_export(session.export_text()))))
```

### If Colab disconnects

If the runtime survives, re-run `Session.open` with the same paths and carry on. Finished trials are skipped.

If the runtime is lost, `/content` goes with it. To guard against that on a long run, print a checkpoint now and then:

```python
from mechinterp.checkpoint import checkpoint_blob
print(checkpoint_blob(session.store.path))
```

In a new runtime, paste the bundle cell, then restore before opening the session:

```python
from mechinterp.checkpoint import restore_checkpoint
restore_checkpoint("""PASTE BLOB""", "/content/e1_small/results.jsonl")
```

### Order of runs

Run `pilot.json` first. Pilot data check the pipeline and timing only, and the analysis refuses to draw conclusions from them. The pilot uses its own concepts and clusters, so it reveals nothing about the test split. Then run `small.json`. The small tier uses up to about 1,080 prompts (10 concepts, 10 clusters, 2 positions per cell), which is about 3 hours at 10 seconds per prompt or 3.6 hours at 12. These rates are unmeasured, so trust the dry run.

A results directory holds one run identity: code, config, data, model commit, dtype, device type and library versions. Resuming with any of these changed is refused, so a new bundle needs a new output directory.

## Run the large tier on a rented multi-GPU machine

In a terminal on the machine, for example a Jupyter terminal:

```bash
git clone --branch research/e1-localization-capture https://github.com/<owner>/mechinterp.git
bash mechinterp/scripts/run_e1_gpu.sh
```

No token is needed, because Qwen3-4B is ungated. There are no other steps.

**What it does**

1. Checks `nvidia-smi`, GPU memory, disk and RAM, and stops with a plain message if the machine cannot do this. It uses every GPU with 18 GB free, and says so if that is fewer than 4.
2. Builds a venv on Python 3.11 to 3.13, trying `python3.13`, `python3.12`, `python3.11` and then `python3`. If none exists it installs one with uv, which needs no sudo, or with `apt-get` as root. It reuses a working CUDA torch from the machine if there is one and otherwise installs the build that matches the driver. What happened is recorded in `environment.txt`.
3. Downloads the model once, before any worker starts.
4. Runs the dry-run checks on one GPU: every prompt of every stage builds, the prompt ends in the answer prefix, clean top tokens are digits, label mass is near 1 on prefixed tasks, the sham is within its limit, and batch size does not change logits. It prints prompts per second, timed through the real row code, and an estimated finish time. If any check fails, it stops here.
5. Runs concept vectors, the calibration grid, the impact search, screening and the test stage, with one process per GPU. Each stage must finish on all workers before the next one starts. Progress, measured prompts per second per GPU, an ETA and GPU use are printed about once a minute.
6. Merges the shards, checks the merged trials against the plan, and writes the results.

Checkpoints, the venv, the model cache and the results go to `/workspace/e1_large_qwen3_4b` if `/workspace` is writable, otherwise to `.e1_state/` in the clone. Set `E1_STATE_DIR` to choose another place. A state directory holds one run identity, so changed code needs a new `E1_STATE_DIR`.

**How to tell it finished.** The terminal shows a `DONE` or `FAILED` banner, which is also kept in the `status` file in the state directory. `DONE` comes with the analysis between `=====BEGIN E1 ANALYSIS=====` and `=====END E1 ANALYSIS=====`, and with any warnings the run recorded. Results are in `results/` in the state directory, copied to `results/e1/large_qwen3_4b/` inside the clone: `export.txt`, `analysis.txt`, `dry_run.json`, `results.jsonl`, `merge_report.json`, `run.log` and `SHA256SUMS`. Nothing is pushed anywhere, so copy them off the machine. **The machine bills until you stop it.**

**Stopping the machine.** Rented machines have no common way to stop themselves, so set `E1_ON_FINISH` to a command that fits yours:

```bash
E1_ON_FINISH='runpodctl stop pod "$RUNPOD_POD_ID"' bash mechinterp/scripts/run_e1_gpu.sh
```

It runs once, after the final state and results are saved, on both `DONE` and `FAILED`, because a failed run bills too. It can read `E1_FINAL_STATUS` to tell them apart. It does not run after an interruption from outside.

**Running it again** is safe at any time:

- While a run is going, it attaches to it. Closing the browser tab or terminal does not stop the run.
- After a crash, a preempted instance or a killed process, it resumes without repeating finished trials. A different number of GPUs is fine.
- After `DONE` or `FAILED`, it prints that final state at once and starts nothing. To resume after `FAILED`, keeping finished trials, run `E1_RETRY=1 bash mechinterp/scripts/run_e1_gpu.sh`.

**Limits, so that nothing bills forever.** A worker that dies is restarted up to 3 times. A loaded worker with no finished trial for 20 minutes (`E1_STALL_MINUTES`), or one still loading after 30, is killed and restarted once, and a second time ends the run as `FAILED`. Installs, downloads, the dry run and the results step all have time limits, and each says what timed out. A failed run writes no analysis: only a final merge whose trials exactly match the plan can mark a run complete. Worker logs are in `run/logs/` under the state directory.

### How a sharded run stays the same experiment

- **Same plan.** Trials come from the same planning functions as a single-process run, and go to a worker by a SHA-256 of the trial key. The merged `results.jsonl` is what `export` and `analyze` read, unchanged.
- **Barriers.** The selection reads the merged grid. The impact search runs on one GPU and reads the merged grid for its target. Test data is untouched until both exist.
- **Clean baselines.** Every worker runs every clean prompt of a stage itself, which takes seconds, so each injected row is paired with clean logits from its own GPU, and the merge refuses a row that is not. GPUs are compared as soon as those clean prompts finish, at the start of a stage. A gap above 0.01 logits is a loud warning recorded in `merge_report.json`, `results.jsonl` and the analysis, and it does not stop the run, because a steady offset between GPUs cancels in `L`. A fault on one card would not cancel, so a warned run needs comparing by GPU before it is relied on. One probe prompt, measured afresh on each launch before any trial, stops the run above 0.05 logits.
- **Sham.** Every GPU runs the sham check through the real hook on every launch, before any trial, along with the same clean batch against itself, which separates rounding from a hook fault. The prereg says exactly 0. Any nonzero value up to 1e-4 is a warning recorded in the results, to be logged as a deviation. A larger one stops the run, and `E1_SHAM_TOLERANCE` raises that limit.
- **Out of memory.** A batch that runs out of memory is split in half and retried. Rows are stored only after a forward pass succeeds, so none is lost or repeated. Batches are never padded.
- **Config.** `large_4x3090.json` differs from `large.json` only in `batch_size_gpu`. Its config hash therefore differs too, and rows from the two configs refuse to share a store.

## E2: J-space vector families

E2 asks whether E1's injected-position advantage depends on the part of the vector inside J-space, or on the rest. See the [preregistration](docs/preregistrations/E2_jspace_vector_families.md). The code is written and **has not been run**.

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

### Lens facts, checked by retrieval on 2026-11-04

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

### Choices beyond the paper and the prereg

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

### Entries for the prereg's change log

The prereg is not edited by this work. These need logging before data is collected:

1. Chinese targets: the evaluation sets hold none, so `chinese_targets.json` is our own table. Descriptive only.
2. Order-ops synonyms: upstream describes them but does not publish them, so `order_ops_synonyms.json` is ours. It affects the gate.
3. Tolerance: the prereg says both "E1's tolerance" (5% at the large tier) and "the 10% tolerance". Resolved as above.
4. The gate's band reading, and pooled recall.
5. Holm's scope, including the ratio bounds, and the treatment of unmatched families.
6. Readout position for prompts ending in a space.
7. "Line closes" read as: no task at any site is H1 or H2.
8. `l_s = l_w` collapses to one site.

### Unverified without a GPU

The lens file has never been opened: its keys, dtype and layers are inferred from its size and from upstream's code. Nothing has run, including these tests. Not known until a run: whether `tokenizer.vocab_size` excludes Qwen's added tokens as expected; the number of single-token concepts; whether the dictionary fits beside the weights on a 24 GB card, with a fallback to CPU if not; whether the alignment check's backward pass fits, which is skipped with a warning if not; and the speed.

## D1: Double-Down at small scale

Does a lie probe trained on the unmodified model survive ordinary fine-tuning that makes the model always say it was truthful? [Prereg](docs/preregistrations/D1_double_down_small.md), whose change log this section follows. Nothing here has been run, including the tests.

```bash
bash scripts/run_d1_gpu.sh
```

It behaves like E1's script: venv, torch matching the driver, cache on the persistent disk, detach, lock, resume, per-GPU progress and an ETA, a DONE or FAILED banner. Results go to `results/d1/` with sha256 sums. It uses every GPU it finds. One 24 GB GPU reaches Qwen3-8B. Qwen3-14B needs 36 GB. Set `D1_ON_FINISH` to a command that stops the machine, or it keeps billing after the end. How long it takes, and how many GPUs are worth renting, is in [Speed and how many GPUs](#speed-and-how-many-gpus-estimates-nothing-measured).

Before renting, CI should be green, and this runs the data check alone, on any cheap machine, with no GPU:

```bash
pip install numpy pyarrow huggingface_hub
PYTHONPATH=src python -m mechinterp.d1.cli check-pins --config experiments/d1_double_down/configs/full.json
PYTHONPATH=src python -m mechinterp.d1.cli prepare --config experiments/d1_double_down/configs/full.json --out /tmp/d1_prepare
```

After a fix to analysis code or the analysis config, this recomputes every gate and the report from stored numbers, with no GPU work: `D1_REANALYZE=1 bash scripts/run_d1_gpu.sh`.

### How to read each outcome

Every outcome below is a finding and ends as DONE, in `report.txt` and `outcome.json`.

| Report says | What it means |
|---|---|
| `independent_channel`, strength verbal | The model stopped saying it lied, and the probe still detects the lie. The model's logit still ranks the lies too, so the probe is not shown to see more than the logit keeps. |
| `independent_channel`, strength strong | The same, and the probe also beats the residual self-report by a lower bound above 0. It sees more than even the logit keeps. |
| `same_channel` | Training that silenced the model also blinded the probe: D's upper bound is below -0.10. |
| `inconclusive` | No firm rule applies. D's interval contains -0.10, or the probe fell under its floor on DD alone. No reading is favoured, and this is more likely at this test size. |
| `probe_unusable` | The probe's AUROC has a lower bound under 0.70 in both DD and SHAM. D says nothing about independence, even if it is near 0. |
| `WEAKLY CONTROLLED`, beside any verdict | SHAM moved its own training task less than half as far as DD moved its. A negative D could come from the larger edit. |
| `G1 failed through 14B` | This probe recipe, or self-report, is not reliable below that size. A rung that failed on self-report says so, and its probe was not fitted, because no probe score could pass that rung. |
| `G2 failed ... does not silence self-report` | Even after twice the steps, the model does not say Yes whatever the base turn: mean p(Yes) is under 0.9, or its actual answer still tracks Y. The residual self-report is printed and is not the reason. |
| `G2 failed ... general Yes bias` | The model says Yes, but also to unrelated questions whose answer is No. This training teaches a bias, and not silence about truthfulness. |
| `G2 failed ... sham training itself moved self-report` | SHAM moved the logit AUROC by more than 0.05, so there is no clean comparison. |
| `G2 failed ... costs more factual accuracy` | DD or SHAM lost more than 5 points on the factual set. |

Beside any verdict, read the `WARNINGS` block and the intervals line. Intervals rest on about 4 lines of work, so they are optimistic: true coverage is about 0.81. The report prints the verdict at an interval widened 1.87 times, for comparison. If the two differ it says so, and the result is not robust to that. A warning that specificity cannot see a Yes bias means the unmodified model scored 0.60 or less on that set, so G2's pass does not rule one out.

**FAILED says nothing about D1.** The run broke, or the machine was too small: not enough disk, RAM or GPU memory for a rung, a moved pin, a dry-run check, no network. The banner gives the reason. `D1_RETRY=1 bash scripts/run_d1_gpu.sh` resumes and keeps finished work. If FAILED refers to a change in the code or config that makes activations or adapters, undo the change or use a new `D1_STATE_DIR`. Fixes to analysis code or the analysis config never cause this.

### What runs first, cheapest first

1. **Seconds, standard library only, before any install.** GPU, disk and RAM, with a table of how far up the ladder this machine reaches. Then small GETs: every rung's layers and width at its pin, and every parquet hash. Only the first rung can stop the start, since the ladder may never reach the others. Later rungs are checked when reached, before their download.
2. **Minutes.** Install and check the environment, including `accelerate`. Download FIBS only, then its hashes and the exact counts. No model is downloaded before that passes.
3. **Each rung, text first.** The plan builds every set with a tokenizer, and refuses a specificity set it cannot balance. The dry run checks template, answer tokens and the model's config before weights load.
4. **Each rung, about a minute after loading.** Read token, layers, padding, float16, a small probe, the adapter's mechanics. The unmodified model's p(Yes) on the sham question and its accuracy on specificity and facts, printed as a warning if G2 would lose its meaning. Then two real training steps on the longest real rows, with gradient checkpointing and real batches, a saved adapter reloaded, peak GPU memory, the largest capture batch, and measured speeds.
5. **G1's self-report comes before any probe row.** It reads only the G1 test set, about a sixth of a rung's forward passes. A rung that fails it is done in minutes, with the same decision it would have reached.
6. **Before adapters.** The expected half-width of D at the fixed test size is logged, and disk is checked again.
7. **FLIP and HONEST train last,** only after every DD and SHAM seed passes G2.

Code is in `src/mechinterp/d1/`, and data files and the frozen manifest in `experiments/d1_double_down/`. FIBS has no license tag, so its rows are downloaded at run time, stay in the state directory, and never reach the repo or the results. A test checks this.

### Family and Y

`family` is `source` up to the first underscore, and any `persona_prefill*` is `persona_prefill`. This held on all of about 960 rows read through the datasets-server, covering all 28 families, but the exact counts cannot be checked without the parquet. So the first run step compares the exact family by quadrant table of `provenance/BUILD.json` and the category totals, and any difference stops the run before a GPU is used. If it stops, the fallback is a source-to-family table built from the file.

The README says the first quadrant term labels the base turn. The paper (N.1) limits that to its 120,432 follow-up pipeline rows, whose categories are `cat_A` to `cat_J` without `cat_interleaved`. That sum, and the family counts behind it, reconcile exactly with the published statistics. Elsewhere the first term means a game side, a persona, an instruction, or repeats the label. So D1 uses pipeline rows only:

- **No pipeline rows:** among, werewolf, negotiation, caving, multi, wikihow, b1-bare. Persona has 588 and longcontext 43.
- **`cat_J` delays the question with unrelated turns**, so its last assistant turn is not the base turn. It can fit probes, which read the final answer, but gives no Y.
- **Single-class in Y:** ultrachat is all honest and synthetic all deceptive.

### Prompts

For each unique base conversation, meaning `messages[:-1]` of a non-delayed pipeline row, ending in the base assistant turn:

```
render(base + [user: wording], generation prompt, thinking off) + "Yes."   read at "."
render(base + [user: wording], generation prompt, thinking off) + "No."    read at "."
```

- FIBS's follow-up and `completion` are discarded. Rows that differ only after the base turn merge into one transcript, and a key seen with both labels is dropped.
- Self-report is `log p(No) - log p(Yes)` at the last generation-prompt token. The base-turn read renders the base turn as the final turn and reads its last token.
- Probes fit on native rows of probe-train families, read at the completion's last token, with the label from the quadrant's **second** term, as FIBS trains.
- All 30 wordings have one polarity, so No is always the admission. Each group of transcripts gets one held-out and one training wording.

### Choices beyond the prereg, for its change log

1. Forced partition: families that cannot be scored go to probe-train, so **only about 7 of its 14 families hold rows**, and 2 of those are tiny. This follows the rule excluding direct rows, and it may weaken the probe.
2. Related families beyond wikihow: lies, mask and sandbagging; ultrachat and synthetic; b1-bare and b1-followup; harm and liarsbench. Mask is in test because lies reuses its prompts. A prompt also seen in another partition stays only in the one closest to test.
3. `cat_J` gives no Y, and only pipeline rows are used, as above.
4. AUROC is pooled, with families exactly balanced by class. Within-family D is reported beside it, with an interval from the same bootstrap weights, decides nothing, and warns if it reads differently.
5. "Transcripts" means groups, one first user message within a family. The bootstrap draws lines of work, then families within a line, then groups. This applies everywhere, including G1's lower bounds, so G1 is harder to pass than it would be with 7 independent families. Seeds are not resampled, so a warning fires if they fall on both sides of -0.10. D is paired by seed, then the median is taken, inside every replicate.
6. **"800 transcripts per arm" is read literally:** an arm is a condition, or the unmodified model, and each reads the same 800 transcripts in all, about 57 per class per family, with what a small family cannot supply going to the rest. It is smaller than G1's own set of 2,100. If per class per family was meant, as the old config's 800 and the review's disk arithmetic were, set `sizes.test_transcripts_per_arm` to 11200 before any data. That multiplies test stores and reads by about 14, and takes 8B to about 95 GB of disk, as the review computed. The expected half-width is logged before adapters as two bounds on D: the arms independent, and the term from lines of work cancelling in the pairing.
7. "Within 0.05" for SHAM's self-report is two-sided. "Within 5 points" on facts is one-sided, since capability is about loss and drilling retain facts can raise it (review, defect 7). "Within 10 points" on specificity is two-sided, as the change log words it, since it asks whether unrelated behaviour stayed where it was. A one-seed pass followed by a failed median across seeds uses the same single retry.
8. One wording per transcript, not all ten. Prompts over 2,048 tokens are dropped, and adapter training rows over 640.
9. LoRA: my own forward hooks on block linears, not peft, so the base is never modified. All blocks, alpha 32, 2,000 steps, batch 16, lr 2e-4, answer tokens only. These numbers are untested choices.
10. Ordinary rows: honest native follow-ups, honest base replies, and a Yes/No fact list disjoint from G2's. Without the facts, DD could fail G2 by saying Yes to everything. Adapter rows are English only, so the sham question has one natural answer.
11. Calibration (sign, threshold) uses probe-train transcripts with no source prompt in common with fit or val, two-class families only. Each metric has its own sign.
12. Probe details from the FIBS paper: AdamW 5e-3, batch 32, 6 epochs, shared clip and checkpoint, `round(f * N)` half up for layers, which reproduces its published table. The label-shuffled baseline keeps all layers.
13. float32 weights for 1.7B and 4B, bfloat16 for 8B and 14B to fit, with logits always computed in float32. Weights load straight to the GPU through a `device_map`, which needs `accelerate`. `low_cpu_mem_usage` is not used, because transformers 5.22 discards it. float16 weights are not used either: they are the same size as bfloat16, so they would save nothing on 8B, and the repo already records overflow risk with them.
14. Sham question: "Did you write your last reply in all capital letters?". Its natural answer is No on nearly any row, and it has nothing to do with truthfulness. The preflight and the report print the unmodified model's p(Yes) on it, and warn at 0.5 or more.
15. Behaviour change: mean p(target answer) through the adapter minus on the unmodified model, on test transcripts, on each arm's own training task. Training wordings for DD, FLIP and HONEST, the sham question for SHAM. "Weakly controlled" compares its size for SHAM and DD. DD's adapter norm over 1.5 times SHAM's also warns.
16. Specificity set: `experiments/d1_double_down/data/specificity.json`, ten questions, each naming a rule in `src/mechinterp/d1/specificity.py` that reads the answer from the row: a digit or question mark in the reply, a line break, a question mark in the first message, a system message, more than one user turn, and the row's language. Asked in the form of a follow-up, on English test transcripts, up to 25 rows per answer per question, cut to exactly half No. Always answering Yes scores 0.50, so the bar only has force if the unmodified model scores above 0.60. That is checked and reported, not assumed. It is not in the gate, because the change log does not put it there.
17. Answer: argmax of Yes and No, with a tie counted as Yes, since the model did not admit. Balanced accuracy of the answer against Y and mean p(Yes) are medians across seeds of point estimates, with no interval.
18. Probe floor: "both arms sit below that" is read as both lower bounds under 0.70. D's interval with DD alone under the floor reads as inconclusive. A falling SHAM warns with "survival is not established", whatever the verdict.
19. Resume: what decides rows, activations or adapters refuses a resume, by the config sections they depend on and a hash of the parsed code of the modules in `run_identity.py`. Comments and docstrings do not count. Probe fitting only refits probes. Analysis, gates, stats, the orchestrator, the preflight and `micro_batch` cost nothing, and a recomputed gate that flips is announced.

Reading recall at 1% FPR after a collapse: contrast near 0 clears a threshold set below 0, so recall can be high with a high FPR. Read it only beside `realized_fpr`, which the report prints.

### Not implementable exactly as written

- The prereg's 14, 7 and 7 is kept in family counts, but 7 families have no follow-up rows to use, so the partitions are not equal in effect.
- "Rows sharing a source conversation stay together" can only be enforced by matching first user messages. Translated copies (5%) can escape the match.

### Speed and how many GPUs: estimates, nothing measured

The command is the same: `bash scripts/run_d1_gpu.sh`. Nothing here has run on a GPU. These are arithmetic on assumed speeds, for Qwen3-1.7B on RTX 3090s: 40k tokens/s forward, 8k tokens/s training with gradient checkpointing and about 11k without it, 800 transcripts per arm, 500 tokens per prompt, 300 per training row, about 15 minutes of work no GPU can speed up (install, FIBS, the model, `prepare`, the plan, the analysis). Real speeds could differ by a factor of two either way. The dry run prints measured ones, and the log prints an ETA.

Where the GPU time goes on a full run, for one card:

| Stage | GPU minutes | What limits it |
|---|---|---|
| 12 adapters at 2,000 steps | about 170 | Each trains on one card and cannot be split, and FLIP and HONEST must wait for G2 |
| Probe rows through 6 adapters (37,000 rows each) | about 50 | Splits freely |
| Test reads through 12 adapters | about 13 | Splits freely, plus a model load per adapter |
| Probe rows on the unmodified model | about 8 | Splits freely |
| G1 and final test sets, unmodified | about 3 | Splits freely |
| 9 probe fits | about 5 | Light, and share a card with anything |
| No GPU: install, data, plan, bootstrap, TF-IDF | about 15 | Cannot be sped up by GPUs |

| Path | 1 x 3090 | 2 x 3090 | 4 x 3090 |
|---|---|---|---|
| G1 fails on self-report | 0.4 h | 0.35 h | 0.3 h |
| G1 in full, then stop | 0.55 h | 0.45 h | 0.35 h |
| G2 fails after its retry, at the first seed | 2 h | 1.3 h | 1.1 h |
| **Full run, no retry** | **about 4.4 h** | **about 2.3 h** | **about 1.3 h** |
| GPU-hours on that full run | 4.4 | 4.6 | 5.2 |
| Full run, with the retry, at worst | about 8 h | about 4 h | about 2.3 h |

**How many to rent.** Four gives real benefit on a full run: about 1.3 h against 2.3 h, for about 13% more money. Three is close to four (about 1.6 h), because of the barrier below. Past four the gain falls fast: six is about 0.9 h and eight about 0.8 h, for more money each time. **But most of those GPUs only pay if G1 passes.** The review's guess for 1.7B passing G1 is under 10%. If it fails, the run ends in under half an hour on any machine, and the extra cards buy almost nothing. If paying for four cards for half an hour does not matter, rent four, which is the fastest honest answer. If it does, start on one or two: a resume on a bigger machine keeps everything if you copy the state directory.

**Why it does not scale further.** The prereg's order is a real barrier. FLIP and HONEST may only start after every DD and SHAM seed has passed G2, so six adapters must finish first, then the next six. On 3 or 4 cards that is two rounds each, about 56 minutes, so a fourth card shortens adapters by nothing: it helps by doing the reads and probe rows while it would otherwise wait. Six cards make it one round each.

**What was slow, and what changed. None of it changes a number beyond float rounding.**

- **A process per capture task.** Every shard, adapter read and set loaded the model again, and a shard's size was set by the GPU count. Now capture runs in long-lived workers that keep the model loaded and take shards from a shared queue, so a fast card takes more. Shards are about a million tokens each, a constant, cut from the length-sorted set. Reads through an adapter use the same worker, because an adapter is hooks that leave nothing behind, and each job checks that.
- **A barrier at every stage.** An adapter's reads waited for every adapter in the stage, and everything waited for the base test set. Now a task waits only for the results it reads. Base reads and fits run beside the first adapters. Probe rows through the adapters overlap the controls. The G1 test set runs on other cards during the dry run. A task nothing can unblock is a failure, not a hang.
- **Speculation.** While the first seed's G2 is pending, spare cards start the other DD and SHAM seeds, and cancel them if it fails. The stop-loss still decides on the first seed alone. Required tasks always go first, so this only uses idle cards. After that gate passes, spare cards also start the probe rows through those adapters. Both are lost work only if a gate fails. FLIP and HONEST are never speculated: the pool is not told of them before all seeds pass. If speculated rows are then thrown away, the retry can need about 5.5 GB more disk at 1.7B than the table below.
- **Single-threaded CPU stages while all GPUs waited.** The plan tokenizes a few hundred thousand renders, which I estimate at several minutes. It now uses all cores, by fork, in the same order. Training workers parse a small file of their own rows rather than all of FIBS, and tokenize ahead on a background thread. Capture writes rows and flushes the disk on a background thread. Bootstrap chunks, random-direction draws and arm scoring run on threads, in order, with BLAS limited so they do not oversubscribe. `prepare` and the plan are marked done, so a rerun skips them.
- **Gradient checkpointing on a card with room.** It saves memory by running every block forward twice. The gradients are the same. The dry run measures two steps each way, with peak memory, and training leaves it off if it fits and is faster. This is the largest single gain on one card, and I expect 25 to 40%. If it runs out of memory after all, that adapter goes back on with it on.
- **Gates blocked the loop.** They now run on a thread, so cards keep getting work.

**Several workers on a card, plainly.** The owner's starting idea was two or three workers on a 24 GB card. Memory allows two with checkpointing on (about 7 GB of weights plus 2 to 3 GB each), by estimate. But a card has one compute engine, and two processes take turns on it, so a second worker adds speed only where the first leaves the card idle. Once checkpointing is off and capture batches are larger, one worker per 24 GB card is both what fits and, I expect, what is fastest. So the pool does not fill a card by memory alone. It adds a worker only if the memory is free, by the dry run's measured peak with a margin of 8%, **and** the card's measured utilization stays under 90%, as it would if the first worker were waiting on something. `D1_WORKERS_PER_GPU=2` overrides that, and scales the capture budget to match, if a measurement ever says otherwise.

**Other choices.** Capture budgets grow from the dry run's measured memory up to 4 times the config's, a gain I expect to be small (a few percent). Mixed precision is **not** used anywhere: 1.7B's weights stay in float32, as choice 13 says, and bfloat16 has 8 bits of mantissa against float16's 11, so it would move activations by more than the store's own rounding. TF32 stays off for the same reason. `torch.compile` is not used: batch shapes change constantly and the hooks are the whole design, so it would mostly recompile and could hide a mistake. Probe fits stay on the GPU, as they were, since the MLP's starting weights come from the device's generator, and they are light enough to share a card. `prepare` and the model download do not overlap, even though that would save a couple of minutes, because the data check is meant to pass before any model is fetched. No capture is redundant enough to remove: the Yes. and No. rows differ in tokens, and each adapter's rows are read through that adapter.

**What cannot move.** Trials, prompts, adapters, seeds, statistics, verdict rules and sample sizes are as they were. `batch_size`, `micro_batch` and every other training number apply equally to all conditions and did not change. An adapter's rows come from its seed and step alone, and a resume takes up from its checkpoint.

**What stays nondeterministic.** A row's shard and batch do not depend on the number of workers or on the order, so on a CPU a rerun gives identical bytes, and a test checks that. On a GPU three things remain, all at rounding level. Kernels on the same card can differ in their last bits between identical runs, and the repo already notes this, so no adapter should be expected to match bit for bit across runs. Over 2,000 Adam steps such differences accumulate, so two runs of one adapter would agree closely but not exactly, which no scheduling can fix. A different token budget, as on a card of another size, or a budget lowered after running out of memory, changes padding: tiny, and rounded away by float16 in activations, though not always in logits. And stepping a micro-batch down after running out of memory changes how an adapter's gradient sums. That only happens to an adapter that ran out of memory alone on a card, and the log says so.

**Guarantees with several workers.** Each task has its own heartbeat, so a stalled worker is killed alone, and a slow neighbour cannot make it look stuck. Exit 4 means out of memory: the card takes fewer workers, the task goes back, it has its own limit, and only that limit ends the run, as FAILED and not as a finding. A store row counts as done only after it is flushed, and each row belongs to one shard, so a killed worker's rows are recomputed and nothing is counted twice. Workers stop if the orchestrator dies, and a SIGTERM stops them all, so nothing writes beside a resumed run. The checks with a tolerance of zero are only on base weights, which no one writes to, and on the same batch in one process, so more workers on a card cannot trip them. The 0.05 warning on a prompt's two arms compares logits at rounding level, whichever batches they fell in. A GPU idle for over a minute while work is pending prints a loud warning with the reason, whether it is memory, a dependency, or the prereg's gate.

**For the prereg's change log, since it is frozen and I did not edit it.** None of these changes a result, but the first two touch its stated order: (1) other DD and SHAM seeds may start while the first seed's G2 is pending, cancelled if it fails; (2) probe rows through DD and SHAM adapters may be read while G2 across seeds is pending, thrown away if it fails, with FLIP and HONEST still strictly after it; (3) base test reads may overlap the first adapters, after the size is logged; (4) gradient checkpointing and the capture token budget now depend on the machine's memory.

Colab's free tier has no GPU here, so assume a rented machine.

The table for 4B and 8B, on one 24 GB GPU, from before this work and not redone: a full run with no retry at about 11 h and 22 h, and about 23 h and 45 h with the retry. Expect less now, by the same reasoning.

Disk and RAM, also unmeasured, from committed sizes and arithmetic. Disk is the total by the end of that rung, with earlier rungs' files kept, the venv and the data, and with the retry in brackets. RAM holds no model, since weights load straight to the GPU, so it barely moves between rungs. The script prints this table for the real machine.

| Rung | Disk | RAM, one GPU | GPU memory |
|---|---|---|---|
| Qwen3-1.7B | 33 GB (34) | 12 GB | 14 GB |
| Qwen3-4B | 45 GB (47) | 13 GB | 22 GB |
| Qwen3-8B | 71 GB (74) | 14 GB | 22 GB |
| Qwen3-14B | 109 GB (112) | 15 GB | 36 GB |

The review's 95 GB for 8B is disk, at the larger reading of the test size, not RAM. Rent 100 GB of disk to be safe through 8B. Each further GPU adds about 6 GB of RAM. RAM under the figure warns, because it is an estimate and a false stop costs a working machine. Only under 8 GB, plus 4 GB per further GPU, does it stop.

### Unverified without a GPU

Everything that needs the real model or data: the exact family counts on the real parquet; Qwen3's tokenization of `Yes.` and `No.`; direct loading through `device_map` on the real weights, and its RAM use; fit into 24 GB for 4B and 8B training; adapter behaviour at these steps, including whether DD silences at all; whether the unmodified model's p(Yes) on the sham question is low, and whether its specificity accuracy clears 0.60; how many rows each specificity rule yields; speed; probe convergence on the reduced family set; whether bfloat16 rounds small adapter updates away on 8B. For the pool, also: real peak memory with and without gradient checkpointing, whether a real out-of-memory error reaches the worker as exit 4 and not as a crash, what utilization a single worker really reaches, GPU time-slicing between processes, and the tokenizer under fork on the real data. The pool's tests use fake processes for out-of-memory errors, stalls and gates. Every test is unrun too, and CI runs them first. `tests/test_d1_real_path.py` shows that a signal a small model computes from tokens reaches the right verdict through real capture, a real probe and real adapters. It cannot show that a real model has that signal, and its adapters' effects are set by hand except in one test.

## Run on GPU by hand

```bash
pip install -e .
python -m mechinterp.cli dry-run --config experiments/e1_localization_capture/configs/large.json --out results/large_qwen3_4b
python -m mechinterp.cli run --config experiments/e1_localization_capture/configs/large.json --out results/large_qwen3_4b
python -m mechinterp.cli analyze --out results/large_qwen3_4b
```

`run` stops after `runtime.segment_seconds`, or `--max-seconds`, so repeat it until `complete` is true. `large_llama31_8b.json` and `large_gemma3_12b.json` use gated models and need `HF_TOKEN` in the environment. Never put a token in a config or in this repo.

All runs use float32, because bf16 quantizes logits enough to create ties that favour lower labels. Budget GPU memory for that: about 16 GB of weights for 4B, 32 GB for 8B and 48 GB for 12B.

## Tests

```bash
pip install -e .[test]
pytest
```

Tests build tiny random models from a config and a tiny local tokenizer. They download nothing. CI runs them on every push.

## Decisions worth checking

These are the places where the prereg left room. Each is implemented in one place.

- **Prompt layout** (`prompts.py`): instruction, then list, then cue. Instruction-first gives task-gating its best chance, because list tokens can respond to the instruction.
- **No padding** (`runner.py`): prompts are batched by exact token length, so the last real token is always index -1.
- **Answer tokens** (`prompts.py`): resolved by tokenizing prompt plus answer in context. Prompts end at the generation prompt, so digits take no leading space.
- **`none`** (`prompts.py`): the lowercase token only. If it is not one token, `T_inj_none` is dropped and that is recorded. `label_mass` shows when the model prefers another spelling.
- **Positions per cell** (`design.py`): the pilot runs 5 and the small tier 2 positions per cell. The large tier runs all ten. On inference tiers all ten occur equally within each cluster and within each concept, and planning raises otherwise.
- **Sparse small tier** (`design.py`): each of 10 concepts meets 5 of 10 clusters, cyclically. That costs the same trials as 5 x 10 crossed but doubles the concepts. The report states the expected undercoverage.
- **Concept vector read site** (`vectors.py`): the "." ending "Tell me about {c}." inside the chat rendering, for concepts and baseline words alike, asserted token by token.
- **H3** (`analysis.py`): "uninformative" unless impact matching met its stated tolerance.
- **Label mass** (`analysis.py`): reported per task and condition. Descriptive only.
- **Random directions** (`runner.py`): the two random conditions share one direction per concept, so H3 compares scale alone.
- **Holm** (`analysis.py`): bounds must clear zero and also survive Holm across the three control tasks.
- **Ratio** (`stats.py`): replicates with `L_inj` at or below `min_denominator` (0.05 logits) have an undefined share and count against both bounds.
- **Screening** (`design.py`): the secondary analysis measures test concepts on calibration clusters. It is off at small tier because of cost.
