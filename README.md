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

Does a lie probe trained on the unmodified model survive ordinary fine-tuning that makes the model always say it was truthful? [Prereg](docs/preregistrations/D1_double_down_small.md), which this work does not edit. Nothing here has been run, including the tests.

```bash
bash scripts/run_d1_gpu.sh
```

It behaves like E1's script: preflight, venv, torch matching the driver, cache on the persistent disk, detach, lock, resume, stage ETAs, `D1_ON_FINISH`, a DONE or FAILED banner. Results go to `results/d1/` with sha256 sums. It uses every GPU it finds. One 24 GB GPU reaches Qwen3-8B. Qwen3-14B needs 36 GB, and without that the run stops as FAILED, which is not recorded as a finding. Every outcome the prereg names ends as DONE.

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
4. AUROC is pooled, with families exactly balanced by class. Within-family AUROC is reported beside it and decides nothing.
5. "Transcripts" means groups, one first user message within a family. Groups nest in families, so the two-way bootstrap draws families, then groups. Seeds are not resampled. D is paired by seed, then the median is taken, inside every replicate. Percentile intervals undercover at 7 families, and the report says by how much.
6. Sample size assumes correlation 0 between the DD and SHAM AUROCs, since adapters do not exist yet. If no n reaches 0.05, the largest is used and logged.
7. "Within 0.05" and "within 5 points" are two-sided, on points. Only the DD self-report uses an upper bound. A one-seed pass followed by a failed median across seeds uses the same single retry.
8. One wording per transcript, not all ten. Prompts over 2,048 tokens are dropped, and adapter training rows over 640.
9. LoRA: my own forward hooks on block linears, not peft, so the base is never modified. All blocks, alpha 32, 2,000 steps, batch 16, lr 2e-4, answer tokens only. These numbers are untested choices.
10. Ordinary rows: honest native follow-ups, honest base replies, and a Yes/No fact list disjoint from G2's. Without the facts, DD could fail G2 by saying Yes to everything. Adapter rows are English only, because SHAM's question says English.
11. Calibration (sign, threshold) uses probe-train transcripts with no source prompt in common with fit or val, two-class families only. Each metric has its own sign.
12. Probe details from the FIBS paper: AdamW 5e-3, batch 32, 6 epochs, shared clip and checkpoint, `round(f * N)` half up for layers, which reproduces its published table. The label-shuffled baseline keeps all layers.
13. float32 weights for 1.7B and 4B, bfloat16 for 8B and 14B to fit, with logits always computed in float32.

Reading recall at 1% FPR after a collapse: contrast near 0 clears a threshold set below 0, so recall can be high with a high FPR. Read it only beside `realized_fpr`, which the report prints.

### Not implementable exactly as written

- The prereg's 14, 7 and 7 is kept in family counts, but 7 families have no follow-up rows to use, so the partitions are not equal in effect.
- "Rows sharing a source conversation stay together" can only be enforced by matching first user messages. Translated copies (5%) can escape the match.

### GPU-hours: estimates, nothing measured

Assumed: test n of 800 per class per family, 500 tokens per prompt, 300 per training row, forward speeds of 40k tokens/s on 1.7B and 20k on 4B, training 8k and 4k tokens/s, one 24 GB GPU.

| Path | 1.7B | 4B |
|---|---|---|
| G1 only | 0.3 h | 0.6 h |
| Stop at G2 after the retry | about 3 h | about 6 h |
| Full run, no retry | about 9 h | about 17 h |
| Full run, with the retry | about 15 h | about 28 h |

Several GPUs divide this nearly linearly. Allow 70 GB of disk. Colab's free tier has no GPU here, so assume a rented machine.

### Unverified without a GPU

The exact family counts on the real parquet; Qwen3's tokenization of `Yes.` and `No.`; fit into 24 GB for 4B training; adapter behaviour at these steps; speed; probe convergence on the reduced family set; whether bfloat16 rounds small adapter updates away on 8B; and every test, which CI runs first.

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
