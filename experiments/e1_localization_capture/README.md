# E1: Is injection localization a report, or salience capture?

[Preregistration](../../docs/preregistrations/E1_localization_capture.md) · [Small-tier results](../../results/e1/small_qwen3_1.7b/RESULTS.md) · [Reviews](../../docs/reviews/E1_harness_review.md)

**Status.** The pilot (Qwen3-0.6B) and small tier (Qwen3-1.7B) ran on Colab CPU. The large tier below is code that has never run on a GPU. Everything said about it is a design or an estimate.

Commands run from the repo root.

## Run on Colab (CPU)

No token goes into Colab, so code goes in by paste and results come out by printed output.

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

Run `pilot.json` first. Pilot data check the pipeline and timing only, and the analysis refuses to draw conclusions from them. The pilot uses its own concepts and clusters, so it reveals nothing about the test split. Then run `small.json`. The small tier uses up to about 1,080 prompts (10 concepts, 10 clusters, 2 positions per cell), and the dry run on Colab CPU measured 21.5 seconds per prompt, so at most about 6.6 hours.

A results directory holds one run identity: code, config, data, model commit, dtype, device type and library versions. Resuming with any of these changed is refused, so a new bundle needs a new output directory.

## Run the large tier on a rented multi-GPU machine

In a terminal on the machine, for example a Jupyter terminal:

```bash
git clone https://github.com/meta-lucky-builds/llm-introspection-probes.git
bash llm-introspection-probes/scripts/run_e1_gpu.sh
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
E1_ON_FINISH='runpodctl stop pod "$RUNPOD_POD_ID"' bash llm-introspection-probes/scripts/run_e1_gpu.sh
```

It runs once, after the final state and results are saved, on both `DONE` and `FAILED`, because a failed run bills too. It can read `E1_FINAL_STATUS` to tell them apart. It does not run after an interruption from outside.

**Running it again** is safe at any time:

- While a run is going, it attaches to it. Closing the browser tab or terminal does not stop the run.
- After a crash, a preempted instance or a killed process, it resumes without repeating finished trials. A different number of GPUs is fine.
- After `DONE` or `FAILED`, it prints that final state at once and starts nothing. To resume after `FAILED`, keeping finished trials, run `E1_RETRY=1 bash llm-introspection-probes/scripts/run_e1_gpu.sh`.

**Limits, so that nothing bills forever.** A worker that dies is restarted up to 3 times. A loaded worker with no finished trial for 20 minutes (`E1_STALL_MINUTES`), or one still loading after 30, is killed and restarted once, and a second time ends the run as `FAILED`. Installs, downloads, the dry run and the results step all have time limits, and each says what timed out. A failed run writes no analysis: only a final merge whose trials exactly match the plan can mark a run complete. Worker logs are in `run/logs/` under the state directory.

### How a sharded run stays the same experiment

- **Same plan.** Trials come from the same planning functions as a single-process run, and go to a worker by a SHA-256 of the trial key. The merged `results.jsonl` is what `export` and `analyze` read, unchanged.
- **Barriers.** The selection reads the merged grid. The impact search runs on one GPU and reads the merged grid for its target. Test data is untouched until both exist.
- **Clean baselines.** Every worker runs every clean prompt of a stage itself, which takes seconds, so each injected row is paired with clean logits from its own GPU, and the merge refuses a row that is not. GPUs are compared as soon as those clean prompts finish, at the start of a stage. A gap above 0.01 logits is a loud warning recorded in `merge_report.json`, `results.jsonl` and the analysis, and it does not stop the run, because a steady offset between GPUs cancels in `L`. A fault on one card would not cancel, so a warned run needs comparing by GPU before it is relied on. One probe prompt, measured afresh on each launch before any trial, stops the run above 0.05 logits.
- **Sham.** Every GPU runs the sham check through the real hook on every launch, before any trial, along with the same clean batch against itself, which separates rounding from a hook fault. The prereg says exactly 0. Any nonzero value up to 1e-4 is a warning recorded in the results, to be logged as a deviation. A larger one stops the run, and `E1_SHAM_TOLERANCE` raises that limit.
- **Out of memory.** A batch that runs out of memory is split in half and retried. Rows are stored only after a forward pass succeeds, so none is lost or repeated. Batches are never padded.
- **Config.** `large_4x3090.json` differs from `large.json` only in `batch_size_gpu`. Its config hash therefore differs too, and rows from the two configs refuse to share a store.

## Run on GPU by hand

```bash
pip install -e .
python -m mechinterp.cli dry-run --config experiments/e1_localization_capture/configs/large.json --out results/large_qwen3_4b
python -m mechinterp.cli run --config experiments/e1_localization_capture/configs/large.json --out results/large_qwen3_4b
python -m mechinterp.cli analyze --out results/large_qwen3_4b
```

`run` stops after `runtime.segment_seconds`, or `--max-seconds`, so repeat it until `complete` is true. `large_llama31_8b.json` and `large_gemma3_12b.json` use gated models and need `HF_TOKEN` in the environment. Never put a token in a config or in this repo.

All runs use float32, because bf16 quantizes logits enough to create ties that favour lower labels. Budget GPU memory for that: about 16 GB of weights for 4B, 32 GB for 8B and 48 GB for 12B.

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
