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
| `scripts/make_colab_bundle.py` | Packs the code into one Colab cell |

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

## Run on GPU

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
