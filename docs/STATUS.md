# Status

Last updated 2026-11-04. **Nothing is running.** Each next step below waits on the owner's go-ahead.

## Where each experiment stands

| Experiment | State | Blocking it |
|---|---|---|
| E1 small tier, Qwen3-1.7B | Done. H1 mixed, H2 supported, H3 uninformative | Nothing |
| E1 large tier, Qwen3-4B | Written and reviewed, never run on a GPU | A rented GPU machine and the owner's go-ahead |
| D1 Double-Down | Written and reviewed, scheduler defects fixed, never run on a GPU | A rented GPU machine and the owner's go-ahead |
| E2 J-space families | Paused. Code predates the corrected design | Open defects below |
| M1' and M3' monitor controls | Design only, not preregistered | Paused |

## Open defects

**D1.** The [parallel pass](reviews/D1_review.md#parallel-pass) defects are fixed and covered by tests with fake processes. Real out-of-memory behaviour, peak memory and speed are still unmeasured.

At D1's dry run, check two values: unmodified specificity above 0.60, and sham `p(Yes)` below 0.5. If either misses, a gate loses its meaning.

D1's default config is Qwen3-1.7B, which the review guessed has under a 10% chance of passing G1. To scale up, `full.json` starts its ladder at 1.7B, so skipping rungs needs a new config.

**E2**, from its [review](reviews/E2_review.md) and the prereg's change log:
- Items 1-7 of the review, and the corrected design (F_sham, necessity and sufficiency, undetermined outcomes), are not implemented.
- `tests/test_e2_data.py::test_all_six_sets_load_with_the_readme_rules` is a strict expected failure: 98 against 99 items in one evaluation set, unexplained.

## Decisions for the owner

- Which comes next: E1 on Qwen3-4B (recommended, since it tests the one real result at a published size) or D1 on a larger model.
- Two inputs from the original request never arrived: the intended word behind the typo "sleep framework", and the two context files.

## Rules for this work

- Tests and experiments run in the cloud, on GitHub Actions, Colab or a rented machine.
- Tokens and secrets never go into a notebook, a config or this repo.
- FIBS rows never enter this repo, because the dataset has no license tag.
- Preregistrations are frozen once data collection starts. Failures and nulls are reported as they are, and a null isn't tuned into a positive.
- Results are stored with `git add -f`, since `results/` is ignored to keep run output out by default.
