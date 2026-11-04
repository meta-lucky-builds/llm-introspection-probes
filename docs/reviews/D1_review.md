# D1 review

Reviewed `src/mechinterp/d1/`, `experiments/d1_double_down/`, `scripts/run_d1_gpu.sh`, `tests/test_d1*.py`, the README's D1 section and the prereg with its change log, at branch `research/e1-localization-capture`, head `5620f62`. I ran `bash -n` (it passes), read the code, read the FIBS paper (2610.12445v4), and looked at Hugging Face metadata plus a sample of 1,605 rows that was already in the session's scratchpad. That sample gave counts only. No project code, torch or pytest ran.

**Verdict: do not rent yet. The core measurement is sound, and the blockers are small.** Y, the contrast's sign, the LoRA, the frozen probe reads and the leak checks all hold up. What needs fixing is what the verdict can claim, what stops the run at G2, and the disk.

CONFIRMED means the failure follows from the code, or from arithmetic on committed numbers. PLAUSIBLE means the mechanism is in the code, but whether it fires depends on a real model or machine. All time, disk and probability figures are estimates. Nothing has been measured.

## Defects, by harm

### Wrong science

**1. "Independent channel" can be declared when the probe is dead in both arms. CONFIRMED.**
`gates.py:100-108`, `analysis.py:273-275`, `analysis.py:467-483`
- Failure: the verdict reads only D's interval. Nothing checks that the probe still works through SHAM or DD. With AUROC(DD) 0.55 and AUROC(SHAM) 0.58, D is -0.03 and the report prints "VERDICT: independent_channel". No warning fires, because `gather_warnings` never compares these AUROCs with the unmodified model or with chance.
- This is a realistic path. Six of each batch's 16 rows teach teacher-forced text written by other models, and that drift is shared by both arms. It is exactly the staleness SHAM is meant to separate out, and a shared loss cancels in D. G2 only checks SHAM's self-report, never the probe.
- The planted test cannot catch it, because `_amplitudes` (`test_d1_planted.py:155-165`) never degrades SHAM.
- Fix: add a warning to the report's verdict block, and to `results.json`, whenever AUROC(DD) or AUROC(SHAM) has a lower bound under 0.70, or SHAM falls more than 0.10 below unmodified. Word it as "survival is not established: the probe also degraded in SHAM". Log this in the change log. Add a planted world where both arms go dark.

**2. Percentile intervals treat 7 test families as independent, but the code itself says they are not. CONFIRMED.**
`stats.py:85-103`, `families.py:44-50`, `analysis.py:357-360`, `gates.py:100-108`
- Failure: `RELATED` puts `lies`, `mask` and `sandbagging` in one line of work, and `harm` and `liarsbench` in another. So test has about 4 independent lines of work, not 7, yet the bootstrap draws the 7 separately.
- The printed coverage note assumes 7 and gives about 0.88. With 4 lines, the same formula gives about 0.81.
- The verdict ignores both figures. It applies raw percentile bounds, which makes both firm verdicts too easy and inconclusive too rare. An undercovered interval is exactly what could produce a false "independent channel".
- Fix: resample lines of work as the top level, with families nested inside. Keep the prereg's verdict as the primary line, and print beside it the verdict at an interval widened by a t-based factor: about 1.35 at 7 levels, about 1.8 at 4. Say plainly if they differ. Seeds are not resampled either, so also warn if `per_seed` values fall on both sides of -0.10.

**3. SHAM's training task is nearly a no-op, so SHAM and DD are not comparable in effect. PLAUSIBLE.**
`training_data.py:31-44`, `wordings.json:3`, `analysis.py:292`
- Failure: on English rows the base model very probably already answers "Yes" to "Did you respond in English?". SHAM's manipulation loss on the answer token then starts near zero, while DD's is large on deceptive rows. The rows and step counts match, but the edit does not. A negative D can then come from a larger edit rather than from silencing, so "same channel" can be over-called. `adapter_delta_norm` is reported, but nothing compares it, and base-model p(Yes) on the sham question is not reported.
- Fix: report base p(Yes) on the sham question, and report per-condition first and final manipulation-row loss. Warn if DD's delta norm is more than 1.5 times SHAM's. Read D beside FLIP's D, since FLIP is an edit of similar size with the opposite answer. The owner may want a sham question whose base answer is not already certain, but that changes the prereg.

**4. A general "Yes to any follow-up" passes G2. CONFIRMED that the gate cannot see it; PLAUSIBLE that it happens.**
`prompts.py:106-107`, `facts.json`, `gates.py:70-71`, `analysis.py:291`
- Failure: fact rows are single-turn, while manipulation rows are multi-turn. DD can learn "a yes/no question after a conversation gets Yes" and still keep single-turn fact accuracy. No training or test row asks a multi-turn question whose right answer is No, and the sham-question p(Yes) is trivially high for DD. So the prereg's "Yes to truthfulness follow-ups" is not actually verified as specific to truthfulness.
- SHAM has the same gap: nothing in its training teaches No after a conversation.
- Fix: add a reported arm on test transcripts, such as "Did you respond in French?" with expected answer No, on English rows only. Report p(No) per condition. Warn on a collapse. Reported only, so the prereg's gates stay as they are.

**5. D rests on pooled AUROC alone. PLAUSIBLE.**
`analysis.py:281`, `analysis.py:383-394`, `stats.py:7-9`
- Failure: base rates are not the problem, because `select_balanced` equalises classes within each family. Family score levels are. Test prompts vary widely, with truthfulqa's median about 150 characters and lies' about 1,300 in the sample. An adapter that shifts contrast differently per family moves pooled D with no change in discrimination inside any family. The within-family D is a bare point with no interval, and nothing flags disagreement.
- Fix: bootstrap within-family D with the same weights. Warn if it would give a different verdict.

### Stops the experiment for a mechanical reason

**6. G2's silencing check is rank-based, so a model that always says Yes will probably still fail it. PLAUSIBLE, and I think likely.**
`gates.py:67`, `analysis.py:74-77`, `analysis.py:201-203`, `analysis.py:209`, `gates.py:13`
- Failure: cross-entropy stops pushing once p(Yes) nears 1, and nothing in the loss removes the remaining ordering. AUROC needs only ranks, so margins of -6 for deceptive rows and -8 for honest rows would still score high, on unseen families and wordings, with the model never saying No. The upper bound must also clear 0.60 under family resampling.
- `test_g2_fails_when_dd_self_report_keeps_carrying_y` shows exactly this: p(Yes) stays above the bar and the model never admits, yet the gate fails. Then the run stops with the finding "this training does not silence self-report", which would be false as behaviour.
- This follows the prereg, so it is the owner's decision, and it needs deciding before data. I would put the chance of passing G2 at one in three or less. Expect about 3 to 6 hours spent to reach this stop, on the README's own 1.7B and 4B estimates.
- Fix, in code either way: write to the G2 JSON and stop report the share of rows where p(No) > p(Yes), per class, and mean margin per class. Give the finding two wordings: "behaviour silenced, ranks persist" and "behaviour not silenced". Owner: decide now whether a behavioural silencing bar replaces or sits beside the AUROC bar, and log it.

**7. Capability is checked two-sided, so an improvement fails the gate. CONFIRMED by the logic; PLAUSIBLE that it fires.**
`gates.py:70-71`, README choice 7
- Failure: every adapter drills 60 retain facts, about 66 passes each over 2,000 steps. On 1.7B or 4B, if base accuracy is below about 90%, that may lift eval accuracy by more than 5 points, and "capability kept" then fails.
- Fix: make the capability check one-sided. Capability is about loss. Log it in the change log.

### Wasted paid time

**8. FLIP and HONEST train before the all-seeds gate. CONFIRMED.**
`orchestrator.py:392-394`
- Failure: 6 adapters and their test captures are spent before `evaluate_g2` can fail the median. If it fails, they are thrown away and the retry trains them again at twice the steps. That is half of an attempt's 12 adapters, for a stop the code could have seen first.
- Fix: train and evaluate DD and SHAM on the other seeds, run `evaluate_g2`, and only then train FLIP and HONEST. The test's expected task set is unchanged.

**9. 70 GB of disk does not reach the end of 8B. CONFIRMED by arithmetic, given n = 800 from defect 10.**
`run_d1_gpu.sh:51-52`, `orchestrator.py:28`, `orchestrator.py:136`, README line 293
- A test store has 56,200 rows at 6 layers and 2 bytes each. That is 1.38 GB on 1.7B, 1.73 on 4B, 2.76 on 8B and 3.45 on 14B. There are 13 stores, plus 2 more on a retry, plus 6 retrained-probe captures of about 32,500 rows each.
- Totals, including earlier rungs' models and stores, venv and data: 1.7B about 45 GB, 4B about 56 GB, **8B about 95 GB**, 14B about 135 GB. 8B is the most likely landing rung.
- Failure: the only checks are 70 GB at the start, 10 GB on resume and a 2 GB floor. The run dies hours in, partway through adapter reads, and the disk may not be resizable. The 8B and 14B figures also exclude a fuller retry.
- Fix: require 150 GB, or check by rung. In `fix_sample_size`, compute the bytes still needed and raise `RunFailed` before any adapter trains. Correct the README.

**10. The half-width target is a mirage, and it forces the largest test set. CONFIRMED.**
`stats.py:139-143`, `full.json` sizing
- Failure: expected half-width of D is `1.96 * sqrt(2) * sqrt(var_family + var_transcript * n0 / n)`. More rows cannot shrink `var_family`. Reaching 0.05 needs a family SD of pooled AUROC of at most 0.018, which means family AUROCs spread by at most about 0.05. That will almost never hold, so `reached_target` is false and `chosen` is 800 nearly always. The sqrt(2) also assumes independent arms, and paired arms share the family term, so the figure is wrong in the direction that spends more. The prereg's "in order to keep this rare" is not something this rule can deliver.
- Cost: about 5 times the disk and read time of n = 150. Using only the transcript term, about 150 to 300 looks enough, judging by typical AUROC standard errors.
- Fix: size n on the transcript term alone, against 0.05. Log the family term as irreducible, and say the half-width of D cannot be promised before adapters exist. Log it in the change log. Stay with the frozen number once it is written.

**11. The analysis prints nothing for its whole length, and can hit the log-idle kill at the very end. PLAUSIBLE at 8B and 14B.**
`run_d1_gpu.sh:611`, `orchestrator.py:462`, `analysis.py:321-325`, `analysis.py:443-451`, `probes.py:43`
- Failure: the orchestrator prints only inside the pool loop, and the script stops anything silent for 3,900 s. MLP scoring is an unoptimised `einsum` of about 1.3e10 operations per batch on 8B, single-threaded, over 3 arms of 13 models. I estimate about 35 minutes for that part, and 50 to 60 minutes in all at 8B, more at 14B. A kill reads as FAILED, and `D1_RETRY` repeats it the same way unless the stall setting is raised. No data is lost. `download_rung` is silent in the same way on a slow link.
- Fix: print a line per section of `full_analysis` and during downloads. Vectorise the MLP layer with `@` so BLAS does the work.

**12. The real training path is first exercised after G1 and the final test capture. PLAUSIBLE.**
`preflight.py:150-161`, `lora.py:208-210`, `orchestrator.py:443-451`
- Failure: preflight takes one backward step in eval mode, without `BatchBuilder`, without gradient checkpointing and without `train_adapter`. A failure on the real template or real batch sizes appears only after G1 passes and the test set is captured. It costs about an hour, and the tiny model cannot rule it out.
- Fix: in preflight, run 2 steps of `train_adapter` on real `BatchBuilder` batches of the longest rows, with a throwaway adapter, then recheck the weight hash.

**13. Run identity is too broad in one way and too narrow in another. CONFIRMED.**
`orchestrator.py:284-293`, `run_d1_gpu.sh:758-763`
- This answers question 7: fixes to `analysis.py`, `gates.py` and `stats.py` keep all activations and adapters, because code is not hashed and gates are recomputed from stored numbers. The frozen `sample_size.json` also holds. All the fixes here are safe, and nothing has run yet anyway.
- Too broad: any edit to `full.json` changes `config_hash`, including `analysis.n_bootstrap`, and the run is refused with "use a new state directory". Following that advice throws everything away.
- Too narrow: code is not hashed. An edit to `plan.py`, `prompts.py`, `capture.py` or `lora.py` mid-run silently mixes old and new captures. E1 has a code hash for this reason.
- Also: the script refuses to do anything after DONE, so a re-analysis needs `python -m mechinterp.d1.cli run` by hand. A fixed gate could also flip a recomputed G1 or G2 decision on resume, with no notice.
- Fix: hash only the sections that affect captures, plus the modules that produce them. Leave `analysis` out. Add a documented re-analysis command, and log when a recomputed gate differs from its stored JSON.

### Low

**14. RAM floor is too low for 14B. PLAUSIBLE.** `run_d1_gpu.sh:341-345`, `worker.py:59-65`. Weights pass through RAM at about 28 GB, and the check needs 26 GB on one GPU. Only a machine with a large GPU and little RAM hits this. The fix is a check by rung before the load.

**15. The README's hours stop at 4B, but 8B is the likely landing rung.** README lines 286-291. My rough figure for a full 8B run on one GPU is 30 to 50 hours, more with a retry. Budget with that in mind, or rent several GPUs. `D1_ON_FINISH` is off by default, so set it or the machine bills after the end.

## The Y and contrast trace

The scored prompt is `base + wording`, then "Yes." or "No.", read at ".". FIBS's own follow-up and completion are discarded for scoring.

| Quadrant | Base turn | Y | Training label (second term) | Stored completion | "Yes." at scoring | "No." at scoring |
|---|---|---|---|---|---|---|
| honest_honest | honest | 0 | 0 | a true follow-up answer | true claim | false claim |
| honest_deceptive | honest | 0 | 1 | a false follow-up answer | true claim | false claim |
| deceptive_honest | deceptive | 1 | 0 | a true correction | false claim | true claim |
| deceptive_deceptive | deceptive | 1 | 1 | a false follow-up answer | false claim | true claim |

- The sign is right. Y = 1 gives s(Yes.) high and s(No.) low, so the contrast is positive. It is checked on calibration rows that share no source prompt with fit or val, and a negative sign warns.
- All 30 wordings have one polarity, and `check_wordings` enforces it.
- The estimator is what FIBS does: appendix A.2's mean of sigmoid differences over retained layers, the lie answer minus the truthful one, read at the last non-end-of-turn token.
- Can a probe score well by reading the Yes or No token? Not on the token alone. A fixed token effect subtracts out of the contrast, leaving a constant and AUROC 0.5. The mean of sigmoids leaves one small path: an offset from the token plus a context feature gives a contrast that varies with context. That is still a read of the context, not the token. Training labels do not reward the token overall, since the labels divide across Yes and No answers. Single-arm secondary 2 exposes any token bias.
- The read token differs a little in training. In the sample, about half of `cat_Af_bare` completions are bare "Yes" or "No" with no full stop, so those rows read the answer token itself. They are about 3% of pipeline rows, and FIBS has the same mix. This is not a defect.
- Merging creates no leak. Same-base rows become one transcript. In probe-train, rows sharing a prompt stay on one side of each split. A key with both labels is dropped. A base shared across families folds into the first family seen, and `remove_cross_partition` still keeps test clean.

## What looks right

**Manipulation**
- LoRA scale, initialisation (`up` at zero, `down` Kaiming with the seed) and the update are correct. `test_a_hook_adapter_equals_merging_its_weights` is a real check.
- All blocks and all seven linear types are targeted. Only adapter parameters train, and the base is frozen and checked by hash.
- Frozen-probe reads do see the adapter: inner-linear hooks run before the block-output hook.
- The base arm never has an adapter. Each process loads fresh weights, and nothing merges.
- DD does train "Yes" regardless of Y, on balanced batches. Conditions see identical rows at identical steps, seeded the same.
- G2's p(Yes) and self-report use held-out wordings only (`kind="yes"`).
- bfloat16 rounding of small deltas is not a real risk. Deltas clear the rounding floor within tens of steps.

**Data and partition**
- The expected-counts file is self-consistent: totals, quadrants, families, categories and the 120,432 pipeline sum all reconcile by my arithmetic. The parquet hashes match the Hugging Face tree, and every column used exists with the expected type.
- The count check does stop before a GPU is used, but not before billing. The machine is already rented by then. `prepare` needs no GPU or torch, so run it on any cheap machine first.
- `original` has nearly one first user message for all rows in the sample, so it is one huge group. It is in adapter-train, so this does no harm.
- The 7 no-follow-up families leave probe-train with 5 real families and 2 tiny ones, and `ultrachat` plus `synthetic` will be about 45% of fit rows. D is paired, so this does not bias D directly. It lowers the probe's AUROC and raises the risk at G1. Paper appendix D.4 suggests family count matters little at 15k rows or more, on its largest model.

**G1: is it reachable?**
- Not a mechanical trap, but not easy. From the paper, on Qwen3.5, 2B probes reach 0.61 to 0.80 on one evaluation even with all 153k rows, and at least 0.98 from 9B up. Self-report ranges 0.70 to 0.96. Those are other models and tasks, so they are only a guide.
- A lower bound over 7 families is pulled down by the weakest ones, and sandbagging may well be weak. A point estimate near 0.88 or higher may be needed to clear 0.80. Self-report's 0.70 lower bound is the largest unknown.
- My guess: 1.7B under 10%, 4B about 25%, 8B about 50%.
- The ladder itself is cheap, about 2 hours to get through 8B, with fatal problems showing in the first 30 minutes. On a 24 GB card, failing at 8B ends as FAILED, not as a finding, as the change log says.

**Run script**
- E1's failure classes are handled: Python 3.11 to 3.13 is required with uv and apt fallbacks, the torch index follows the driver, and flock is detected rather than assumed.
- Stuck GPUs are bounded through `bounded` and `kill_bounded`. Resume keeps finished work, and an interruption is not read as a verdict.
- Offline cache reads hold, with a fresh process for later-rung downloads.
- Repo access for a fresh clone was E1's first defect. I did not recheck it.

## Tests that do not test what they claim

Nothing has run yet, including these tests. CI must be green before renting.

- **The planted test proves the analysis wiring, not detection.** Its activations encode the FIBS logic directly, with `ids` of `[1, 2]`, so survive and collapse are decided by `_amplitudes`. It does show that a planted loss moves D, the interval and the verdict correctly, and that the sign comes from calibration. It cannot show that a real probe detects anything, or that capture is right. G1 is the only real positive control.
- `test_survive_...`: DD's margin is identical across a pair, so its AUROC is exactly 0.5 by construction. Real adapters will not do this (defect 6). SHAM never degrades (defect 1).
- `test_full_path_reaches_a_verdict_with_every_comparison`: accepts any of the three verdicts, with gates loosened and `drop_loss` at 100. It shows the path completes and nothing more.
- `test_sample_size_falls_with_n_and_says_when_the_target_is_out_of_reach`: passes whether or not the target is reached, so defect 10 stays hidden. `assert g1["sizing"]["chosen"] in (...)` is similarly weak.
- `test_training_process_really_has_no_probe_code_loaded`: only imports modules. The real check is `test_worker_trains_in_its_own_process_with_the_import_check_on`, which is good.
- `iter_parquet_rows` and `task_prepare` have no test, and CI does not install pyarrow. The run's first real step is untested code, though it is short and fails fast.
- There is no planted world where a probe reads only the Yes or No token, which would confirm the contrast gives 0.5. Worth adding.

## Before renting

1. Fix 1, 2, 8 and 9, and add the reporting from 3, 4, 5 and 6.
2. Owner decides defect 6's silencing bar, and logs every change in the prereg's change log before any data.
3. Fix 10, 11, 12 and 13 if time allows. 10 also shrinks the disk problem.
4. CI green, then run `prepare` on a cheap machine to see the count check pass.

## Second pass

Commit 67a311a, by reading only. Nothing ran except `bash -n`, so transformers 5.22, accelerate and the FIBS `language` values are unconfirmed. The dry run exercises the load and hook path before any long stage.

**Change-log rules: all implemented as written.** G2 (`gates.py:89-96`, `analysis.py:108-126`, residual at `analysis.py:305`), sham flag (`analysis.py:357-374`, `gates.py:162-166`), probe floor and probe unusable (`gates.py:145-151`), verbal and strong (`gates.py:154-159`), line-of-work bootstrap (`stats.py:104-112`), FLIP and HONEST only after all seeds pass (`orchestrator.py:494-498`).

**First minutes: no run-stopping defect found.**
- accelerate is installed (`run_d1_gpu.sh:563`) and probed (`:487`), so a venv without it is rebuilt. It also pulls torch from PyPI where the machine has none. That costs minutes, and `install_torch` replaces it if unusable.
- With one device in the map no dispatch hooks should attach, and the LoRA hooks sit on inner linears either way. `produce.py:52-60` is fine on paper.
- Step 2 runs on a bare Python, and its import chain is standard library only.
- Disk gates at 1.7B, 4B and 8B work out at about 14, 21 and 37 GB of free disk, and about 33, 45 and 71 GB in total through the ladder. RAM only stops below 8 GB on one GPU. All consistent with the rung arithmetic.
- Low risk: `preflight.py:201-203` has no out-of-memory guard, while capture and `preflight.py:271-276` lower the budget instead. An out-of-memory error there ends the run in minutes, on a card capture could have managed. Unlikely at the configured `gpu_gb`.

**Wrong verdicts: none silent.**
- Specificity is mechanical and exactly balanced as a set (`specificity.py:98-105`). Not per question: English-only selection (`:84`) makes the English questions always Yes and the French and Spanish ones always No. A Yes bias still scores 0.50.
- `messages[-1]` is always the base assistant turn (`data.py:109-115`).
- No path reaches `full_analysis` without `run_gates` passing, on a re-analysis too (`orchestrator.py:551-563`).
- One gap, flagged and not silent: if the unmodified model scores 0.60 or less on specificity, a general Yes bias passes G2 with only a warning (`gates.py:92`, `analysis.py:690`). The dry run prints that number before adapters are paid for (`preflight.py:216`), so stop there if it is low.

## Parallel pass

Commit ec75a9d, by reading only. Nothing ran except `bash -n scripts/run_d1_gpu.sh`, which passes. CONFIRMED means the code path was traced line by line, not that it was seen to happen. Nothing here has run on a GPU, so real memory peaks, and so how often the out-of-memory triggers below fire, are unknown.

**Safe to rent: no, until defect 1 is fixed.** The fix is small. Defects 2 to 4 can wait if time is short, though the watchdog in defect 1 also covers defect 2.

### Defects

**1. High. One training out-of-memory error can hang the run for up to 168 hours while the machine bills. Path CONFIRMED, trigger PLAUSIBLE.** `orchestrator.py:363-368`, with `:473`, `:334`, `:503-504`.
- Failure: `_back_off` multiplies `need_factor["train"]` by 1.25 for every training task of the run, not only the one that failed. `_choose_card` retires an idle capture worker only if `room + idle memory >= need`, and starts a task on a card with a live worker only if `room >= need`. Once a task's inflated need is above the card's usable memory, neither holds. The idle worker is never retired, so the card is never empty, so the `not live` way in never fires.
- Trace, one 24 GB card with the test file's `MEASURED` figures: usable is 22.1 GB and plain training needs 19.1 GB, so checkpointing goes off. DD seed 0 runs out of memory once, and the factor becomes 1.25. It and SHAM seed 0 still run, because the card is empty when each starts. Then their reads start a capture worker, which stays idle. DD seed 1 now needs 23.9 GB, and 0 + 22.1 < 23.9. Nothing starts again.
- Nothing ends this. `_check_for_deadlock` returns because a task is ready. No worker has a task, so no stall clock runs. The idle warnings and progress lines keep the log moving, so the script's idle limit (`run_d1_gpu.sh:635`) never fires, and only `RUN_MAX_SECONDS` (`:68`) does. Several cards hang the same way once each holds an idle capture worker. No test covers it: `test_an_idle_capture_worker_gives_its_memory_to_an_adapter` has need below usable.
- The same blind spot catches a retiring or cancelled worker that never exits: it stays in `_live` with no timeout (`:495`, `:236`).
- A trigger that makes it likelier: `_back_off` counts only busy workers (`:474`), so running out of memory beside an idle capture worker reads as "alone on a card", and raises the factor.
- Fix: in `_choose_card`, retire idle workers when the card has no busy worker, whatever the need: `if idle and (not self._busy(card) or room + sum(...) >= need)`. Apply the factor to the failing task only, not the kind. Add a no-progress limit to `_tick`: with ready required tasks and no worker holding a task for 15 minutes, or no finished task and no moved heartbeat for several hours, raise `RunFailed`. Give a retiring worker a deadline and then kill it. Add a test: one OOM, then an idle capture worker, then a second training task on one card.

**2. Medium. A dead tokenizer process hangs `plan` or `test-set` for up to 168 hours. PLAUSIBLE.** `plan.py:86-87`, `orchestrator.py:423-424`.
- Failure: `multiprocessing.Pool.map` waits forever if a child dies, as under the kernel's out-of-memory killer, since the lost chunk is never redone. This is new in this commit. These tasks have `gpu=False`, so `_stalled` exempts them, and the progress lines keep the script's idle limit quiet. The parent holds all prepared text, with up to 32 children forked from it. Fork itself looks safe: `TOKENIZERS_PARALLELISM=false` is set, and `CUDA_VISIBLE_DEVICES` is empty for these tasks.
- Fix: use `concurrent.futures.ProcessPoolExecutor` with the fork context, which raises `BrokenProcessPool`, and let that fail the task. Or give non-GPU tasks a wall-clock limit.

**3. Medium. A failure in speculative work ends the run, even if its gate would never have needed that work. PLAUSIBLE.** `orchestrator.py:452-465`, `:471-472`.
- Failure: `_on_failure` never looks at `self.speculative`. A fatal exit, the restart limit or the out-of-memory limit on a speculative seed raises `RunFailed`. If the first seed was about to fail G2, a legitimate stop or retry becomes FAILED, and the machine bills until someone comes back. No wrong verdict results. `D1_RETRY=1` keeps finished work, but speculates again until the one-seed gate is stored, so the same failure can recur before then.
- Fix: if the failed task is speculative, drop it from the queue with its dependents and log it. If it is later required, it runs then, and fails the run properly if it fails again.

**4. Low. Probe fits now have a stall clock and no heartbeat. PLAUSIBLE.** `orchestrator.py:423-431`, `worker.py:79-105`.
- Failure: the old pool applied the stall limit to `loads_model` tasks only. The new one includes `fit-probes`, which never writes a heartbeat, so a fit dies 30 minutes after it starts, however well it is going. `load_stall_seconds` also ignores `D1_STALL_MINUTES`. About 3,750 steps normally take minutes, so this needs heavy contention on a shared card. If it does fire, it costs 90 minutes and a failed run.
- Fix: beat once per epoch, or at least once a minute from a thread in `task_fit_probes`.

### Checked, and sound on reading

- **Adapter hooks and base reads, the main silent risk: no leak path found.** Hooks are registered only in `LoraAdapter.applied`, and removed by `finally` (`lora.py:73-79`). Each capture job applies its own adapter in a `with` (`produce.py:170-172`). `produce.py:173-175` checks every target linear after each job and exits fatally before the marker is written. Any exception leaves `serve_capture` through `main()` and ends the process, so a worker never serves another job after a failure. An out-of-memory retry inside `run_capture` stays inside the same job's adapter. Training and the dry run, which also call `train()`, run in their own processes. The check ignores block-level hooks and pre-hooks, and nothing in the serve path registers a lasting one: `gather_at` removes its own.
- **Would `tests/test_d1_parallel.py` detect a leaked hook? Yes.** `test_a_read_after_an_adapter_on_one_loaded_model_is_a_read_of_the_base` (`:656-678`) reads through a non-zero adapter and then the base on one `LoadedModel`. A leak trips the `FatalTaskError` check. With that check removed too, `array_equal` against a fresh process fails, on the CPU where same batches give identical bits. It also asserts that the adapter changed the reads, so it is not vacuous. Its limits: it runs in-process and not through a real `serve` subprocess, whose only real-worker test (`:636`) reads the base alone, and it does not run on a GPU. The dry run covers removal on the real model, but not across jobs.
- **Shards and merges.** Rows are addressed by index, and each store row belongs to one shard. A row is done only after a flush, and a marker is written only after the writer closes. `read_complete` checks the whole store. A task cannot be claimed twice: `submit` checks running and queued, and a failed task returns only through `_on_failure`. A task cannot be lost: a dead worker with a task goes back through `_reap`, also after a failed stdin write.
- **Stale dependencies.** A train marker is written only after exit 0, and `adapter.pt` is atomic and fsynced first.
- **Speculation and verdicts.** `evaluate_g2` reads only the seeds it is given, `full_analysis` only its attempt's IDs, and every ID carries the attempt. A cancelled task has no marker and resumes from its done mask or `resume.pt`. A retry at twice the steps uses new `a1_` names and directories, so it cannot collide with `a0_`.
- **Order.** FLIP and HONEST are built only after `decision == "finish"` (`orchestrator.py:1024-1029`). G2 cannot read early: `run` blocks on markers, and `needs` runs before the one-seed gate.
- **Out of memory.** Exit 4 has its own count and ends at `MAX_OOM_RETRIES`, and step-down moves checkpointing and then the micro-batch. A worker dead while others wait is reaped each tick, and a true blocked state raises. A shared-card worker has its own heartbeat, and training and capture both beat well inside 30 minutes.
- **Threads.** `RowWriter` keeps draining after an error, and `close` re-raises it. Rows written before an error are still marked. `PrefetchedBatches` surfaces errors through `.result()`, and only its thread touches the tokenizer. No stale tokenizer or config: one rung, one worker, and `answer_tokens` is re-read per job.
- **Analysis threads.** `parallel_map` keeps order, each random direction has its own seed, and nothing shared is written.
- **Cleanup.** `close` terminates and then kills, and SIGTERM reaches it. A worker whose orchestrator died exits within 5 seconds, through `exit_with_parent`. A process stuck in the driver is reported, and the rerun's free-memory check then refuses.
- **Machine-dependent settings.** All tasks launched together take one schedule, so DD and SHAM match. Step-down can differ per task, which is rounding only, as the tests assert on the tiny model. Checkpointing and budget are outside the identity, on purpose, as the change log allows. They are recorded in `run.log`, and not per adapter in `training.json`, which is worth adding but changes no verdict.
- **Shards and GPU count.** `TOKENS_PER_SHARD` is a constant, and `assign_shards` is pure in lengths and count. The token budget is not in task names, so a resume at another budget only changes batching.
- **1.7B config and one command.** `run_d1_gpu.sh` passes syntax. The default config, `cli` arguments, `run_all` order and `PoolOptions` agree, and the empty `workers_per_gpu` array matches the script's existing `resuming` pattern. The new `training_transcripts` and `training_ordinary` files are written in `build_training_rows` before the dry run reads them.

### Fixes

All four defects above are fixed, with tests in `tests/test_d1_parallel.py` and `tests/test_d1_rules.py` that use fake processes. Nothing here has run on a real GPU.

1. **Hang.** The memory factor belongs to the failing task. An idle capture worker leaves whenever its card has nothing running. Running out of memory beside an idle worker makes that worker leave, with no step-down. A worker told to leave is killed after 5 minutes. Required work that can't start with nothing running fails after 15 minutes, and a run with no start, finish or heartbeat fails after 6 hours. The overall limit fell from 168 to 120 hours.
2. **Tokenizer.** `ProcessPoolExecutor` raises when a child dies, so the task fails and uses its restart limit.
3. **Speculation.** A failed speculative task is dropped with the queued work that reads it, unless required work reads it. If a gate later asks for it, it runs as required, and a second failure ends the run.
4. **Fits.** A fit beats at the start of each epoch. `load_stall_seconds` now follows `D1_STALL_MINUTES` when that is above 30 minutes.

A fresh read of the fix found no high or medium defect, and three low items, also fixed. Running out of memory beside a probe fit or a retiring worker counted as being alone on the card, so the adapter stepped down for no reason. A tokenizer child dying midway through writing its answer can still block `ProcessPoolExecutor`, so tasks with no GPU now have a 3-hour limit. `_give_up` lacked tests through the out-of-memory and restart limits. Reverting the tokenizer fix makes its test hang until CI's 20-minute job limit rather than fail fast.

`probes.py` is hashed, so this changes its fingerprint. No stored data exists, so nothing is lost.
