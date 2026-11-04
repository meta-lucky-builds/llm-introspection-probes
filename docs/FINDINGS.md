# Findings log

Everything this project has found so far, strongest evidence first.

| Level | Meaning |
|---|---|
| **Measured** | I ran it. The data is in this repo and CI recomputes it |
| **Derived** | Arithmetic or algebra from stated definitions. No model was run |
| **Source audit** | My reading of paper excerpts from the alphaXiv PDF tool, not full reads. A claim that something is absent means absent from the pages retrieved |
| **Fetched** | Came through a summarizing web fetch, so numbers are unverified against the page |
| **Survey** | A pattern across my notes. Absence here is weak evidence |

## 1. Injected-position advantage appears without asking about injection

**Measured.** Qwen3-1.7B, Colab CPU, 1,095 records, layer 3, alpha 6 (chosen on calibration data). [Results](../results/e1/small_qwen3_1.7b/RESULTS.md), [prereg](preregistrations/E1_localization_capture.md).

- Gates passed: V1, clean lookup accuracy 0.99; V2, injection task accuracy 0.58 [0.40, 0.73] against chance 0.10. Sham was exactly 0.
- "Pick one at random" gave a lift of 7.13 [5.24, 8.63], at a share of 1.14 [0.80, 1.65] of the injection task. It meets the capture rule.
- "Most interesting" gave 4.15 [1.92, 6.82], at a share of 0.66 [0.37, 1.00]. Its lower bound is under 0.5, so it is mixed, and H1 is mixed overall as the prereg requires.
- "Does not belong" gave 13.88 [11.35, 16.07], far above the injection task. H2 is supported.
- Norm-matched random directions did about as well as concept vectors: difference 0.46 [-1.58, 3.03]. Zou et al. report random directions failing, on Qwen3-4B with `none` allowed, so the designs differ and this is not a direct contradiction.
- Lookup stayed intact under injection, at 0.98 accuracy, and the model never answered the injected position when that was wrong (0.00). So the perturbation shifts choice among positions without breaking a task that has a right answer.
- Answer labels took nearly all the first-token probability on every task (label mass about 1.00), so these logits score real answers.
- H3 was not tested, because impact matching missed tolerance (11.3% against 10%).

**Limits.** One model, 10 concepts, 10 clusters, and coverage near 90% for a nominal 95%. Calibration accuracy at layer 10 was 0.20 and 0.00, so the effect found here is an early-layer one at 6× norm, within the range Zou et al. selected (layers 0-3 at 3-6×). The forced answer prefix hides declining. The pilot, seen before this run, also showed pick above injection. The decision rules, stimuli and counts did not change after it.

## 2. Weak points in the injection literature

**Source audit.** [Log](literature/source_verification.md). These are my readings and do not claim the authors are wrong.

- **Hahami et al. (2607.14111).** The r = 0.999 between detection and factual-no shifts is computed across (layer, strength) configurations, so it may be dominated by dose. It does not by itself exclude a specific component within one configuration.
- **Hahami et al.** The same number, -0.006 ± 0.033, is read as confound in the main text and as genuine sensitivity in Appendix B.
- **Hahami et al.** Localization is called immune to global affirmative bias. Cycling the injection through positions averages out primacy and recency bias, but I saw no control for salience at the injected position itself.
- **Zou et al. (2609.35108).** Mean localization is 54.49 / 44.31 / 65.95% at selected settings in early layers (0 and 3) at 3-6× norm, on a screened concept pool. In the pages retrieved, no control task omits injection.
- **Ferrara (2608.20569).** A possible floor effect: 7,570 of 11,216 dose-battery pairs come from one model whose report rates elsewhere in the paper are near 0. Not established, because the excerpts did not show its dose-battery rates.

## 3. What the FIBS contrast score can and cannot rule out

**Derived, from definitions in a source I read closely.** [Attack, part A](designs/monitor_experiments_attack.md).

- The score's difference between lies and honest rows is `[s(Y|lie) − s(N|lie)] − [s(Y|honest) − s(N|honest)]`, which is the goal-by-answer interaction. On raw scores, an additive "goal is represented in me" probe scores AUROC 0.5 on it, and an interaction of active goal with affirmation survives.
- That cancellation is exact only on raw scores. FIBS appendix A.2 takes the difference after a sigmoid for linear and MLP ensembles, where an additive effect doesn't fully cancel. Axial scores look raw (around ±15), but I couldn't confirm that.
- FIBS's hidden-goal evaluation keeps denials only, so admissions are filtered out before scoring. I found no published row where a probe scores an active goal with a true statement.
- This is why my first design, M1, [was dropped](designs/monitor_experiments_attack.md#0-verdicts): it attacked a read the headline doesn't use, and an alternative the contrast differences out on raw scores.

## 4. The lie-detector plateau may partly be a label ceiling

**Derived, on Fetched inputs.** [Attack, part C](designs/monitor_experiments_attack.md#part-c-m2-and-m3), [notes](literature/monitor_generalization.md).

- With symmetric label noise ε independent of score, observed AUROC = ε + (1 − 2ε)·AUC. At ε = 0.2 the ceiling is 0.80, and an observed 0.75 implies a true AUC of 0.917.
- The post reports about 25% of labels changing under relabelling and 46% instability for self-sycophancy. Residual ε is **unknown**, and estimating it should come before any GPU work on this question.
- FIBS's own families include `lies`, `mask`, `sycophancy` and `sandbagging`, so a FIBS probe has already seen those "held-out" categories.
- Correction to my own earlier note, per my fetch: the reported fine-tuned detectors were on Gemma-3. Qwen3 and GPT-OSS appear in lie-rate statistics, and the post calls its larger-model experiments inconclusive. The 0.98-0.99 for Qwen3-235B is a prompted detector.

## 5. Open questions the survey surfaced

**Survey.** [Monitor notes](literature/monitor_generalization.md), [J-space notes](literature/j_space_and_recent_work.md).

- **Evasion.** Every positive result on evading a probe in my table needed an explicit trigger, instruction, optimization against a monitor, or a trained organism. Whether ordinary "say you were truthful" training blinds a probe looks untested, after only two targeted searches. D1 asks this.
- **Hidden-goal organisms.** FIBS reports 90.4-99.7% AUC on five of eight, while AISI (2606.12618) reports median balanced accuracy 0.52 (Apollo probe) and 0.65 (Did-You-Lie probe). Those are different metrics on largely the same organisms: all eight FIBS organisms are on Qwen3.6-27B, and seven are AISI's published ones. As FIBS reports it, median AUROC is 0.94 for its axial probe against 0.75 for Did-You-Lie. Architecture accounts for 0.09 of that gap. Training data, read position and negatives are not separated, so the gap is only partly explained.
- **Detect versus resist.** Mean pooling detects deception better in two sources, while max-pooled and attention probes resisted RL-Obfuscation in another (harmfulness only). The trade-off is my inference and is untested on deception.
- **J-lens on Qwen.** A third-party README reports the J-lens advantage over the logit lens reversing on Qwen3-8B (0.246 against 0.291). It is a README only and I did not rerun it. E2's gate exists because of this.

## 6. Design defects caught before any data

**Derived, on my own designs.** These are not results about models. They show why the preregistrations changed.

| Design | Defect | Fix |
|---|---|---|
| D1 | "Independent channel" could be declared with a dead probe in both arms | Probe floor; new `probe_unusable` outcome ([review](reviews/D1_review.md)) |
| D1 | Silencing gate used logit AUROC, so a model always saying "Yes" could still fail it | Gate on the model's actual answers |
| D1 | Sham training was nearly a no-op, so it wasn't comparable to the manipulation | Sham question whose natural answer is No |
| D1 | Seven test families are about four lines of work, so intervals were too narrow | Bootstrap by line of work |
| E2 | `cos(F_res, F_full) = sqrt(1 − j_share)`, about 0.97 at the paper's J-share, so H2 was near-guaranteed | Added F_sham; necessity and sufficiency tests ([review](reviews/E2_review.md)) |
| M2 | A 0.80 bar can sit at the label ceiling, and there are only two folds | Dropped as a flagship |

## What I do not have

- **No D1 result.** D1 has never run on a GPU, so it says nothing yet about whether probes are an independent channel.
- **No E2 result.**
- **No E1 result at 4B or above.** The closest test of the published claims is unrun.
