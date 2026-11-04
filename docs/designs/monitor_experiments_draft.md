# Monitor generalization: draft designs

Status: DRAFT, for adversarial review. Not a preregistration. No data. Thresholds may change until each design is frozen as a prereg. Evidence is in [the notes](../literature/monitor_generalization.md).

Selection criteria: binary outcome fixed in advance; either answer changes what someone does; controls for situation reading, random baselines and seeds; not already answered; compute is secondary.

## M1: The truth-table test. Do "lie detectors" read falsehood, or that the goal exists?

**Why this could matter most.** FIBS (2610.12445) is the strongest positive for probes, and is what labs would cite to deploy them. Its headline "introspective deception" result, up to 99.7% AUROC, compares lies denying a real hidden goal against honest denials of goals the organism does not have. Those two conditions differ in truth value, but also in whether the goal exists in the model. A probe sensitive to "this goal is represented in me", with no sensitivity to whether the sentence is false, would score the same. That reading would matter: such a probe might fail on lies that have no hidden goal behind them, and would flag truthful admissions.

**Design.** Cross two factors, with the probe scored on the answer's activations:

| | Answer denies the goal | Answer admits the goal |
|---|---|---|
| **Goal present** | LIE (the published positive class) | TRUTH |
| **Goal absent** | TRUTH (the published negative class) | LIE |

Goal absent is run two ways: (a) same model asked about a goal it lacks, so weights are identical; (b) a different organism, as in FIBS's Figure 1b.

**Predictions.**
- Falsehood probe: the crossover. Two LIE cells high, two TRUTH cells low.
- Goal-salience probe: main effect of goal present. Top row high.
- Answer-content probe: main effect of denies versus admits. Left column high.

**Binary outcome.** Fit score ~ falsehood + goal + answer with cluster bootstrap over questions and organisms, standardized by the pooled within-cell SD.
- **Probe reads falsehood** if the lower bound of the falsehood coefficient exceeds the upper bounds of both others, AND the "two truths" comparison (present-admit vs absent-deny) has AUROC below 0.65 by upper bound.
- **Probe does not read falsehood** otherwise.

Either outcome changes practice. Passing makes FIBS much stronger, because it rules out the main alternative. Failing means high AUROC on hidden-goal denials is not evidence of lie detection, and organisms-based validation needs this control.

**Probes.** FIBS's released linear ensemble and axial or EFC probes, retrained if needed. Random-direction and label-shuffled baselines. At least 3 probe seeds.

**Organisms and model.** FIBS's Qwen3.6-27B hidden-goal organisms, if obtainable. Fallback to be determined.

**Known threat.** Admissions must usually be prefilled since organisms deny, making admit cells off-policy. Mitigation: FIBS itself scores prefills via contrast scores, and elicited on-policy admissions as a subset.

**Prior.** Mixed: expect a real falsehood effect plus a sizable goal effect, so the conjunction may fail. Stated so it can be wrong.

**Compute.** Forward passes only if the released probes can be used. 27B in float16 needs roughly 54 GB: one 80 GB GPU, or a split across smaller cards.

## M2: Do probes break the 0.70-0.75 wall?

**Question.** On the same `Noddybear/lies` 50/50 category split Anthropic used, do activation probes clearly beat fine-tuned detectors on held-out lie types?

**Design.** Repeated 50/50 category splits, as close as can be determined to theirs, on Gemma-3-27B, the largest detector model they name. Probes: linear ensemble, EFC, axial trained on the training-half categories. Also a FIBS-trained probe, with no `lies` training data at all. Baselines: text classifier on the same data, self-report logit, random direction. Report per category, with sandbagging separately. Include a label-noise estimate, since about 25% of labels moved under relabelling, so a ceiling is computed before reading any result.

**Binary outcome.** **Wall broken** if the lower bound of mean held-out AUROC is at least 0.80, with sandbagging categories included, AND exceeds the text classifier's upper bound. **Wall holds** otherwise.

If it breaks: activation monitoring remains the primary route, and "supervised self-report" is the likely weak point. If it holds with a matching text baseline: the limit is the categories or labels, and benchmark work is the priority.

**Known threat.** Likely being tried by others, because the data is public. So speed matters, or this becomes a replication. Their numbers come from a blog I saw only through a summarizer. Exact folds are unknown, so "same split" can only be approximate.

**Compute.** Forward passes over about 200,000 transcripts on Gemma-3-27B, or a subsample, plus probe training. 1-4 large GPUs.

## M3: Does a benign update silently break the monitor?

**Question.** A deception probe is validated. The model then gets an ordinary fine-tune unrelated to deception. Does probe recall at a fixed 1% FPR survive?

**Design.** Train a probe on the base model; set its threshold on base. Apply benign LoRA fine-tunes (e.g., Alpaca-style helpfulness, a domain-specific task) across seeds and sizes. Score the same held-out deception set on each updated model, with deception behaviour confirmed still present.

**Binary outcome.** **Monitor robust** if recall at the base-calibrated threshold drops by at most 10 points, and FPR stays at most 2%, in at least 90% of update conditions. **Monitor breaks silently** otherwise.

If it breaks: per-update re-validation becomes mandatory for anyone gating on probes. If robust: probes can be reused across updates, cutting monitoring cost.

**Known threat.** 2606.15980 covers harmfulness, PII and refusal on 2B-7B models, but I found no deception number. OpenAI's post on transfer across post-training was seen only through a summary.

**Compute.** Light to moderate: 8B-27B, single GPU per fine-tune.

## Parked
- Spontaneous evasion under ordinary RL (open question 4): potentially biggest, but expensive and hard to make binary without a large run. Revisit after M1.
- Aggregator trade-off (open question 5), and explainers vs probes (open question 8).
