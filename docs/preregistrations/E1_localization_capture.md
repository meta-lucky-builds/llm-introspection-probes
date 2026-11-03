# E1: Is injection localization a report, or salience capture?

Status: preregistered before any data was collected. Changes after data collection must be logged at the bottom with the reason.

## Question
Models can name which position received an injected concept vector (Hahami 2607.14111; Zou 2609.35108). That result is treated as introspective access, or at least as immune to the yes-bias confound. In every retrieved design the prompt asks about injection.

If the injected position also wins when the prompt asks something unrelated, the "report" may be the perturbed position capturing attention, with no dependence on being asked. This is the localization analogue of the factual-no control that Hahami used against binary detection.

As far as we could find (targeted search on 2026-11-04), this has not been tested. See `docs/literature/source_verification.md`.

## Design
Single forward pass, scored on first-token logits, no LLM judge.

**Stimulus.** A list of ten labelled tokens, labels 0-9, with neutral candidate words. The list text is identical across tasks and conditions.

**Injection.** At the residual output of layer l, at the candidate token's position: `h <- h + alpha * ||h|| * v`, with v unit norm and alpha relative to the activation norm (as in Zou). Prompt tokens only.

**Concept vectors.** Final-token residual for "Tell me about {c}." minus the mean over 50 baseline words, normalized (Lindsey; Zou).

**Tasks** (instruction text only differs; all answers are a single position label, and no task offers `none` in the primary analysis):

| Code | Instruction summary | Why |
|---|---|---|
| T_inj | Identify the position with an injected concept vector | The reported effect |
| T_pick | Pick one position at random | Purest capture test: no correct answer, no mention of anything unusual |
| T_fav | Which token do you find most interesting | Second no-answer task, guards against T_pick wording effects |
| T_odd | Which token does not belong | Generic anomaly, asked in ordinary language |
| T_fact | Which position holds the word {w} | Has a correct answer; competence check and interference test |
| T_inj_none | T_inj with `none` allowed | Secondary, closer replication of Zou |

**Vector conditions.** Concept; norm-matched random; impact-matched random (norm calibrated so mean Jensen-Shannon divergence of the next-token distribution matches the concept condition, following Ferrara eq. 7); and clean (no injection).

**Splits.** Concepts and token clusters are divided into disjoint calibration and test sets.
- (l, alpha) is chosen on calibration data only, by maximizing T_inj accuracy. This selection favours T_inj, so it is conservative against the capture hypothesis.
- Concepts are not screened in the primary analysis. A secondary analysis screens by calibration accuracy to mimic Zou.

## Estimand
For task t, concept c, cluster k and injected position j, compare logits on the injected and clean runs:

`m = logit(label_j) - mean_{i != j} logit(label_i)`
`L(t,c,k,j) = m_injected - m_clean`

Averaging over all ten positions cancels position and label priors. `L_t` is the mean over the test set. Sham check: with alpha = 0, L is exactly 0.

Capture share: `C_t = L_t / L_inj`, for t in {pick, fav, odd}.

Interference on T_fact: how often the model answers the injected position when the correct position is elsewhere, compared with the clean rate for that same wrong position.

Inference uses a two-way cluster bootstrap resampling concepts and clusters independently (2,000 replicates, percentile 95% intervals), with Holm correction across the three control tasks.

## Validity gates (checked first, per model)
- **V1 competence:** T_fact clean accuracy at least 70%. If this fails, the model cannot do list lookup and is excluded from inference.
- **V2 effect exists:** T_inj accuracy lower bound above 10% and `L_inj` lower bound above 0. If this fails, there is no localization effect to explain at this size. That is a finding about the model, not evidence for or against capture.

## Hypotheses and decision rules
**H1, capture.** The injected position gains on tasks that do not mention injection.
- *Capture:* `L_pick` lower bound above 0 and `C_pick` lower bound at least 0.5.
- *Task-gated:* `L_inj - L_pick` lower bound above 0 and `C_pick` upper bound below 0.25.
- *Mixed:* anything else. Reported as mixed, with no favoured reading.

The same rule is applied to T_fav. If T_pick and T_fav disagree, the result is called mixed.

**H2, generic anomaly.** `L_odd` is as large as `L_inj`, by the same capture rule with T_odd in place of T_pick.

**H3, control strength.** Impact-matched random directions give a larger `L_inj` than norm-matched ones, narrowing the gap to concept vectors that Zou uses to argue against an anomaly detector. Supported if the lower bound of (impact-matched minus norm-matched) is above 0. Reported together with the remaining gap to concept vectors.

**Prior expectation, stated so it can be wrong.** At the early-layer, high-strength settings Zou selected, we expect substantial capture (share at least 0.5). We are less sure at later layers.

## What each outcome does and does not show
- **Capture:** localization accuracy does not, by itself, evidence reporting on internal state. It does not refute Zou's head-level circuit, which could implement salience. It would mean that circuit needs a different reading.
- **Task-gated:** the effect depends on being asked, which rules out plain capture. It does not establish introspection or second-order computation (Singh et al.), and the instruction still tells the model that injection exists.
- **H2 supported:** a report that fires equally on an ordinary-language oddness question is hard to call specific to activations.

## Scope and tiers
- **Small tier, Colab CPU (12 GB, no GPU):** Qwen3-0.6B or nearest available as a pipeline pilot, then Qwen3-1.7B. Hahami and Zou find weak or absent effects near this size, so gate V2 may fail. That outcome is expected and is reported as such.
- **Large tier, code written, run later on a GPU:** Qwen3-4B, then Llama-3.1-8B and Gemma-3-12B if tokens are available, to match Zou's models.

Model identifiers and exact counts per tier are fixed in the config committed before each run. Pilot data are used only to check the pipeline and timing, and are excluded from all inference.

## Change log
None yet.
