# E2: Does injected-position advantage travel through J-space?

Status: preregistered before any E1 small-tier or large-tier result was seen and before any E2 data. Pilot E1 numbers on Qwen3-0.6B had been seen. Changes after data collection must be logged at the bottom with the reason.

## Question
E1 asks whether the injected position wins even on tasks that do not mention injection. It cannot say what carries the effect. Gurnee et al. (2607.15495) describe a J-space of verbalizable directions and argue it is causally used for reasoning and report. So: does the advantage depend on the part of the injected vector inside J-space, or on the rest?

As far as I found, nobody has tested this for position localization, or on tasks that do not mention injection, or on open-weight models. See `docs/literature/j_space_and_recent_work.md`, whose caveat applies: only about 15 of about 100 pages of that paper were read.

## Stage 0: lens check
Use the published Qwen3-4B J-lens if it was fit on the same checkpoint E1 uses. If not, fit one with `anthropics/jacobian-lens` on about 100 prompts, and record the difference. Record the fit data, the target layer and the file hash.

## Stage 1: validity gate on Qwen3-4B
A third-party README reports the J-lens advantage over the logit lens reversing on Qwen3-8B, so validity on this family cannot be assumed.

Using the evaluation sets in `anthropics/jacobian-lens`, measure intermediate-concept recall (pass@10), J-lens versus logit lens, per layer, and report the best layer band. Intervals use a bootstrap over prompts. Compare using English targets as the primary measure, and report Chinese targets too.

**Gate passes** if, in some contiguous band of at least 3 layers, the lower bound of (J-lens minus logit-lens recall) is above 0. The band with the largest mean gain gives the workspace layer `l_w`, its middle layer. The band is fixed before Stage 3.

**If the gate fails:** E2 stops. I record that the J-lens does not validate on Qwen3-4B, and no J-space claim is made for this model. Stages 2 and 3 do not run. This is a stop rule, not a reason to tune the lens.

## Stage 2: decomposition of E1's vectors (descriptive)
Requires E1 large-tier data. For each E1 test concept vector at E1's selected layer `l_s` and at `l_w`: top lens tokens, the k = 16 gradient-pursuit J-space component, and its share of the vector's squared norm. Spearman correlation across concepts between J-share and each of `L_inj` and `L_pick`, with bootstrap intervals over concepts.

This stage is descriptive and decides nothing alone. E1's vectors were measured at `l_s` only, so correlations are reported at `l_s`.

## Stage 3: vector-family experiment
Harness, estimand (`L`) and bootstrap as in E1. Stimuli are E1's test concepts and clusters. No selection happens on them here.

**Sites.** `l_s` and `l_w`. The lens is reported to be noisy in roughly the first third of layers, so `l_s` is the weak case for any J-space reading if E1 selects an early layer.

**Tasks.** T_inj, T_pick, T_odd.

**Vector families**, each unit-normalized, then scaled per family and site so that mean next-token Jensen-Shannon divergence matches the full vector's, on calibration stimuli only, using E1's search and tolerance. A family missing the 10% tolerance is reported as unmatched and excluded from the decision rules.

| Code | Vector |
|---|---|
| F_full | Full concept vector, at E1's alpha at `l_s` and at the alpha found on calibration stimuli at `l_w` |
| F_j | Its J-space component |
| F_res | Its residual (full minus F_j) |
| F_lens | The J-lens vector for the concept's token, where the concept is a single token |
| F_rand | Random direction |

## Hypotheses and decision rules
Evaluated separately per site. Let `R(F, t) = L(F, t) / L(F_full, t)`, undefined if `L(F_full, t)` is below 0.05 logits or its lower bound is not above 0, in which case that task and site are reported as having no effect to decompose.

**H1, J-space route.** `R(F_j, t)` lower bound at least 0.5, and `L(F_j, t) - L(F_res, t)` lower bound above 0.

**H2, residual route.** `R(F_res, t)` lower bound at least 0.5, and `L(F_res, t) - L(F_j, t)` lower bound above 0.

**H3, content-free route.** `L(F_rand, t)` is at least half of `L(F_full, t)` by lower bound, at matched impact. Then direction content matters little, and impact explains most of the effect. This takes precedence over H1 and H2 when interpreting.

**Mixed:** anything else. Reported as mixed, with no favoured reading.

**H4, dissociation.** T_inj is classified as H1 while T_pick is classified as H2 or H3. That would mean the asked-about report and the capture effect use different routes. Holm correction applies across the two sites and three tasks for each comparison.

**Prior expectation, stated so it can be wrong.** At an early `l_s`, I expect H3 or H2, since the injection is large relative to the token and the lens is noisy there. At `l_w` I have no strong expectation.

## What each outcome does and does not show
- **H1 on T_inj only:** evidence of a J-space route to the report on an open model. It does not establish a global workspace, or second-order monitoring.
- **H2 or H3:** localization here does not need verbalizable content. It says nothing about Claude, where the published experiments ran.
- All results are for one model, one concept set, and injected vectors of this size.

## Stop rule for this line of work
If the gate fails, or both sites come out H3 or mixed, close the J-space line for Qwen3 and pick the next ledger item. Ablation (a necessity test) runs only if Stage 3 finds H1 on some task.

## Change log
All entries below were made before any E2 data was collected, and before any E1 small-tier or large-tier result was seen. The README section on E2 gives further detail.

2026-11-04, design correction after independent review ([review](../reviews/E2_review.md)). This overrides the Stage 3 hypotheses above where they conflict:
- **The flaw.** The J-space component is expected to hold about 6-7% of a vector's squared norm, so F_res has cosine about 0.97 with F_full. H2 as written would pass almost regardless of how the effect works, so it could not count as evidence.
- **New family F_sham.** F_full minus a random combination of k = 16 randomly drawn J-lens atoms, scaled so that the removed vector's norm equals that of the J-space component for the same concept and site. It removes as much as F_res does, but not the J-space part. Impact-matched like the others.
- **H1, replaced by two tests.** *Necessity:* `L(F_sham, t) - L(F_res, t)` lower bound above 0. *Sufficiency:* `L(F_j, t) - L(F_rand, t)` lower bound above 0, and `R(F_j, t)` lower bound at least 0.5. **J-space route** requires both.
- **H2, replaced.** **Residual route** requires `L(F_sham, t) - L(F_res, t)` upper bound below 10% of `L(F_full, t)`, so removing J-space costs no more than removing the same amount at random, and F_j fails sufficiency.
- **H3 and precedence are unchanged.** H3 still takes precedence when interpreting. It also needs F_rand matched: if F_rand is unmatched, H3 is untested, and that task and site are reported as undetermined, not classified.
- **Unmatched families.** If F_j, F_res, F_sham or F_rand misses the match, the affected test is reported as undetermined. Undetermined is distinct from mixed, and does not by itself trigger the stop rule.

2026-11-04, implementation choices the text above left open:
- **Chinese targets.** The published evaluation sets contain none, so my own table was written. Descriptive only; it does not enter the gate.
- **Order-of-operations synonyms.** The published sets do not give them, so these are mine, and they affect the gate. Reported so the gate can be recomputed without them.
- **Impact tolerance.** The search stops at 5%; a family is excluded only above the prereg's 10%.
- **Bands and pooling.** A band is a maximal run of qualifying layers; `l_w` is its middle layer, taking the lower of two for an even-length band. Recall is pooled over items.
- **Holm.** Also applied to ratio bounds, with the unadjusted classification reported beside it.
- **Readout position.** As fixed in the code and README, with prompt-plus-target tokenization.
- **Stop rule reading.** Stated in the README.
- **Collapsed sites.** If `l_s = l_w`, Stage 3 runs one site.
- **Gradient pursuit.** Unit atoms, a final exact non-negative least squares step, special tokens excluded. The paper does not fix these details and no published code exists, so this decomposition is my implementation of its definition.
- **Lens.** Fit on Qwen3-4B's newest commit, which E1 pins, so no refit branch is implemented. `norm` in the readout is the final RMSNorm, not unit normalization. The lens file is pinned by its LFS pointer's size and sha256.
