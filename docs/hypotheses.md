# Hypothesis ledger

One line per hypothesis. Update status with a link to evidence. Refuted hypotheses stay here.

| ID | Hypothesis | Status | Where |
|---|---|---|---|
| E1-H1 | Injected-position advantage also appears on tasks that do not mention injection (salience capture) | Preregistered, not run | [prereg](preregistrations/E1_localization_capture.md) |
| E1-H2 | An ordinary-language "which does not belong" question gives as large an advantage as the injection question | Preregistered, not run | same |
| E1-H3 | Impact-matched random directions localize better than norm-matched ones | Preregistered, not run | same |
| E2-Gate | The J-lens beats the logit lens on Qwen3-4B | Preregistered, not run | [prereg](preregistrations/E2_jspace_vector_families.md) |
| E2-H1/H2/H3 | Injected-position advantage travels through the J-space component, the residual, or impact alone | Preregistered, not run | same |
| E2-H4 | The asked-about report and capture use different routes | Preregistered, not run | same |

## Candidates, not yet preregistered
- **Fine-tuned lie detectors.** Anthropic reports held-out AUROC plateauing near 0.70-0.75 (blog, Aug 2026; see [notes](literature/j_space_and_recent_work.md)). Do J-space or simple probes generalize better across held-out deception settings?
- **J-space monitor evasion.** 2610.04316 reports GCG suffixes jailbreaking while a J-space monitor stays quiet, and first-order lens-fit seed variance. Replicate and bound both before trusting any J-space monitor.
- **Within-configuration yes-bias.** Hahami's r = 0.999 spans a wide dose grid. Does a detection-specific interaction exist within a fixed layer and strength, at trial level? Cheap on CPU.
- **First-order removal.** Remove the component of the injected vector along the gradient of logit(yes) - logit(no) at the injection site. Does any detection signal survive? Separates first-order push from anything else.
- **Binary floors hide signal.** Do first-token logit margins show effects where binary report rates sit at a floor (see the Ferrara note in [source verification](literature/source_verification.md))?
- **Trained introspection and capture.** After Introspection Fine-Tuning, is the gain task-gated or does it generalize to neutral tasks? Needs GPU.
- **Post-training origin.** Does DPO create injection-specific detection or generic anomaly detection? Three-way controls across public OLMo checkpoints. Needs GPU.
- **Emergent-misalignment controls.** Replicate small-model EM claims with response-length controls (2607.09053). Needs GPU.
