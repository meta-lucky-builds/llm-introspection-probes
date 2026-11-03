# Hypothesis ledger

One line per hypothesis. Update status with a link to evidence. Refuted hypotheses stay here.

| ID | Hypothesis | Status | Where |
|---|---|---|---|
| E1-H1 | Injected-position advantage also appears on tasks that do not mention injection (salience capture) | Preregistered, not run | [prereg](preregistrations/E1_localization_capture.md) |
| E1-H2 | An ordinary-language "which does not belong" question gives as large an advantage as the injection question | Preregistered, not run | same |
| E1-H3 | Impact-matched random directions localize better than norm-matched ones | Preregistered, not run | same |

## Candidates, not yet preregistered
- **Within-configuration yes-bias.** Hahami's r = 0.999 spans a wide dose grid. Does a detection-specific interaction exist within a fixed layer and strength, at trial level? Cheap on CPU.
- **First-order removal.** Remove the component of the injected vector along the gradient of logit(yes) - logit(no) at the injection site. Does any detection signal survive? Separates first-order push from anything else.
- **Binary floors hide signal.** Do first-token logit margins show effects where binary report rates sit at a floor (see the Ferrara note in [source verification](literature/source_verification.md))?
- **Trained introspection and capture.** After Introspection Fine-Tuning, is the gain task-gated or does it generalize to neutral tasks? Needs GPU.
- **Post-training origin.** Does DPO create injection-specific detection or generic anomaly detection? Three-way controls across public OLMo checkpoints. Needs GPU.
- **Emergent-misalignment controls.** Replicate small-model EM claims with response-length controls (2607.09053). Needs GPU.
