# E1 small tier: Qwen3-1.7B

Decision rules: [prereg](../../../docs/preregistrations/E1_localization_capture.md). Full run on Colab CPU, 1,095 records. `export.txt` here regenerates the analysis with `mechinterp.analysis`. Colab's analysis file had sha256 `05ce8305786757a0a40a117de5465a72cd73cbc3e0a6a455c848751415adf725`; it was read from cell output, not stored here.

## Preregistered verdicts
- **Gates:** V1 passed (T_fact clean accuracy 0.99). V2 passed (T_inj accuracy 0.58, interval [0.40, 0.73]).
- **H1, capture: mixed overall.** T_pick is classified as capture. T_fav is mixed. The prereg calls disagreement mixed.
- **H2, generic anomaly: supported.** T_odd is classified as capture.
- **H3: uninformative.** Impact matching missed tolerance (11.3% vs 10%), so H3 was not tested.
- Selected site: layer 3, alpha 6. Grid accuracy on calibration data: layer 0: 0.55, 0.60; layer 3: 0.65, 0.70; layer 10: 0.20, 0.00.
- Sham: exactly 0.

## Estimates (test split, 95% percentile bootstrap)
| Condition | L (logits) | Accuracy on injected position | Share of L_inj |
|---|---|---|---|
| T_inj, concept | 6.28 [4.26, 8.68] | 0.58 [0.40, 0.73] | |
| T_pick, concept | 7.13 [5.24, 8.63] | 0.61 [0.43, 0.75] | 1.14 [0.80, 1.65] |
| T_fav, concept | 4.15 [1.92, 6.82] | 0.39 [0.24, 0.57] | 0.66 [0.37, 1.00] |
| T_odd, concept | 13.88 [11.35, 16.07] | 0.81 [0.68, 0.92] | 2.21 [1.67, 3.00] |
| T_inj, random, norm-matched | 5.82 [3.80, 7.65] | 0.60 [0.40, 0.76] | |
| T_inj, random, impact search | 4.65 [3.21, 6.26] | 0.54 [0.37, 0.69] | |

T_fact: accuracy under injection 0.98 and no interference.

## Reading (not preregistered)
- At this site, a prompt saying "pick at random" favours the injected position as much as a prompt asking about injection. The ordinary question "which does not belong" favours it far more.
- Norm-matched random directions localize about as well as concept vectors here (difference 0.46, interval [-1.58, 3.03]). Zou et al. report random directions failing, on Qwen3-4B with `none` allowed, so this does not directly contradict them. The design differs.
- Together these fit salience capture at an early layer, at 6x activation norm, rather than a report about internal state. This is what the prereg expected.

## Limits
- One model, 10 concepts, 10 clusters. Intervals are optimistic: coverage is about 90% for a nominal 95%.
- A large perturbation at layer 3. Layer 10 shows little localization on T_inj. The result says nothing about later layers.
- Forced answer prefix, so declining cannot be observed.
- Does not refute Zou's circuit, which could implement salience.
- The large tier is paused while the mission focuses on Double-Down (owner's instruction, 2026-11-04), so this is not replicated at a size matching published models.
