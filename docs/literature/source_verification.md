# Source verification log

Claims checked directly against paper text (excerpts returned by the alphaXiv PDF tool, 2026-11-04), because later work depends on them. Excerpts, not full reads. arXiv IDs and dates are as the tool returned them.

## Hahami et al., IFT (2607.14111v1)
- **Confirmed:** r = 0.999 between detection and factual-no logit shifts on Llama-3.1-8B.
- **The r is computed across (layer, strength) configurations.** Section 2.3.1 uses 40 configurations (8 layers, alpha 1-5). Appendix B uses layers {3,6,9} and alpha {5,10,50,250}. A correlation across a grid with this dose range is dominated by dose. It does not, alone, exclude a specific component within a configuration. My reading, not the paper's.
- **Same number, two meanings.** Main text: "mean net signal" -0.006 +/- 0.033, read as confound. Appendix B: "mean control net signal" -0.006 +/- 0.033, read as "genuine sensitivity ... not a general affirmative bias". These cannot both stand.
- **Localization is called "immune" to global affirmative bias.** Cycling the injection through positions averages out primacy and recency bias, but I saw no control for salience at the injected position. A perturbation at one position is not uniform, so immunity to a uniform shift says nothing about that salience.
- **Peak versus average differ greatly.** Table 2 baselines: Llama-3B avg localization 14.4% (chance 10%), best 100%. Llama-8B avg 16.8%, best 100%. The grid reaches alpha = 250.

## Zou et al. (2609.35108v1)
- **Confirmed:** mean localization 54.49 / 44.31 / 65.95% for Qwen3-4B / Llama-3.1-8B / Gemma-3-12B.
- **Selected settings:** (layer, alpha) = (3,3), (0,6), (0,5), with alpha relative to the activation norm. So the best injections sit at the earliest layers at 3-6x the token's own norm.
- **Concept screening:** the top 300 concepts by calibration accuracy are kept, and test concepts come from that pool. The paper states results are "on this screened pool".
- **Controls run:** norm-matched random direction (below chance, position reports at the clean false-positive level) and lexical replacement (0.87-11.13%). The paper notes the lexical control does "not measure general semantic anomaly detection under an explicit semantic task".
- **Not run, in the pages retrieved:** any control task whose instructions do not mention injection.
- All evaluation prompts tell the model about injection.
- The anti-anomaly argument rests on norm-matched random directions. Ferrara et al. (below) argue norm matching is the weakest such control, because random directions do less downstream damage at equal norm.

## Ferrara, OWMI (2608.20569v1)
- **Confirmed:** AUROC about 0.5007; equivalence bound; linear probe 95.8% / 75.0%.
- **The notes' "null at 0.5B-15B" over-reads the small end.** Qwen2.5-0.5B is an integration check with 10 complete pairs and 16 new tokens.
- **Possible floor effect, not established.** Of 11,216 dose-battery pairs, 7,570 come from Qwen2.5-7B. In the eight-model battery the same model's parseable report rates are 0.000 / 0.000 / 0.042 / 0.000. If the same floor holds in the dose battery, the pooled binary AUROC near 0.5 and its tight bound would largely reflect a model that never says yes. The excerpts did not show dose-battery report rates, so this needs checking. The paper does report that confidence separates conditions at AUROC 0.647 in one model.
- Only Qwen2.5-7B has a properly impact-matched random control.

## Singh et al. (2605.26242v2)
- **Confirmed:** gaslight condition and two-way and three-way designs; models label gaslight trials as injections; three-way fails.
- Results reported are the best setting over a layer and strength search, steering only prompt tokens (except Gemma-3-27B).
- The authors state that passing the three-way test would still not establish second-order computation.

## Not yet verified
Everything else in `docs/literature/` remains as the survey threads reported it, with the reading-status tags they assigned.
