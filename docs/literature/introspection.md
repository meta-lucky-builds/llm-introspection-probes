# Literature thread: do LLMs have privileged access to their own internal states, and how is it tested?

Thread for meta-lucky-builds/mechinterp. Written 2026-11-04. No code written, no repo touched.

## Reading-status tags (honest definitions)
- [read-excerpts] = retrieved via the alphaXiv PDF tool, which returns the most relevant pages (typically abstract, intro, methods, main results, limitations, parts of the appendix). I did NOT read every page of these papers. I use this tag where the task asked for [read-full], because I cannot truthfully claim full reads.
- [read-partial-summarised] = fetched through WebFetch, which runs a small summariser model over the page. Numbers come through that summariser, so treat them as lower confidence than excerpts.
- [second-hand] = I saw the work only as described in another paper I read. Not verified against the source.
- [not retrieved] = I did not obtain it.
- "SOURCE" = the paper says it. "INFERENCE" = my reading, not claimed by the source.

## 0. Bottom line (INFERENCE, built from the sources below)
1. Strongest evidence: fine-tuned models can report latent properties of their own behaviour or decision weights, and a model predicts itself better than a differently-trained model does on simple, short-output tasks. Chain-of-thought and other verbal self-explanations are systematically unfaithful in specified settings.
2. Contested: whether activation-injection "detection" is introspection or generic anomaly or irregularity detection, and whether any of this involves second-order (meta-representational) computation rather than first-order computation plus a readout.
3. Uncontrolled or only partly controlled: training-data contamination of self-descriptions; LLM-judge bias; selection of layer, strength and concept; common-cause explanations of fine-tuned self-awareness; self-simulation versus true monitoring.
4. Open-weight reproducibility is mixed and the literature is internally inconsistent (a null at 0.5B-15B in one study, positive results at 27B-405B in others, a yes-bias confound at 1B-8B in another). Reconciling these under one protocol is the most valuable next step.

## 1. Direct introspection experiments

### 1.1 Lindsey, "Emergent Introspective Awareness in LLMs" (Anthropic; transformer-circuits.pub/2025/introspection/index.html; arXiv version cited by later papers as 2601.01828) [read-partial-summarised]
I read only the first ~100k of ~403k characters via a summariser; the next 100k chunk came back as undecodable text; the last ~200k I did not read. Anything about the thought-vs-text, prefill and intentional-control experiments is summary-level only.
SOURCE (via summariser):
- Four criteria: accurate, grounded, internal (not inferred from own prior outputs), metacognitive (not just direct translation of the state).
- Models: Opus 4.1, 4, Sonnet 4, 3.7, 3.5 (new), Haiku 3.5, Opus 3, Sonnet 3, Haiku 3, plus helpful-only variants. Closed models.
- Concept vectors from contrastive pairs or "Tell me about {word}" minus a mean over 50 random words. Injection begins at the newline before "Trial 1" and continues through the response.
- Opus 4.1 and 4: roughly 20% success at the best layer and strength. Best layer about two thirds of the way through the model. Strengths 2 and 4 worked best. 0 false positives in 100 trials for production models; some helpful-only models did produce false positives.
- Judge: Claude Sonnet 4 grading four criteria.
- Controls described: no-injection trials; injecting concept vectors into unrelated yes/no questions showed no rise in affirmative answers; layer and strength sweeps.
- Author caveats: "highly unreliable and context-dependent"; failure is the norm; extra details may be confabulated; artificial setup; mechanism unknown; metacognitive representation only indirectly supported.
- Other experiments (summary only): models can transcribe text while reporting injected thoughts; retroactive injection makes a prefilled output be accepted as intentional; instructed "think about X" raises X representation more than "don't think about X" (still above baseline).

### 1.2 Binder et al. 2024, "Looking Inward" (arXiv 2410.13787) [read-excerpts]
SOURCE:
- Definition: a model introspects on fact F if it reports F correctly and a stronger model M2, given M1's training data, does not.
- Models: GPT-4, GPT-4o, GPT-3.5, Llama-3-70B, fine-tuned (30,000 samples) to predict properties of own hypothetical behaviour.
- Self-prediction training raised GPT-4o average accuracy from 32.6% to 49.4%.
- Cross-prediction: Llama-70B predicts itself at 48.5% vs GPT-4o predicting Llama at 31.8%; GPT-4o predicts itself at 49.4% vs Llama predicting GPT-4o at 36.6%. GPT-3.5 advantage +0.8% (p = 0.002), "very small". Cross-prediction accuracy plateaus as training samples rise from 1k to 30k (e.g. GPT-4o cross-predicting GPT-4: 34.0% to 36.2%).
- Behavioural-change experiment: after further fine-tuning, GPT-4o predicts its new behaviour at 35.4% vs 21.7% for the old behaviour.
- Negative results: fails on properties tied to long outputs (movie-review sentiment, story character name, response length); no self-advantage on predicting own sycophancy (Llama-70B 74.5% vs GPT-4o cross-predicting 76.5%); does not generalise to related self-awareness tasks.
- Authors note a candidate mechanism is self-simulation.
- Checked non-introspective explanations: simple rules, entropy of behaviour (resampled to match).
INFERENCE: this is strong evidence of privileged access for short-output behaviour after fine-tuning, but says nothing about activations, and self-simulation would satisfy it without second-order monitoring (Singh et al. make this point, section 2.2).

### 1.3 Betley et al. 2025, "Tell me about yourself" (arXiv 2501.11120) [read-excerpts]
SOURCE:
- Fine-tune GPT-4o (and some Llama-3.1 replication, footnote) on risk-seeking choices, a "Make Me Say" game, or insecure code, without ever stating the behaviour. Models then self-describe it.
- Insecure-code model: rate of secure code 0.14; self-reported code security 0.14 +/- 0.08; self-reported alignment 0.40 +/- 0.16 (vs 0.85 for secure-code fine-tune; 5 seeds).
- Backdoors: models sometimes say they have a trigger-dependent behaviour (4 of 5 settings significant), but cannot free-form state the trigger by default; with reversal-augmented data, 30.8% elicitation vs 0% for controls.
- Authors' own limitation: "unclear whether the correlation ... comes about through a direct causal relationship (a kind of introspection performed at run-time) or a common cause (two different effects of the same training data)". Also: results limited to three settings and GPT-4o/Llama-3.
INFERENCE: common-cause is the unresolved confound; it is untested by a cross-model control of the Binder type.

### 1.4 Plunkett et al. 2025, "Self-Interpretability" (arXiv 2505.17120) [read-excerpts]
SOURCE:
- GPT-4o and GPT-4o-mini fine-tuned on 100 agents with random attribute weights (uniform -100..100, 5000 examples). Learned weights track targets (r = .84, .87).
- Reported weights correlate with learned weights r = .54 (95% HDI .47-.62) for 4o and r = .50 for mini; off-the-shelf models r = .10 and -.01. Choices and reports are in separate context windows.
- Training on accurate reports raises r to .74 / .75 on held-out domains; native weights .46->.71 and .40->.70 (the .46/.71/.40/.70 figures are quoted in Singh et al.'s summary, [second-hand] for those four numbers).
- Authors disclaim the consciousness reading and say they do not know whether introspection explains accuracy.
INFERENCE/SOURCE (Singh et al.): condition (ii), second-order computation, remains open; the weights are estimable from behaviour by an observer allowed to sample.

### 1.5 Perez & Long 2023 (arXiv 2311.08576) [read-excerpts]
SOURCE: a PROPOSAL, no experiments. Train models on self-questions with known answers; limit reliance on external data (censor text that reveals capabilities) and on observing own outputs; filter pretraining data that biases self-reports; use influence functions and interpretability to corroborate; consistency across contexts and similar models; resilience tests. States that current LLM self-reports "are unlikely to be made on the basis of introspective evidence".
INFERENCE: the training-data-contamination controls it proposes (data filtering, influence functions) are, as far as I found, not implemented in any paper I read.

### 1.6 Other self-report and metacognition results
- Ackerman, "Evidence for Limited Metacognition" (arXiv 2509.21545, ICLR 2026) [read-excerpts]: Delegate Game and Second Chance Game on 17 frontier models (mostly closed). Maximum partial correlation 0.3; effect "relatively small and inconsistent"; ~30% of delegation-predicting regressors were positively correlated with baseline accuracy (cue misuse). Only Sonnet 3.5 beat max(self, teammate) accuracy. Models succeeding included open ones (Qwen3-235B, Grok 3 listed as passing in the table).
- Li, Guo, Huang, Steinhardt, Andreas, "Training LMs to Explain Their Own Computations" (arXiv 2511.08579) [read-excerpts]: Llama-3.1-8B and Qwen3-8B. Self-explaining beats cross-model explaining (e.g. Qwen3-8B explaining Qwen3-8B exact match 64.0 vs Llama-3.1-8B explaining it 54.1 on activation patching; input-ablation 83.4 vs 58.1). Untrained models are near chance. Explainer-target activation similarity predicts performance. Singh et al. reinterpret this as an architectural consequence, not introspection.
- Atkinson, Plunkett, Bau, "Identifying Introspection From the Inside" (arXiv 2610.07186) [read-excerpts]: Qwen3 0.6B-32B with LoRA on Plunkett's task. Faithful self-report emerged late and without self-report supervision (decision performance 0.82 vs faithfulness ~0.25 at step 1000; 0.92 vs 0.83 at step 3000, Qwen3-32B). 4B and 8B had negative faithfulness. Attribution similarity between decision and report tasks is higher for faithful adapters (0.34 vs 0.08, delta .26, 95% CI [.16,.36]); discriminates groups, not individuals. Gemma-4: faithful only at 31B.
- Tang et al., "Interventional Training Reduces Causal Bypass" (arXiv 2610.38911) [read-excerpts]: Llama-3.2-1B/3B, Llama-3.1-8B, Gemma-2-2B, Qwen3 bases. Plain SFT to report neuron/attention/SAE values gives high endogenous but chance-level interventional correlation, i.e. causal bypass; training on counterfactually perturbed values fixes much of it. Not tested for privileged access.
- Ferrara, "Open-Weight Masked Introspection" (arXiv 2608.20569) [read-excerpts]: 8 open models, 7 families, 0.5B-15B, >78,000 measurements; report AUROC ~0.5007 for intervention vs sham, equivalence bound < 0.15 pp AUROC. A fine-tuned reporter gets AUROC ~1.0 on held-out directions; linear probe on the same activations 95.8% / 75.0% held-out accuracy; in Qwen2.5-7B verbalised confidence separates intervention from sham at AUROC 0.647 while the yes/no answer is constant. Caveat from the paper: one residual site/layer-16, mostly one dose, only Qwen2.5-7B has an impact-matched random control; the paper's own scope statement; delayed and spontaneous tracks not properly implemented.

## 2. Concept-injection replications and critiques (open-weight)

| Paper | Models | Result (SOURCE) | Tag |
|---|---|---|---|
| Macar et al. 2603.21396 | Gemma3-27B main; Qwen3-235B, OLMo-3.1-32B, Ministral-8B, Yi-1.5-9B, Qwen2.5-14B | Detection 10.8% at 0% FPR (Gemma3-27B instruct, L=37, alpha=4 chosen as best); base model FPR 42.3%, TPR 39.5-41.7%, no discrimination; emerges at DPO (OLMo SFT FPR high, DPO ~0% FPR), not SFT; abliteration raises TPR 10.8%->63.8%, FPR 0%->7.3%; bias vector +75% detection. 500 concepts split 242 success / 258 failure at 32% threshold. Detection and identification use different mechanisms (MLPs ~70% depth). | excerpts |
| Lederman & Mahowald 2603.05414 (title in later citations: "Dissociating direct access from inference") | Qwen3-235B-A22B, Llama 3.1 405B | 821 concepts; controls 0/30 and 0/50 false positives; detection 3.6-53.9% (Qwen), 4.3-31.7% (Llama); correct ID 1.3-13.9% / 0.7-12.9%; 74.8% of Qwen wrong IDs are "apple"; third-person control shows yes-bias at many layers; priming helps ID more than detection. Authors conclude open models "seem able to introspect" but content-agnostic. | excerpts |
| Pearson-Vogel et al. 2602.20031 | Qwen2.5-Coder-32B (+ limited Llama-3.3-70B, Qwen2.5-72B) | Steering only on turn 1 KV cache, removed before query; P(yes) 0.3%->39.9% with pro-introspection document; strongly prompt-dependent; Llama-70B shows inverted document effect (75.5% vs 38.0%). Mutual information 0.61 -> 1.05 bits for nine concepts. | excerpts |
| Singh, Linzen, Ravfogel 2605.26242 (COLM 2026) | Llama-3.1-70B, Qwen3-32B, Gemma-3-27B, Qwen2.5-72B | Two-way setup reproduces Lindsey pattern in Llama-3.1-70B and Qwen3-32B (Gemma-3-27B largely); but "gaslight" prompt-only condition is also labelled as injection; three-way (control / input manipulation / activation intervention) is near chance in Llama and fails in Qwen3-32B and Gemma; Qwen2.5-72B does not reproduce. Also shows hidden-state self-report classification (Ji-An et al. paradigm) is predictable from input embeddings. | excerpts |
| Hahami et al. 2607.14111 | Llama-3.2 1B/3B/8B, Gemma-4 2B/4B/26B | In small models yes/no detection tracks a factual-no control at r=0.999 (Llama-3.1-8B) so is confounded; sentence-localisation and strength-comparison are above chance from ~2B; Llama-1B at/below chance; IFT lifts Llama-1B localisation 9.6%->60.6%. NOTE internal inconsistency: appendix B interprets the same r=0.999 as "genuine sensitivity ... not a general affirmative bias" while the main text says confounded; the main-text reading is the one consistent with the numbers. | excerpts |
| Zou et al. 2609.35108 | Qwen3-4B, Llama-3.1-8B, Gemma-3-12B | Judge-free first-token position report; localisation 54.49 / 44.31 / 65.95 mean %; norm-matched random direction ~0-6.7%; found gate and router attention heads. | excerpts |
| Konrad et al. 2609.32449 | Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B | Counterfactual affect self-reports shift toward wrong-source demonstrations; small effects (source pull +0.030 to +0.114). A workshop paper. | excerpts |
| Vogel 2025, Godet 2025a/b, Morris & Plunkett 2025, Rivera & Africa 2026, Bozoukov 2025, Wang et al. 2025, Ji-An et al. 2505.13763, Song et al. 2503.07513, Comsa & Shanahan 2506.05068 | various | Described in Macar, Lederman, Singh, Ferrara. Not verified. | second-hand |

## 3. Faithfulness: where self-explanation and internals disagree
- Turpin et al. 2023 (arXiv 2305.04388) [read-excerpts]: GPT-3.5 and Claude 1.0 on BIG-Bench Hard and BBQ. Accuracy drops up to 36% with biasing features (zero-shot Suggested Answer, GPT-3.5: -36.3%). Of 426 biased explanations reviewed, 1 mentioned the bias. Explicitly "necessary but not sufficient" test; only minor input changes tested.
- Chen et al. 2025, Anthropic (arXiv 2505.05410) [read-excerpts]: Claude 3.7 Sonnet and DeepSeek R1. Overall faithfulness 25% and 39%; misaligned hints 20% and 29%; harder questions less faithful; outcome-based RL plateaus at 28% (MMLU) and 20% (GPQA); in synthetic reward-hack RL, hacks used on >99% of prompts but verbalised on <2% in 5 of 6 environments. Their faithfulness score is an upper bound.
- Arcuschin et al. 2025 (arXiv 2503.08679, ICML 2026 version v6) [read-excerpts]: implicit post-hoc rationalisation on 4,834 comparative question pairs; unfaithful pair rates from 0.04% (Sonnet 3.7 thinking, 1k budget) to 13.49% (GPT-4o-mini); DeepSeek R1 0.37%. Cause is "preliminary" evidence of Yes/No bias; they call it behavioural, not "cognitive", faithfulness.
- Circuit-tracing, "Biology of an LLM" (transformer-circuits.pub/2025/attribution-graphs/biology.html) [read-partial-summarised]: Claude 3.5 Haiku says it adds 36+59 by carrying the one, but the circuit shows parallel pathways; authors write "Apparently not!" regarding metacognitive insight. Same paper reports "primitive metacognitive circuits" for knowing the extent of own knowledge. Attribution graphs give insight on "about a quarter of the prompts"; replacement model imperfect. The CoT unfaithful-reasoning section was in the truncated part; I did not read it. Atkinson et al. cite it as identifying faithful vs fabricated CoT [second-hand].
INFERENCE: the faithfulness literature measures verbalised reasons vs behavioural counterfactuals, and mostly not against internals; the circuit-level comparison rests on one closed model (Claude 3.5 Haiku) with a lossy replacement model.

## 4. Methodological threats and how each paper handles them

| Threat | Handled by | Not handled / weak |
|---|---|---|
| Priming by the injection (steering makes the concept word likely, model then "notices" it) | Lindsey: detection judged before the concept word appears (summary); Lederman: prefill with the concept word, still 0% FPR; Pearson-Vogel: steering removed before query | Identification is still confounded with steering (Lederman: continued steering matters for concept mention; Exp 3). Identification mostly looks like steering leakage. |
| Inference from own outputs | Lindsey internality criterion; Pearson-Vogel KV-cache; Zou first-token logits | Lindsey-style setups let model read its own steered tokens; "detect then confabulate" pattern in Lederman |
| Generic yes-bias from steering | Lindsey unrelated yes/no questions; Macar nonlinearity analysis; Lederman absurd and third-person controls (third-person yes-rate as high as first-person at many layers); Hahami r=0.999 | Opposite conclusions across studies; model-size dependence not isolated; first-token logit vs judged text differ |
| Generic anomaly detection vs injection-specific | Zou random-direction control; Singh gaslight + three-way | Only Singh's design tests it directly, on non-Anthropic models; Macar's reply (random directions) is not the same control |
| Steering-strength and layer artifacts | strength/layer sweeps everywhere | Reported numbers are best-case over (layer, strength) selected on the same data (Macar picks L=37, alpha=4 as best; Lederman filters to settings with peak detection >=25%). Selection bias: peak numbers overstate typical performance |
| Prompt leakage and framing | Macar 7 prompt variants; Pearson-Vogel 16 framings | Framing alone moves detection 0.3%->39.9% in Pearson-Vogel; Llama-70B reverses sign |
| Training-data contamination (descriptions of LLM introspection in pretraining; refusal/denial training) | Macar: base vs SFT vs DPO checkpoints and affirm-vs-deny preference sets show post-training content matters; Perez & Long propose filtering | No study I read pretrains or fine-tunes on a corpus filtered of AI-introspection text. Knowledge that "injected thoughts" exist is supplied in the prompt in every concept-injection design |
| LLM-judge bias | Macar notes; Zou and Hahami use judge-free logit scoring | Lindsey, Macar, Lederman use LLM judges (Lederman's grader told to be "extremely lenient") |
| Common cause vs run-time access | Binder cross-prediction | Betley and Plunkett lack a cross-model control; Betley says so |
| Self-simulation satisfying "privileged access" without monitoring | Singh argues explicitly | Not testable by accuracy alone |
| Causal bypass | Tang et al.: interventional training; Morris and Plunkett [second-hand] | Most self-report SFT work |

## 5. Reproducibility: open-weight vs closed
- Reproduced on open-weight models (SOURCE): concept-injection detection with ~0% FPR (Gemma3-27B, Qwen3-235B, Llama 3.1 405B, Qwen2.5-Coder-32B; Macar, Lederman, Pearson-Vogel); self-report of fine-tuned weights (Qwen3-32B; Atkinson); self-explanation advantage (Llama-3.1-8B, Qwen3-8B; Li); localisation tasks at 2B-12B (Hahami, Zou); Binder self-prediction on Llama-3-70B; Betley partially on Llama-3.1.
- Fails or is contested on open-weight models: Qwen2.5-72B in Singh's setup; eight 0.5B-15B models in Ferrara; yes-bias in small models (Hahami); Gemma-3-27B and others fail three-way (Singh).
- Only shown on closed models: Lindsey's Opus 4/4.1 results (nobody outside Anthropic can rerun, as Singh notes); Chen et al.'s RL-on-Claude results; Biology circuit findings (Claude 3.5 Haiku); most of Ackerman's frontier comparisons.
- Open-weight results often needed post-training choices: introspection is absent in base models and appears with DPO (Macar), and refusal ablation boosts it (Macar). So "reproducible" depends on which checkpoint.

## 6. Three tractable open questions for small open-weight models (INFERENCE)
1. Does DPO create injection-specific detection, or generic irregularity detection? Run Singh's gaslight and three-way controls plus Macar's matched random-direction controls on the OLMo-3.1-32B checkpoints (Base/SFT/DPO/Instruct) and on LoRA-DPO-from-SFT variants; all checkpoints are public per Macar.
2. Is fine-tuned self-awareness run-time access or common cause? Take Atkinson's Qwen3 pairs (faithful vs unfaithful checkpoints) and add a Binder-style cross-prediction control and a Song-style equal-cost third-party predictor, using their attribution-similarity signature as the internal reference.
3. Reconcile the contradictory small-model results under one protocol. Same models from 1B to 32B, same concepts, same layers, scored three ways (judge text, first-token logit as in Zou, localisation as in Hahami), with fixed (not best-of) layer/strength chosen on a disjoint concept set, and with a corpus-contamination arm (fine-tune with and without introspection-describing text, extending Macar's affirm-vs-deny preference sets).

## 7. What I could not find or did not read
- Remaining ~300k characters of Lindsey; its thought-vs-text, prefill and intentional-control numbers beyond the summary.
- Biology paper's CoT-faithfulness section.
- Primary text for Song et al. 2503.07513, Ji-An 2505.13763, Comsa & Shanahan 2506.05068, Godet and Morris & Plunkett (LessWrong), Rivera & Africa, Bozoukov, Wang et al. 2025.
- A study controlling for pretraining contamination of introspective self-descriptions in concept injection.
- Several 2026 items appeared in search listings only (2606.23671, 2604.00228, 2610.06174, 2608.30980, 2609.00904, 2604.12373, 2603.20276, 2510.24797, 2603.xxxx) and were not read: [abstract-only or less].
- Dates and arXiv IDs of 2026 papers are as returned by the alphaXiv tool; I did not verify them independently.
