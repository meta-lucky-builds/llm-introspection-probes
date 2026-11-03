# Literature thread: where can a small independent effort contribute? (as of 2026-11-04)

Evidence labels. [read-full] = full text obtained (raw text or alphaXiv extraction). [read-report] = alphaXiv AI-generated report of full paper (not raw text). [abstract-only] = abstract/summary from arxiv page via WebFetch (a small model paraphrased it; not verbatim). [web-snippet] = WebSearch result summary, secondary. [doc-fetch] = official doc page fetched, answered by a small model (quotes are as relayed). Items marked INFERENCE are mine.

## 0. Caveats
- Several 2026 arXiv IDs (26xx.xxxxx) come only from the alphaXiv index; I did not independently verify them off alphaXiv. I treat them as data. One item (2609.35799, an "OpenAI-HuggingFace" breach reproduction) I did not open and do not rely on.
- Kaggle official docs pages return only titles to WebFetch (JS-rendered). Kaggle quota/session numbers below come from search-result summaries quoting official Kaggle pages, not from a fetch I could read. Verify in your account.
- Colab free GPU specs are not published (see 4.3).
- I found no 2026 Anthropic, Apollo, Goodfire or Transluce agenda document by search; absence in search is not absence in the world.

## 1. Stated open problems

### 1.1 Sharkey et al., "Open Problems in Mechanistic Interpretability", arXiv 2501.16496 (Jan 2025) [read-full: appendix A.1-A.2 read verbatim from full text; body via report]
Synthesis by Apollo Research for Schmidt Sciences; authors from Anthropic, GDM, MIT, Harvard, EleutherAI, FAR, Goodfire, Timaeus, MATS and others (report).
Items from Appendix A that a small compute-limited effort can plausibly touch (verbatim question text, abbreviated):
- A.1.1.1d "Can we develop a wider suite of networks with known ground truth explanations to validate techniques against?"
- A.1.1.1f "...engineering goals in a way that improves upon black box baselines?"
- A.1.1.2 "Can we develop 'model organisms' as a community..."; A.1.1.3 standardized baselines and benchmarks "where the ground truth is known"; A.1.1.4 stress tests for interpretability illusions; A.1.1.5 average-case and worst-case evaluation.
- A.1.2.1 "How can we reliably distinguish causal from merely correlated features when probing"; A.1.2.3 regularization/validation to prevent spurious probe correlations.
- A.2.1.1b "distinguish between features that merely recognize deceptive behavior versus mechanisms that generate" it; A.2.1.3a mechanistic anomaly detection.
- A.2.2.1a "make activation steering more precise and reduce its side effects"; A.2.2.3b "analysing differences between model versions" (finetuning diffing).
- A.2.3.6b "detect when finetuning has masked rather than removed capabilities"; A.2.3.4b "How do model mechanisms evolve dynamically during training?"
- A.2.6.1a whether SDL and circuit analysis transfer to SSMs.
Body (report): lists SAE problems: reconstruction error, feature splitting/absorption, dataset dependence; says model organisms ideally "open source, easy and cheap to use" (I read lines ~885-890 of the full text).

Have any been "solved"? INFERENCE from the 2026 results below: none is solved. Partially advanced: ground-truth testbeds (2610.19907, 2602.14687), model-organism suites and auditing benchmarks (2602.22755, 2610.10203), white-box vs black-box comparison (2602.22755). Several newer papers report the opposite of solved (probe monitoring fails controls, 2606.30449; SAE benchmarks unreliable, 2605.18229).

### 1.2 Google DeepMind: "A Pragmatic Vision for Interpretability" (Nanda et al., AF, 1 Dec 2025) [doc-fetch of AF post, first 100k of ~102k chars]
Source claims (relayed by fetch model): team shifted from ambitious reverse-engineering to problems on the critical path with proxy tasks; "disappointed by the amount of progress" from ambitious work; SAE reconstruction error did not show usefulness while downstream tasks did; recommends "method minimalism" (prompting, steering, probing, CoT reading, prefill) before complex methods; time-box exploration ~2 weeks; settings: reasoning-model internals, real-world concerning behaviours, model psychology; avoid GPT-2-small-style toy settings without a convincing proxy; "Simple methods need less compute, so they suit researchers outside AGI companies."
Gemma Scope 2 post (AF, Dec 2025) [doc-fetch]: team says it has "deprioritized fundamental research on tools like SAEs" but released tools for the community.
MATS Summer 2026 Nanda stream page [doc-fetch]: pragmatic interpretability and control/monitoring; example topics: detecting deception, why a model took a concerning action, stopping models realising they are tested. No project list found.

### 1.3 OpenAI
- "Open Sourcing Monitorability Evaluations", 23 Apr 2026, alignment.openai.com/monitorability-evals [doc-fetch]: released most CoT-monitorability datasets (GPQA, AIME, Sandbagging, Scruples, etc.) and g-mean2 reference code (repo openai/monitorability-evals); stated open question: whether CoT monitoring stays reliable as training/inference methods change; a selection-bias (winner's-curse) point: a cross-fit filter showed an apparent late-RL monitorability drop in Sandbagging was largely an artifact. Withheld sets include FrontierMath, Anti-Scheming.
- Other OpenAI items (Sept 2026 chief-scientist remark, weight-sparse circuits attribution) [web-snippet, unverified]. Do not cite.

### 1.4 Anthropic
- Open-sourced circuit-tracing (29 May 2025) with Neuronpedia frontend; invites outside researchers to find circuits [web-snippet of anthropic.com/research/open-source-circuit-tracing]. No 2026 agenda found.
- Anthropic Fellows output visible in 2026 literature: AuditBench (2602.22755), introspection (2601.01828, 2603.21396), self-modeling (2608.30980), explanation evaluation via counterfactuals (2608.16747), auditor RL (2608.25460).

### 1.5 Apollo, Goodfire, Transluce
Search found nothing on 2026 agendas. Apollo is the synthesizer of 2501.16496. Goodfire reported $150M Series B at $1.25B valuation, Feb 2026 [web-snippet]. No claim beyond that.

## 2. Results that reshaped priorities (2025-2026)
Each: source claim, then what changed.
1. Pragmatic pivot at GDM (above). Changed: SAE work is judged on downstream tasks vs baselines; "beat the simple baseline" is now the bar. [doc-fetch]
2. AuditBench, arXiv 2602.22755 (Anthropic Fellows, Feb 2026) [abstract-only]: 56 models with implanted hidden behaviours across 14 types; "tool-to-agent gap"; agent performs best with black-box tools; white-box can help; easier to audit if trained on synthetic documents than demonstrations. Changed: evaluate tools inside an auditing loop, not standalone; models and framework said to be released (link not retrieved).
3. "How to train your model organism", 2610.10203 (Northeastern, Oct 2026) [abstract-only]: organisms validated only on behaviour installation; two public suites showed reduced conversational quality and naturalness; proposes validating on behaviour, capability, naturalness; DPO stays closer to base than SFT; model-merging recipe. Changed: results from unnatural organisms may not transfer; a validation-metrics contribution is open.
4. Emergent misalignment (EM) and its fragility. Model organisms for EM, 2506.11613 [abstract-only]: ~0.5B models, rank-1 LoRA suffice. "An Emergent Mirage", 2607.09053 (UMD/CMU, Jul 2026) [read-full]: on Qwen2.5-14B-Instruct, apparent rapid realignment largely disappeared after controlling response length; LoRA-space "phase transition" signature did not consistently correlate with behaviour. Changed: EM claims need length/dataset controls; replications with controls are valued. Field is crowded (many Aug-Oct 2026 EM papers appeared in index; INFERENCE: novelty risk high for generic EM mechanism papers).
5. Introspection. 2601.01828 (Lindsey, Anthropic) [abstract-only]: concept-injection detection exists but "highly unreliable and context-dependent"; Opus 4/4.1 best. 2603.21396 "Mechanisms of Introspective Awareness" [abstract-only]: open-weights models; emerges from post-training, DPO elicits it, SFT does not; two-stage circuit (evidence carriers + gate features); ablating refusal directions raises detection by 53%, trained bias vector by 75%. Changed: introspection became mechanistically tractable on open-weights models, i.e. accessible to small teams. Follow-ups: 2607.14111 small-LLM introspection fine-tuning, 2610.38911 causal bypass [titles/snippets only].
6. Probe monitoring negative results, 2606.30449 (ICML 2026 AIWILD) [abstract-only]: probes decode the situation not the action; best future-behaviour probe only modestly above majority class on Llama-3.1-8B; steering loses specificity on Gemma-3-27B-IT. Changed: monitoring claims must pass scenario generalization and concept-specificity controls. Related: 2609.24821, 2609.18080 [snippets only].
7. SAE evaluation turmoil [snippets only]: 2602.14111 (sanity checks vs random baselines), 2605.18229 ("Are SAE benchmarks reliable?"), 2602.14687 (SynthSAEBench), 2607.19386 (pipeline choices dominate autointerp score variance), 2605.31183 (SAEs can beat simple steering baselines on AxBench, contra earlier). Mixed evidence; changed: SAE claims need controls and multiple pipelines.
8. Ground-truth testbed: Sparse Transformer, 2610.19907 [read-report, partial]: 5-layer TinyStories model with TopK-sparse MLP; neurons 43% better on AutoInterp than loss-matched dense; transcoder recovers neuron vectors with 52.6% higher cosine similarity than SAE. Code and models on GitHub/HF (AntonKorznikov). Directly fits Sharkey A.1.1.1d. CPU/T4-scale.
9. Gemma Scope 2 (Dec 2025) [doc-fetch AF post]: Gemma 3 270m-27b, PT and IT; SAEs on resid, MLP out, attention out; MLP transcoders (with/without skip); CLTs only 270m and 1b; 16k and 262k widths at three L0 bands; HF google/gemma-scope-2; Neuronpedia demo. Changed: transcoder/attribution-graph work possible on small Gemma 3 models.
10. CoT monitorability evals open-sourced (OpenAI, Apr 2026; above). Changed: shared, reusable eval sets for CoT-monitor research.
11. Evaluation awareness in open-weight models [snippets only]: 2606.29196, 2609.22119, 2607.25907, 2609.01611 (EvalDetectBench). Cheap to replicate across sizes.

## 3. Tooling and assets (all [doc-fetch] unless stated)
- TransformerLens (github.com/TransformerLensOrg/TransformerLens): README claims 15,000+ models across 140+ architecture families via TransformerBridge; legacy from_pretrained API "removed in TransformerLens 4.0"; Mamba/SSM experimental; maintained (Bryce Meyer, Jonah Larson); 2026 footer. Releases section was empty in fetch; version unverified.
- nnsight (github.com/ndif-team/nnsight): local default; remote=True runs on NDIF with an API key (example Llama-3.1-8B); NDIF cost/quotas not stated; check nnsight.net/status. ~2,709 commits, active.
- SAELens (github.com/decoderesearch/SAELens): "v6" major refactor; TransformerLens integration; pretrained list on separate docs page (not retrieved).
- circuit-tracer (github.com/decoderesearch/circuit-tracer; moved from Anthropic-hosted org): MIT license; transcoders listed: Gemma-2 2B (PLT, CLT), Llama-3.2 1B, Qwen-3 0.6B/1.7B/4B/8B/14B PLTs, GPT-OSS 20B CLT, Gemma-3 270M/1B/4B/12B/27B PLTs (need nnsight backend), Llama-3.1-8B-Instruct TopK PLTs. README: "Working with Gemma-2 (2B) is possible with relatively limited GPU resources; Colab GPUs have 15GB of RAM." Demos run on Colab except Llama.
- Gemma Scope 2: see 2.9. Neuronpedia hosts features.
- Model organisms: EM organisms from 2506.11613 (0.5B rank-1 LoRA; HF checkpoint release not confirmed by fetch). AuditBench 56 models (2602.22755; base models and links not retrieved). Sparse Transformer (2610.19907).
- Behaviour eval datasets: openai/monitorability-evals (GitHub); EM eval used 8 benign questions x 50 samples at T=1 with LLM judge (from 2607.09053 [read-full]); StrongReject was a poor EM signal there.
- Model sizes: INFERENCE: 270M-4B Gemma 3 and Qwen3 0.6-4B fit a 16 GB T4 in fp16 for activations work; 8B needs fp16 ~16 GB weights so quantize or use NDIF; 14B+ no. Not from a source.

## 4. Compute

### 4.1 GitHub Actions hosted runners, private repo
- Job limit: "Each job in a workflow can run for up to 6 hours of execution time." Workflow run: 35 days. Concurrent jobs, Free: 20. (docs.github.com/en/actions/reference/limits) [doc-fetch]
- Standard runners, private repos: ubuntu-latest 2 vCPU, 8 GB RAM, 14 GB SSD (x64); ubuntu-slim 1 vCPU, 5 GB; arm64 2 vCPU 8 GB. Public repos get 4 vCPU/16 GB. (docs.github.com/en/actions/reference/runners/github-hosted-runners) [doc-fetch]
- Billing: Free plan included compute $12/month, artifact storage 500 MB, cache 10 GB/repo; Linux 2-core $0.006/min, 1-core $0.002/min. INFERENCE: $12 / $0.006 = 2,000 min/month on 2-core Linux (matches a third-party claim of 2,000 min [web-snippet]); ubuntu-slim would stretch further but 5 GB RAM.
- GPUs: standard runners have none. GPU runners are "larger runners"; docs say they require Team/Enterprise plan for larger runners and are billed per minute; a third-party/mirror lists $0.052/min Linux 4-core GPU and says larger runners never draw on included minutes [web-snippet, unverified]. Conclusion: no free GPU on Actions.
- Non-interactive: yes by design; 6 h/job. Public repo would be free and unlimited on standard runners but the target repo is private.
- Fits: CPU-only work on tiny models (Sparse Transformer, TinyStories-scale, GPT-2-small, Pythia-70M-160M, Gemma-3-270M inference with a few hundred prompts), aggregating results, stats, plotting, LLM-judge calls via API. Does not fit: anything needing a GPU, 1B+ with long sequences (8 GB RAM, CPU speed).

### 4.2 Kaggle
- Quota: weekly GPU quota "30 hours or sometimes higher" with a "floating" quota; session limit raised from 9 h to 12 h for CPU/GPU notebooks, 9 h for TPU [web-snippet quoting Kaggle product-feedback posts 173129 and 302908; page text not readable by fetch]. GPU types: reports of 2x T4 or P100 [web-snippet, third-party]. Colab Pro linkage adds GPU hours [web-snippet].
- Non-interactive: yes. Kaggle CLI [doc-fetch of Kaggle/kaggle-cli docs]: `kaggle kernels push -p <FOLDER> [--accelerator ...] [-t <SECONDS>]`, `kaggle kernels status <owner/slug>`, `kaggle kernels output <owner/slug> -p <path>`. kernel-metadata.json: kernel_type script|notebook; enable_gpu default false; enable_internet default false; machine_shape e.g. NvidiaTeslaT4, NvidiaL4, TpuV5E8. Accelerator list in push docs includes A100/H100 but "some are limited to specific competition participants or admins". Requires a Kaggle API token (the Kaggle MCP in this environment is not authorised; user must do /mcp auth).
- Internet default off; enabling it needs account settings (INFERENCE from general knowledge, unverified here), which matters for downloading HF weights (workaround: attach as Kaggle datasets).
- Fits: 1B-4B model activation caching, SAE/transcoder inference with Gemma Scope 2 (270m-4B), circuit-tracer on Gemma-2-2B, LoRA fine-tuning of 0.5B EM organisms, 7-8B in 4-bit.

### 4.3 Google Colab free
- Official FAQ [doc-fetch]: limits not published ("Colab does not publish these limits"); GPU types vary; "notebooks can run for at most 12 hours, depending on availability"; runtimes time out if idle; background execution listed as a benefit of paid plans; Colab "prioritizes interactive compute". FAQ search snippet: free runtimes may be terminated without warning; prohibited uses include using multiple accounts to get around limits [web-snippet].
- Non-interactive: not on free tier. Not suitable for unattended CI-style runs. (A colab-mcp tool exists in this environment; I did not test it.)
- circuit-tracer README says its demos run on free Colab GPUs (15 GB).

## 5. Recommended problems for this setup (INFERENCE, ranked)
1. Controlled replication/stress-test of model-organism claims (EM, hidden-behaviour organisms): length/format confounds, naturalness validation (2607.09053, 2610.10203; Sharkey A.1.1.2-5). Compute: 0.5B rank-1 LoRA on Kaggle T4; evaluation by API judge. Risk: crowded; value comes from controls and negative results.
2. Baselines-first monitoring study: do simple probes/steering/prompting beat SAE-feature monitors on small models under scenario and concept-specificity controls (2606.30449; GDM method minimalism; Sharkey A.2.1.3). Compute: Gemma-3 1B-4B + Gemma Scope 2, Kaggle.
3. SAE/transcoder evaluation on ground-truth testbeds (2610.19907, 2602.14687, 2605.18229; Sharkey A.1.1.1d, A.1.1.3). CPU-feasible on Actions plus occasional T4.
4. Introspection mechanisms across open-weight sizes and post-training recipes (2603.21396, 2601.01828; small-model gap): concept-injection detection vs size/DPO/SFT, with false-positive controls. Needs 1B-8B on T4. Risk: small models may show near-zero effect (INFERENCE).
5. Evaluation-awareness and CoT-monitorability replication on open reasoning models using public datasets (openai/monitorability-evals; 2606.29196). Needs reasoning-capable open models (size not checked); API/judge cost.
Avoid: generic SAE reconstruction optimization, GPT-2-small circuit hunting without a proxy task, anything needing >16 GB VRAM.

## 6. Unverified / not done
Kaggle official quota text; Colab GPU specs; GitHub plan eligibility for GPU runners on Free private repos; AuditBench release location and base models; EM organism HF checkpoints; Anthropic/Apollo/Goodfire/Transluce 2026 agendas; MATS/Alignment Forum project lists beyond Nanda stream page; whether any Sharkey problem was formally declared solved.
