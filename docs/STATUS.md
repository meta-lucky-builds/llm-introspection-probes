# Mission status

Read this first when resuming. Keep it short and current. Last updated 2026-11-04.

## Mission
Long-running research on alignment, mechanistic interpretability, safety, LLM behaviour and introspection. The owner is away and authorized autonomous, safe operation.

## Rules
- Never run tests or experiments locally. Small experiments run on Colab through the Colab MCP. Large ones: write and push the code, and run them with the owner later. Unit tests run on GitHub Actions.
- Colab, not Kaggle. Never put a token or secret in a Colab notebook. The repo is private, so code reaches Colab as a pasted bundle (`scripts/make_colab_bundle.py`), and results leave through cell output and get committed here.
- Do not ask before creating branches, committing or pushing in this repo. Do not merge to `main` without the owner.
- If compute fails, keep going: write code, push, move to other work.
- Models: Opus 5.5 for implementation and thinking; Sonnet 5.5 to execute finished code, simple tasks and web search; Haiku 5.5 for simple retrieval. All at medium effort. Tell agents to use the how-to-think skill, and use it when forming hypotheses, planning, analyzing or designing.
- Preregistrations are frozen once data collection starts. Report failures and nulls as they are.

## Usage habits
- Check plan usage before launching agents. At 90% of the 5-hour limit or above, or 85% of the weekly limit or above, start no new agents; finish what is running, save state here, and wait for the reset.
- Scope agents narrowly. Send paper reading to agents rather than the coordinator, and do not repeat retrievals already recorded in `docs/literature/`.

## Avoiding local optima
- Every experiment gets a stop rule before it runs: what result ends this line of work, and what comes next.
- Do not tune a null into a positive. A failed preregistered test is logged and closed; a follow-up needs its own preregistration and a new reason.
- After two consecutive experiments on one question teach nothing new, change the question, the method or the level of analysis.
- At each result, reread the hypothesis ledger, add what the result suggests, and pick the next experiment by expected information per unit of compute. Keep the ledger spanning more than one research direction.
- Pivoting changes what we study. It never relaxes rigor or these rules.

## Context habits
- Keep findings in the repo, not in the conversation. Have agents write notes to files and return short summaries.
- Read only the parts of files and papers that a decision needs.

## Current state
- Branch: `research/e1-localization-capture`.
- E1 preregistered ([prereg](preregistrations/E1_localization_capture.md)); no data collected.
- Harness implementation in progress.

## Next steps
1. Finish the E1 harness; independent code review.
2. Push; pass tests on GitHub Actions.
3. Run the E1 pilot, then the small tier, on Colab (CPU, 12 GB at last check). Commit results.
4. Analyze against the prereg's decision rules; update [hypotheses](hypotheses.md).
5. Preregister and run the next candidate from the ledger; write GPU-tier code for later.

## Waiting on the owner
- What word "sleep framework" was a typo for.
- The two context files mentioned in the original request, which never arrived.
- A GPU runtime in Colab would widen what can run; optional.
