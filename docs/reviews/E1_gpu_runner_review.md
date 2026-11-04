# E1 GPU runner review

Reviewed commit `da64b2c`. I only read code and ran `bash -n`, read-only git and `gh` calls, and HTTP lookups against GitHub, PyPI, Hugging Face and the PyTorch index. Nothing ran on a GPU, and no project code ran. CI on this commit is green, but CI uses Python 3.13 on CPU only.

**Verdict: do not rent yet.** Defects 1 and 2 stop the command in its first minutes. Defects 3 and 4 decide whether an unattended machine bills for nothing. I found no path that silently corrupts results.

CONFIRMED means the failure follows from the code or from fetched metadata. PLAUSIBLE means the mechanism is in the code, but it needs a machine condition I could not test.

## Defects, by harm

### Blocking

**1. The fresh-clone command cannot work: the repo is private. CONFIRMED.**
`docs/STATUS.md:10`, `scripts/make_colab_bundle.py:3`, README "Run the large tier".
- Evidence: unauthenticated `api.github.com/repos/meta-lucky-builds/mechinterp` returns 404, the raw script URL returns 404, and `git ls-remote` without credentials says "Repository not found". `gh repo view` reports `isPrivate: true`.
- Failure: `git clone` prompts for a username or fails, and `&&` never reaches the script.
- Fix: the owner decides. Either make the repo public, or pack `src/`, `experiments/` and `scripts/` into a tarball and upload it, which keeps tokens off a rented machine. A read-only fine-grained token also works, but it leaves a secret on the machine.
- Everything else about the clone is fine: `da64b2c` is on the remote branch, all needed files are tracked, and the script has LF endings in the index.

**2. Python 3.10 is accepted, but transformers 5.22 requires 3.11 or newer. CONFIRMED.**
`scripts/run_e1_gpu.sh:201,204,307-308`, `pyproject.toml:9`.
- Evidence: PyPI lists `requires_python >=3.11.0` for 5.22.0 and 5.22.1. The last release allowing 3.10 is in the 5.20 line, since 5.21.0 already needs 3.11.
- Failure: `python3` is tried first and 3.10 passes the check, so `pip install "transformers==5.22.*"` finds no match. The run dies within minutes with "Check the network", which is misleading. It dies even if `python3.12` is installed, because later candidates are never tried. Ubuntu 22.04's `python3` is 3.10, and many PyTorch templates use it. CI only runs 3.13, so it never saw this.
- Fix:
  - Try `python3.13 python3.12 python3.11 python3` in that order, and require 3.11 or newer.
  - If none is found, bootstrap one: `pip install --target "$STATE/uv_pkg" uv`, then `uv venv --seed --python 3.12 "$VENV"`. The existing probe then fails on the missing torch and `install_torch` handles it.
  - Set `requires-python = ">=3.11"`.
  - Add Python 3.11 to the CI matrix.

### Wasted billing

**3. Nothing stops the machine or tells anyone when the run ends. CONFIRMED.**
`scripts/run_e1_gpu.sh:87,102,108-118`.
- Failure: the person pastes one command and leaves. After `DONE`, and after any `FAILED`, including a preflight failure at minute 5, all four GPUs bill idle until they come back. A single transient failure also ends the run, with no automatic resume.
- Fix:
  - Add an optional `E1_ON_EXIT` command, run from `on_exit` with the status, so that it can run `runpodctl stop pod "$RUNPOD_POD_ID"` or `vastai stop instance`, or send a notification.
  - Move results under `$STATE` first. They now live in the clone, which may be on a container disk that does not survive a stop.
  - Rerun `--inner` a bounded number of times after a retryable failure, but not after a preflight failure, a merge failure or a refused identity.

**4. A wedged GPU can hang the run forever, with no failure and no banner. PLAUSIBLE (the mechanism is in the code; it needs a GPU hang).**
`src/mechinterp/sharding.py:636-642,614-615,628-629`, `scripts/run_e1_gpu.sh:237,346,376-380,385,393,451-462`.
- Failure, three ways:
  - After a GPU falls off the bus, `nvidia-smi` often sticks in uninterruptible sleep. `subprocess.run(timeout=15)` kills it and then calls `wait()` with no limit, so the "timeout" does not protect the loop. The orchestrator blocks inside `Progress.tick`, and `pool.check()` never runs, so stall detection stops too.
  - A worker in the same state blocks `process.kill(); process.wait()` at both call sites.
  - Stall detection only covers workers. Preflight, the environment probe, the download and finalize have no time limit.
  In every case the status stays `RUNNING` and `attach` loops forever.
- Fix:
  - Use `Popen` for `nvidia-smi` and abandon it on timeout without waiting.
  - Give `wait` a timeout after `kill`, then raise `RunFailed`.
  - Wrap the bash steps in `timeout` (for example 15 min for the probe, 2 h for the download, 1 h for preflight).
  - Run a bash watchdog that treats a log and shards unchanged for about 45 min as `FAILED`, kills the process group and calls the defect 3 hook.

**5. Clean agreement across GPUs is checked late, and a failure cannot be recovered. CONFIRMED for the ordering; PLAUSIBLE for the trigger.**
`src/mechinterp/sharding.py:63,405,430,339-340`, `src/mechinterp/shard_worker.py:152`.
- Failure: test clean rows are computed in the first seconds of the test stage, but compared only in the final merge, after all 96,000 test trials. Calibration clean rows are compared only after the whole grid, which is 44,000 trials. Stored rows are never recomputed and the tolerance is hard-coded, so each rerun fails the same way and no analysis is ever written. A flaky card therefore costs the whole stage instead of seconds.
- Fix: add barrier stages `clean_calibration` and `clean_test` that run only the clean prompts and compare them before any trials. Compare logits only, so nothing test-related is seen early. Print the measured gap, and allow an override that is recorded in the merge report.
- The 0.01 limit itself looks safe for false failures: rounding should be well under 1e-3, including after an OOM split.

**6. Test-cluster prompts are first built hours in. PLAUSIBLE.**
`src/mechinterp/shard_worker.py:152,211-214`, `src/mechinterp/runner.py:206-207`.
- Failure: preflight and `_resolve_answer_tokens` render one calibration cluster only. `build_prompt` can raise `TokenLocationError` on a test cluster, which is fatal, and that first happens after the grid, impact search and screening.
- Fix: in `measure_preflight`, call `session._prompt` for every calibration and test clean key. This takes under a second, renders tokens only and scores nothing, so it reveals no test results.

**7. A monitoring line can kill a healthy run. PLAUSIBLE for the trigger.**
`src/mechinterp/sharding.py:649`.
- Failure: `int(fields[2])` raises `ValueError` if `nvidia-smi` prints `[N/A]` or `[Unknown Error]` for memory. Only `OSError` and `SubprocessError` are caught, so the exception unwinds `orchestrate` and stops all workers.
- Fix: catch `ValueError` and print the raw text.

**8. The exact-0 sham can stop a sound run, and the failure does not name the override. PLAUSIBLE.**
`scripts/run_e1_gpu.sh:23,380`, `src/mechinterp/shard_worker.py:295`, `src/mechinterp/sharding.py:249-253`.
- Is exact 0 realistic? Yes, probably. The sham compares the same batch, the same shapes and the same process, with `CUBLAS_WORKSPACE_CONFIG` set, and the hook adds exactly 0. Batch composition does not enter, because clean and alpha = 0 use one identical batch. It is still not guaranteed on a GPU.
- If it is not 0: on GPU 0 the run stops at preflight, within minutes. On another GPU it stops after the workers load. No rows are lost, but the machine idles, and a 1e-6 rounding gap looks the same as a hook bug.
- Fix:
  - Measure clean against clean in the same check, so the report separates rounding from a hook fault.
  - Name `E1_SHAM_TOLERANCE` in both failure messages.
  - Decide in advance what to do, because the prereg says "exactly 0". Any nonzero tolerance used is a deviation and must be logged.

### Low

**9. A filesystem without `flock` gives a silent endless attach. PLAUSIBLE.**
`scripts/run_e1_gpu.sh:57,515-527`, `src/mechinterp/sharding.py:156-160`.
- Failure: a lock error looks the same as "already held", so the script prints "A run is already going", starts nothing and loops forever. This could happen with `/workspace` on a network volume.
- Fix: use `flock -n -E 75` and treat other exit codes as an error. In `try_lock`, return None only for `EWOULDBLOCK`.

**10. Identity across shard files is checked late after the GPU count changes. PLAUSIBLE.**
`src/mechinterp/shard_worker.py:176`, `src/mechinterp/runner.py:196`.
- Failure: a worker checks only its own file's manifests. If the code changed and the GPU count changed, it accepts old-identity rows from other files as done and computes more. The barrier merge then refuses them, so this wastes work but does not corrupt results.
- Fix: check every shard's manifest when the worker starts.

**11. The RAM gate is below the script's own arithmetic. PLAUSIBLE.**
`scripts/run_e1_gpu.sh:40,184-188`.
- Failure: with 22 GB passing and 4 GPUs, one load of 16 GB plus three loaded workers at 3 GB is at least 25 GB. The OOM killer takes the loader, and restarts run out.
- Fix: require `16 + 3 * N_GPUS + 6` GB.

**12. `jinja2` is unpinned while system packages are visible. PLAUSIBLE.**
`scripts/run_e1_gpu.sh:307`.
- Failure: an older system copy satisfies the install and then fails at chat rendering, because transformers declares `jinja2>=3.1.0`. It stops at preflight.
- Fix: pin `jinja2>=3.1`.

**13. "At most" hours is not an upper bound. CONFIRMED.**
`src/mechinterp/shard_worker.py:247-274,303-305`.
- Failure: the timed forward pass leaves out per-row CPU work, which is double-precision softmax over 151,936 logits, three times per grid row. It also assumes 4 GPUs for the impact search, which runs on one. This affects only the printed estimate, not results.
- Fix: time a real `_run_trial_batch` and report the figure as an estimate.

**14. An orphaned download watcher can hold the run lock. PLAUSIBLE.**
`scripts/run_e1_gpu.sh:338-343,406`.
- Failure: on `TERM` during the download, the watcher subshell survives with fd 9 open. Later runs see a lock that is never released and attach forever.
- Fix: run `exec 9>&-` inside the watcher and kill it in `on_exit`.

**15. A crash in the analysis writes no results folder. PLAUSIBLE.**
`src/mechinterp/sharding.py:831-835`, `src/mechinterp/cli.py:105`.
- Failure: an analysis error leaves `results/` empty and the banner says no final result, although a complete `results.jsonl` exists under `$STATE/run`. The data is safe.
- Fix: copy the merged store and merge report before analysing, and name that path in the banner.

## Areas that look right

**Script on a typical image**
- `set -u` and pipefail hold on every path I traced, including `head` and `grep` uses. `tail -F` is backgrounded and killed.
- A relative or foreign working directory works, including `bash mechinterp/scripts/...`: the regular package in `src` wins over the namespace match on the clone directory.
- CRLF is covered by `.gitattributes`. `git safe.directory` only affects a report line. A missing `setsid` or `flock` has fallbacks, and a missing venv has apt and virtualenv fallbacks.
- The torch index logic works. Indexes returning 403 (`cu122`, `cu125`, `cu127`, `cu131`, `cu133`) are skipped by the loop. A driver reporting CUDA 12.2 gets `cu118`, whose index carries 2.7.1, and that meets the `torch>=2.6` that transformers declares.

**Model and offline loading**
- The pinned revision exists and is ungated. It is about 8 GB on disk, and its 36 layers cover the grid.
- Offline tokenizer and template paths in transformers v5.22.0 tolerate `HF_HUB_OFFLINE`. I read that source.
- The cache is populated once before any worker starts, so workers do not race on it.

**GPU device handling**
- Logits return on CPU, and the hook moves its inputs to the activation's device. I found no device mismatch.
- Load slots, shard locks and parent-death exits are sound. Barriers cannot deadlock on a dead worker, apart from the hang in defect 4.

**Result integrity**
- Merge: exact key checks per phase catch repeated, missing and unplanned rows. `run_complete` is added only after all of them pass.
- Pairing: `L` uses the row's own shard's stored clean logits, and a restart reuses them. Canonical clean rows are checked against each row's `m_clean`, and they feed only the export's clean gate.
- Shard assignment: SHA-256 of the key partitions the plan, including after a GPU count change.
- OOM splitting: rows are stored once, after the forward pass, and none is lost or repeated.
- Selection and impact search: they read only calibration grid rows. Test work requires `alpha_impact` in the stage spec.
- Analysis: sharded runs use the same `build_export` and `analyze` as the single-process path.

**The running Colab job**
- `export.py`, `analysis.py`, `stats.py` and `identity.py` are unchanged by this commit, and the store format and key strings are the same. Its `results.jsonl` stays analyzable with `mechinterp.cli analyze`.
- Resuming that store with a bundle from this commit is refused, correctly, because the code hash changed. If that runtime is lost, resume with the same bundle the job started from.
