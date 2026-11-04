#!/usr/bin/env bash
# One command for E2 on the same rented multi-GPU machine as E1, after E1 is DONE:
#
#     bash scripts/run_e2_gpu.sh
#
# It refuses to start while E1 is running or unfinished. It reuses E1's venv and model cache, downloads the pinned
# lens once, checks the lens, runs Stage 1's gate, and stops there if the gate fails. Otherwise it runs Stage 2 and
# Stage 3 across all GPUs and writes the results folder. It keeps running if the terminal closes. Running it again
# attaches to a live run, resumes an interrupted one, or prints the final state of one that ended.
#
# Optional environment:
#   E2_ON_FINISH        Shell command to run after the final state is saved, for example to stop the machine. It sees
#                       E2_FINAL_STATUS: DONE, GATE_FAILED or FAILED. Default: nothing
#   E2_RETRY            Set to 1 to resume a run that ended as FAILED. Finished trials are kept
#   E2_STALL_MINUTES    Kill a loaded worker after this long with no finished trial. Default 20
#   E2_CONFIG           E2 config file. Default: experiments/e2_jspace_vector_families/configs/large_4x3090.json
#   E2_E1_STATE_DIR     E1's state directory. Default: E1_STATE_DIR, or wherever run_e1_gpu.sh would have put it
#   E2_STATE_DIR        Checkpoints and results. Default: e2_qwen3_4b inside E1's state directory
#   E2_RESULTS_DIR      A second copy of the results. Default: results/e2/qwen3_4b inside the clone
#   E2_SHAM_TOLERANCE   Largest |L| accepted at alpha = 0. Default 1e-4, as in E1
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)"
CONFIG="${E2_CONFIG:-$REPO_ROOT/experiments/e2_jspace_vector_families/configs/large_4x3090.json}"
RESULTS_COPY="${E2_RESULTS_DIR:-$REPO_ROOT/results/e2/qwen3_4b}"
SHAM_TOLERANCE="${E2_SHAM_TOLERANCE:-1e-4}"
STALL_MINUTES="${E2_STALL_MINUTES:-20}"
ON_FINISH="${E2_ON_FINISH:-}"
if [[ -n "${E2_E1_STATE_DIR:-}" ]]; then
  E1_STATE="$E2_E1_STATE_DIR"
elif [[ -n "${E1_STATE_DIR:-}" ]]; then
  E1_STATE="$E1_STATE_DIR"
elif [[ -d /workspace && -w /workspace ]]; then
  E1_STATE=/workspace/e1_large_qwen3_4b
else
  E1_STATE="$REPO_ROOT/.e1_state/large_qwen3_4b"
fi
STATE="${E2_STATE_DIR:-$E1_STATE/e2_qwen3_4b}"
RUN_DIR="$STATE/run"
VENV="$E1_STATE/venv"
LOG="$STATE/run.log"
LOCK="$STATE/run.lock"
PID_FILE="$STATE/run.pid"
STATUS_FILE="$STATE/status" # First line: the state. The rest: the banner to show again.
ENV_REPORT="$STATE/environment.txt"
PROBE_OUTPUT="$STATE/probe.txt"
GATE_FILE="$RUN_DIR/stage1/gate.json"
RESULTS_DIR="$STATE/results"
SUMMARY="$RESULTS_DIR/summary.txt"
DRY_RUN_JSON="$RUN_DIR/preflight/dry_run.json"
FAILURE_REASON_FILE="$RUN_DIR/failure_reason.txt"
EXIT_GATE_FAILED=10
MIN_DISK_GB=5
MIN_GPU_FREE_MIB=18000
MIN_GPU_FREE_TEXT="18 GB"
NVIDIA_SMI_SECONDS=60
PROBE_SECONDS=900
DOWNLOAD_ATTEMPT_SECONDS=1800
STAGE1_SECONDS=$((4 * 3600))
STAGE1_IDLE_SECONDS=2700
PREFLIGHT_SECONDS=3600
FINALIZE_SECONDS=1800
ON_FINISH_SECONDS=600
RUN_MAX_SECONDS=$((48 * 3600))
STOP_WAIT=30
ACTIVE_PID=""
WATCHER_PID=""
INTERRUPTED=0
STOPPED_AT_GATE=0
USE_FLOCK=0

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

write_status() {
  {
    printf '%s\n' "$1"
    if [[ -n "${2:-}" ]]; then printf '%s\n' "$2"; fi
  } >"$STATUS_FILE.tmp"
  mv -f "$STATUS_FILE.tmp" "$STATUS_FILE"
}

read_status() { sed -n 1p "$STATUS_FILE" 2>/dev/null || true; }

is_final_state() {
  local state
  state="$(read_status)"
  [[ "$state" == "DONE" || "$state" == "FAILED" || "$state" == "GATE_FAILED" ]]
}

print_status_banner() { sed 1d "$STATUS_FILE" 2>/dev/null || true; }

# A lock error must not read as "already held", which would attach forever, so flock is used only if it works here.
detect_flock() {
  USE_FLOCK=0
  command -v flock >/dev/null 2>&1 || return 0
  local code=0
  (flock -n -E 75 9) 9>"$STATE/.flock_probe" || code=$?
  if [[ "$code" -eq 0 ]]; then
    USE_FLOCK=1
  else
    say "flock does not work on $STATE (code $code), so using a process ID file instead"
  fi
}

lock_is_free() {
  if [[ "$USE_FLOCK" -eq 1 ]]; then
    (flock -n -E 75 9) 9>"$LOCK"
  else
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null
  fi
}

# E1 must have finished, and nothing of it may be alive. Each check alone can be fooled by a stale file or a lock
# that does not work on this disk, so all of them must pass. Returns 0 and prints the reason if E1 blocks this run.
e1_blocks_this_run() {
  local status pid code lock
  if [[ ! -f "$E1_STATE/status" ]]; then
    printf 'E2 needs a finished E1 run, and there is no E1 state at %s. Finish E1 with scripts/run_e1_gpu.sh, or set E2_E1_STATE_DIR to its state directory.\n' "$E1_STATE"
    return 0
  fi
  status="$(sed -n 1p "$E1_STATE/status" 2>/dev/null || true)"
  if [[ "$status" != "DONE" ]]; then
    printf 'E1 is still running or did not finish: its status is %s at %s. E2 runs only after E1 is DONE. Nothing was started.\n' "${status:-empty}" "$E1_STATE"
    return 0
  fi
  if command -v flock >/dev/null 2>&1; then
    for lock in "$E1_STATE/run.lock" "$E1_STATE/run/locks/orchestrator.lock"; do
      [[ -f "$lock" ]] || continue
      code=0
      (flock -n -E 75 9) 9>>"$lock" || code=$?
      if [[ "$code" -eq 75 ]]; then
        printf 'E1 is still running: another process holds %s. This includes an E1_ON_FINISH hook, so do not start E2 from one. Nothing was started.\n' "$lock"
        return 0
      fi
    done
  fi
  pid="$(cat "$E1_STATE/run.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && grep -q 'run_e1_gpu' "/proc/$pid/cmdline" 2>/dev/null; then
    printf 'E1 is still running as process %s. Nothing was started.\n' "$pid"
    return 0
  fi
  if command -v pgrep >/dev/null 2>&1 && pgrep -f 'mechinterp[.]cli (preflight|shard-run|shard-worker|run|finalize)' >/dev/null 2>&1; then
    printf 'An E1 process is still running: see pgrep -af mechinterp.cli. Nothing was started.\n'
    return 0
  fi
  return 1
}

# bounded LIMIT[:IDLE] command...
# Runs a command for at most LIMIT seconds. With IDLE, it also stops if the log gets no line for IDLE seconds.
# Returns the command's code, or 124 or 125 on those limits. It stops waiting even if the command cannot be
# killed, which happens when a process is stuck in the GPU driver. timeout(1) would wait for it forever.
bounded() {
  local limit="$1" idle=0 pid started mtime
  shift
  if [[ "$limit" == *:* ]]; then
    idle="${limit#*:}"
    limit="${limit%%:*}"
  fi
  # The explicit stdin keeps a heredoc, which a background job would otherwise lose. Closing 9 keeps a stuck
  # child from holding the run lock.
  "$@" <&0 9>&- &
  pid=$!
  ACTIVE_PID="$pid"
  started=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    if ((SECONDS - started >= limit)); then
      say "Stopping after the limit of $limit seconds: $1"
      stop_process "$pid"
      ACTIVE_PID=""
      return 124
    fi
    if ((idle > 0)); then
      mtime="$(stat -c %Y "$LOG" 2>/dev/null || echo 0)"
      if (($(date +%s) - mtime >= idle)); then
        say "Stopping because the log got no new line for $idle seconds: $1"
        stop_process "$pid"
        ACTIVE_PID=""
        return 125
      fi
    fi
    sleep 1
  done
  ACTIVE_PID=""
  local code=0
  wait "$pid" || code=$?
  return "$code"
}

stop_process() {
  local pid="$1" waited=0
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && ((waited < STOP_WAIT)); do
    sleep 1
    waited=$((waited + 1))
  done
  kill -KILL "$pid" 2>/dev/null || true
  waited=0
  while kill -0 "$pid" 2>/dev/null && ((waited < STOP_WAIT)); do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    say "Process $pid cannot be killed, so it is stuck in the kernel, probably in the GPU driver. Leaving it behind."
  fi
}

check_code() {
  local code="$1" limit="${2%%:*}" what="$3"
  case "$code" in
    0) return 0 ;;
    124) die "$what took longer than its limit of $((limit / 60)) minutes. The network, a disk or a GPU may be hung." ;;
    125) die "$what stopped printing anything, so it looks hung. A GPU or a disk may be stuck." ;;
    129 | 130 | 137 | 143)
      # Stopped from outside, as when an instance is preempted. That is an interruption, not a verdict.
      INTERRUPTED=1
      exit "$code"
      ;;
    *) die "$what failed with code $code. See the lines above." ;;
  esac
}

# Everything below runs detached

FAIL_REASON=""

die() {
  FAIL_REASON="$*"
  exit 1
}

banner_failed() {
  cat <<EOF

################################################################################
##  FAILED
##
##  Reason: ${FAIL_REASON:-see the log above}
##  Log:    $LOG
##  Worker logs: $RUN_DIR/logs/
##
##  No final analysis was written. Finished trials are safe on disk.
##  To try again and keep them:  E2_RETRY=1 bash scripts/run_e2_gpu.sh
##  A message that says E1_RETRY or E1_SHAM_TOLERANCE here means E2_RETRY or
##  E2_SHAM_TOLERANCE, since the check is shared with E1.
##
##  THE MACHINE KEEPS BILLING UNTIL YOU STOP IT.
################################################################################

EOF
}

banner_done() {
  cat <<EOF

################################################################################
##  DONE
##
##  Results: $RESULTS_DIR
##  Copy:    $RESULTS_COPY
##  Copy them off this machine now: nothing was pushed anywhere.
##
##  THE MACHINE KEEPS BILLING UNTIL YOU STOP IT. STOP IT WHEN YOU HAVE THE FILES.
################################################################################

EOF
}

banner_gate_failed() {
  cat <<EOF

################################################################################
##  GATE FAILED: E2 STOPPED HERE, AS THE PREREGISTRATION SAYS
##
##  The J-lens does not validate on this model, so no J-space claim is made.
##  Stages 2 and 3 did not run. This is a stop rule, not a reason to tune the
##  lens. Record it, close the J-space line and pick the next ledger item.
##
##  Results: $RESULTS_DIR
##  Copy:    $RESULTS_COPY
##  Copy them off this machine now: nothing was pushed anywhere.
##
##  THE MACHINE KEEPS BILLING UNTIL YOU STOP IT. STOP IT WHEN YOU HAVE THE FILES.
################################################################################

EOF
}

run_on_finish() {
  [[ -n "$ON_FINISH" ]] || return 0
  say "Running E2_ON_FINISH with E2_FINAL_STATUS=$1"
  E2_FINAL_STATUS="$1" bounded "$ON_FINISH_SECONDS" bash -c "$ON_FINISH" </dev/null \
    || say "E2_ON_FINISH failed or took too long. The final state is unchanged. THE MACHINE MAY STILL BE BILLING."
}

on_exit() {
  local code=$?
  trap - EXIT ERR TERM
  if [[ -n "$WATCHER_PID" ]]; then kill "$WATCHER_PID" 2>/dev/null || true; fi
  if [[ -n "$ACTIVE_PID" ]]; then kill -TERM "$ACTIVE_PID" 2>/dev/null || true; fi
  ACTIVE_PID=""
  if [[ "$code" -eq 0 && "$STOPPED_AT_GATE" -eq 1 ]]; then
    write_status GATE_FAILED "$(banner_gate_failed)"
    run_on_finish GATE_FAILED
  elif [[ "$code" -eq 0 ]]; then
    write_status DONE "$(banner_done)"
    run_on_finish DONE
  elif [[ "$INTERRUPTED" -eq 1 ]]; then
    write_status INTERRUPTED
    say "Interrupted from outside. Finished trials are safe. Run the same command to resume."
  else
    banner_failed
    write_status FAILED "$(banner_failed)"
    run_on_finish FAILED
  fi
  exit "$code"
}

on_error() {
  if [[ -z "$FAIL_REASON" ]]; then FAIL_REASON="Line $1 failed: $2"; fi
}

scan_gpus() {
  local line index name total free visible listing="$STATE/gpus.txt" code=0
  bounded "$NVIDIA_SMI_SECONDS" nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits \
    >"$listing" </dev/null || code=$?
  check_code "$code" "$NVIDIA_SMI_SECONDS" "nvidia-smi"
  visible=",${CUDA_VISIBLE_DEVICES:-},"
  GPU_LIST=""
  N_GPUS=0
  N_SEEN=0
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    N_SEEN=$((N_SEEN + 1))
    IFS=',' read -r index name total free <<<"$line"
    index="${index// /}"
    total="${total// /}"
    free="${free// /}"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    say "  GPU $index:${name} total ${total} MiB, free ${free} MiB"
    # Only plain indices can be compared with nvidia-smi's. Anything else, such as UUIDs, is not a filter here.
    if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9,]+$ && "$visible" != *",$index,"* ]]; then
      say "    skipped: not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      N_SEEN=$((N_SEEN - 1))
    elif ((free < MIN_GPU_FREE_MIB)); then
      say "    busy: float32 weights need about 16 GB plus room to work, so $MIN_GPU_FREE_TEXT must be free"
    else
      GPU_LIST="${GPU_LIST:+$GPU_LIST,}$index"
      N_GPUS=$((N_GPUS + 1))
    fi
  done <"$listing"
}

check_machine() {
  say "Step 1 of 7: checking this machine, and that E1 has finished"
  local blocked
  if blocked="$(e1_blocks_this_run)"; then
    die "$blocked"
  fi
  say "  E1 is DONE at $E1_STATE"
  [[ -f "$VENV/.e1_ready" && -x "$VENV/bin/python" ]] \
    || die "E1's Python environment is not at $VENV, and E2 reuses it rather than building another."
  [[ -f "$E1_STATE/.model_ready" ]] || die "E1's model cache is not ready under $E1_STATE/hf_home, and E2 reuses it."
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi was not found, so this does not look like an NVIDIA GPU machine."
  scan_gpus
  if ((N_GPUS < N_SEEN)); then
    # Workers from a run stopped a moment ago may still be letting go of their memory.
    say "  Some GPUs are busy, so looking again in 20 seconds"
    sleep 20
    scan_gpus
  fi
  ((N_SEEN > 0)) || die "nvidia-smi lists no GPUs that this run may use."
  ((N_GPUS >= 1)) || die "No GPU has $MIN_GPU_FREE_TEXT free. See 'nvidia-smi' for what is using them."
  if ((N_GPUS < 4)); then
    say "  WARNING: expected 4 GPUs, found $N_GPUS usable. Continuing on $N_GPUS, which takes proportionally longer."
  fi
  say "  Using $N_GPUS GPU(s): $GPU_LIST"
  FIRST_GPU="${GPU_LIST%%,*}"

  local free_disk_gb
  free_disk_gb="$(df -Pk "$STATE" | awk 'NR==2 {print int($4 / 1048576)}')"
  say "  Free disk at $STATE: ${free_disk_gb} GB (need ${MIN_DISK_GB} GB)"
  ((free_disk_gb >= MIN_DISK_GB)) || die "Only ${free_disk_gb} GB of disk is free at $STATE. This needs ${MIN_DISK_GB} GB."

  # A container's memory limit can be far below what the host reports as available.
  local ram_kb limit_bytes limit_kb ram_gb min_ram_gb
  ram_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  limit_bytes="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)"
  if [[ "$limit_bytes" =~ ^[0-9]+$ ]]; then
    limit_kb=$((limit_bytes / 1024))
    ((limit_kb >= ram_kb)) || ram_kb="$limit_kb"
  fi
  ram_gb=$((ram_kb / 1048576))
  # As in E1, plus worker 0's lens: about 1.4 GB while loading.
  min_ram_gb=$((18 + 3 * N_GPUS + 6))
  say "  Available RAM: ${ram_gb} GB (need ${min_ram_gb} GB)"
  ((ram_gb >= min_ram_gb)) || die "Only ${ram_gb} GB of RAM is available. This needs about ${min_ram_gb} GB."
  LOAD_SLOTS=$(((ram_gb - 8 - 3 * N_GPUS) / 20))
  ((LOAD_SLOTS >= 1)) || LOAD_SLOTS=1
  ((LOAD_SLOTS <= N_GPUS)) || LOAD_SLOTS="$N_GPUS"

  local cpus
  cpus="$(nproc 2>/dev/null || echo 4)"
  THREADS_PER_WORKER=$((cpus / N_GPUS))
  ((THREADS_PER_WORKER >= 1)) || THREADS_PER_WORKER=1
  ((THREADS_PER_WORKER <= 8)) || THREADS_PER_WORKER=8
  say "  $cpus CPU(s), so $THREADS_PER_WORKER thread(s) per worker, loading at most $LOAD_SLOTS at a time"
}

probe_environment() {
  say "Step 2 of 7: checking E1's Python environment in $VENV"
  PY="$VENV/bin/python"
  local code=0
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" bounded "$PROBE_SECONDS" "$PY" - >"$PROBE_OUTPUT" <<'PROBE' || code=$?
import sys

import huggingface_hub
import numpy
import tokenizers
import torch
import transformers

problems = []
if not transformers.__version__.startswith("5.22."):
    problems.append(f"transformers is {transformers.__version__}, not 5.22.*")
if not torch.cuda.is_available():
    problems.append(f"torch {torch.__version__} cannot use CUDA")
else:
    matrix = torch.ones(256, 256, device="cuda", dtype=torch.float32)
    if float((matrix @ matrix).sum()) != 256.0**3:
        problems.append("a small matrix product on the GPU gave a wrong answer")
if problems:
    print("Environment problem: " + "; ".join(problems))
    sys.exit(1)
print(
    f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
    f"transformers {transformers.__version__}, tokenizers {tokenizers.__version__}, "
    f"huggingface_hub {huggingface_hub.__version__}, numpy {numpy.__version__}, "
    f"GPU {torch.cuda.get_device_name(0)}"
)
PROBE
  cat "$PROBE_OUTPUT"
  check_code "$code" "$PROBE_SECONDS" "The environment check on GPU $FIRST_GPU"
  {
    printf 'date: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'versions: %s\n' "$(cat "$PROBE_OUTPUT")"
    printf 'git commit: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'E1 state: %s\n' "$E1_STATE"
    printf 'gpus used: %s\n' "$GPU_LIST"
    cat "$STATE/gpus.txt"
  } >"$ENV_REPORT"
}

download_lens() {
  say "Step 3 of 7: the lens, at its pinned commit, once"
  if [[ -f "$STATE/.lens_ready" ]]; then
    say "  Already in $HF_HOME, and it is hashed again on load"
    return 0
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    say "  Download attempt $attempt of 5, limit $((DOWNLOAD_ATTEMPT_SECONDS / 60)) minutes"
    if HF_HUB_OFFLINE=0 bounded "$DOWNLOAD_ATTEMPT_SECONDS" "$PY" - "$CONFIG" <<'DOWNLOAD'
import hashlib
import json
import os
import sys

from huggingface_hub import hf_hub_download

with open(sys.argv[1], encoding="utf-8") as handle:
    lens = json.load(handle)["lens"]
path = hf_hub_download(lens["repo"], lens["filename"], revision=lens["commit"])
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if os.path.getsize(path) != lens["size"] or digest.hexdigest() != lens["sha256"]:
    print(f"{path} is {os.path.getsize(path)} bytes with sha256 {digest.hexdigest()}, which is not the pin")
    sys.exit(1)
print(f"Lens at {path}, {lens['size']} bytes, sha256 {lens['sha256']}")
DOWNLOAD
    then
      touch "$STATE/.lens_ready"
      return 0
    fi
    sleep $((attempt * 20))
  done
  die "The lens download failed 5 times. Check the network and free disk."
}

run_stage1() {
  say "Step 4 of 7: lens checks and Stage 1's gate on GPU $FIRST_GPU"
  local code=0
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" OMP_NUM_THREADS="$THREADS_PER_WORKER" CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    bounded "$STAGE1_SECONDS:$STAGE1_IDLE_SECONDS" "$PY" -u -m mechinterp.e2.cli stage1 --config "$CONFIG" --out "$RUN_DIR" </dev/null || code=$?
  if [[ "$code" -eq "$EXIT_GATE_FAILED" ]]; then
    say "THE GATE FAILED. E2 stops here, as the prereg says. Stages 2 and 3 will not run."
    write_results
    STOPPED_AT_GATE=1
    say "Step 7 of 7: finished at the gate"
    banner_gate_failed
    trap - ERR
    exit 0
  fi
  if [[ "$code" -eq 1 && -f "$FAILURE_REASON_FILE" ]]; then
    die "$(cat "$FAILURE_REASON_FILE")"
  fi
  check_code "$code" "$STAGE1_SECONDS" "Stage 1"
  say "THE GATE PASSED. Going on to Stages 2 and 3."
}

run_preflight() {
  say "Step 5 of 7: dry-run checks on GPU $FIRST_GPU, with a runtime estimate from measured speed"
  local code=0
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" OMP_NUM_THREADS="$THREADS_PER_WORKER" CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    bounded "$PREFLIGHT_SECONDS" "$PY" -u -m mechinterp.e2.cli preflight --config "$CONFIG" --out "$RUN_DIR" \
    --gate "$GATE_FILE" --e1-dir "$E1_STATE" --n-workers "$N_GPUS" --sham-tolerance "$SHAM_TOLERANCE" </dev/null || code=$?
  if [[ "$code" -eq 2 ]]; then
    die "The dry-run checks failed, so the run stopped before spending hours. See PASS and FAIL above and $DRY_RUN_JSON"
  fi
  check_code "$code" "$PREFLIGHT_SECONDS" "The dry run"
}

run_stages() {
  say "Step 6 of 7: Stage 2 and Stage 3 on $N_GPUS GPU(s). Progress and speed per GPU are printed about once a minute."
  say "  A worker with no finished trial for $STALL_MINUTES minutes is killed. Set E2_STALL_MINUTES to change that."
  local code=0 idle
  # The orchestrator prints every minute, and handles wedged workers itself. This covers the orchestrator.
  idle=$((${STALL_MINUTES%%.*} * 120 + 300))
  bounded "$RUN_MAX_SECONDS:$idle" "$PY" -u -m mechinterp.e2.cli shard-run --config "$CONFIG" --out "$RUN_DIR" --gate "$GATE_FILE" \
    --e1-dir "$E1_STATE" --devices "$GPU_LIST" --load-slots "$LOAD_SLOTS" --threads-per-worker "$THREADS_PER_WORKER" \
    --sham-tolerance "$SHAM_TOLERANCE" --stall-minutes "$STALL_MINUTES" </dev/null || code=$?
  if [[ "$code" -eq 1 && -f "$FAILURE_REASON_FILE" ]]; then
    die "$(cat "$FAILURE_REASON_FILE")"
  fi
  check_code "$code" "$RUN_MAX_SECONDS" "The run"
}

write_results() {
  say "Writing results"
  local code=0
  rm -f "$FAILURE_REASON_FILE"
  mkdir -p "$RESULTS_DIR"
  bounded "$FINALIZE_SECONDS" "$PY" -u -m mechinterp.e2.cli finalize --config "$CONFIG" --out "$RUN_DIR" --gate "$GATE_FILE" \
    --results-dir "$RESULTS_DIR" --log "$LOG" </dev/null || code=$?
  if [[ "$code" -eq 1 && -f "$FAILURE_REASON_FILE" ]]; then
    die "$(cat "$FAILURE_REASON_FILE")"
  fi
  check_code "$code" "$FINALIZE_SECONDS" "Writing the results"
  # The copied log ends here, and its hash is of that copy. Later lines reach only the live log.
  if mkdir -p "$RESULTS_COPY" && cp -f "$RESULTS_DIR"/* "$RESULTS_COPY"/; then
    say "  Also copied to $RESULTS_COPY"
  else
    say "  WARNING: could not copy to $RESULTS_COPY. The results are safe in $RESULTS_DIR"
  fi
}

inner_main() {
  detect_flock
  if [[ "$USE_FLOCK" -eq 1 ]]; then
    exec 9>"$LOCK"
    flock -n -E 75 9 || exit 0 # Another copy won the race. It owns the status file, so leave it alone.
  fi
  printf '%s\n' "$$" >"$PID_FILE"
  trap on_exit EXIT
  trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
  trap 'INTERRUPTED=1; exit 143' TERM
  trap '' HUP
  write_status RUNNING
  say "======== E2, started with state in $STATE ========"
  say "Config: $CONFIG"

  mkdir -p "$RUN_DIR" "$STATE/tmp"
  export HF_HOME="$E1_STATE/hf_home"
  export TMPDIR="$STATE/tmp"
  export PYTHONNOUSERSITE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export TOKENIZERS_PARALLELISM=false
  # The lens and the model are public, and a stale token on a rented image would turn a download into a 401.
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_ENABLE_HF_TRANSFER
  export PYTHONPATH="$REPO_ROOT/src"

  check_machine
  probe_environment
  download_lens
  # From here the cache is read-only, so workers cannot race on its locks or need the network.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  run_stage1
  run_preflight
  run_stages
  write_results
  say "Step 7 of 7: finished"
  banner_done
  trap - ERR
}

# The foreground part: start or find the run, then show its log

attach() {
  local status tail_pid waited=0
  say "Log: $LOG"
  say "You can close this terminal: the run continues. Run the same command to see it again."
  tail -n 40 -F "$LOG" 2>/dev/null &
  tail_pid=$!
  trap 'kill "$tail_pid" 2>/dev/null || true; printf "\nStopped watching. The run continues in the background.\n"; exit 0' INT TERM
  while true; do
    sleep 2
    waited=$((waited + 2))
    status="$(read_status)"
    if [[ "$status" == "DONE" || "$status" == "FAILED" || "$status" == "GATE_FAILED" || "$status" == "INTERRUPTED" ]]; then
      break
    fi
    if ((waited >= 30)) && lock_is_free; then
      status="VANISHED"
      break
    fi
  done
  sleep 2
  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true
  trap - INT TERM
  case "$status" in
    DONE | GATE_FAILED)
      print_final_state
      return 0
      ;;
    FAILED)
      return 1
      ;;
    *)
      say "The background run stopped before it finished, perhaps from the system's out-of-memory killer or a preemption."
      say "No final result was written. Finished trials are safe. Run the same command to resume."
      say "THE MACHINE KEEPS BILLING UNTIL YOU STOP IT."
      return 1
      ;;
  esac
}

print_final_state() {
  local status
  status="$(read_status)"
  if [[ ("$status" == "DONE" || "$status" == "GATE_FAILED") && -f "$SUMMARY" ]]; then
    cat "$SUMMARY"
  fi
  print_status_banner
}

main() {
  case "${1:-}" in
    --inner)
      shift
      mkdir -p "$STATE"
      inner_main "$@"
      return
      ;;
    "") ;;
    *)
      printf 'Usage: bash scripts/run_e2_gpu.sh\nThis command takes no arguments.\n' >&2
      return 2
      ;;
  esac

  [[ -f "$CONFIG" ]] || {
    printf 'Config not found: %s\n' "$CONFIG" >&2
    return 1
  }
  # Before anything is created, so that a refused run leaves no trace of itself inside E1's state directory.
  local blocked
  if [[ ! -f "$STATUS_FILE" ]] || ! is_final_state; then
    if blocked="$(e1_blocks_this_run)"; then
      say "REFUSED: $blocked"
      return 3
    fi
  fi
  mkdir -p "$STATE" || {
    printf 'Cannot create %s\n' "$STATE" >&2
    return 1
  }
  detect_flock
  if is_final_state; then
    if [[ "$(read_status)" != "FAILED" ]]; then
      say "This run already finished as $(read_status). Nothing was started."
      print_final_state
      return 0
    fi
    if [[ "${E2_RETRY:-0}" != "1" ]]; then
      say "This run already ended as FAILED. Nothing was started."
      print_final_state
      return 1
    fi
    if blocked="$(e1_blocks_this_run)"; then
      say "REFUSED: $blocked"
      return 3
    fi
    say "E2_RETRY=1, so resuming the failed run. Finished trials are kept."
  fi
  touch "$LOG"
  if lock_is_free; then
    say "Starting the run in the background"
    write_status STARTING
    if command -v setsid >/dev/null 2>&1; then
      setsid nohup bash "$SCRIPT_PATH" --inner </dev/null >>"$LOG" 2>&1 &
    else
      nohup bash "$SCRIPT_PATH" --inner </dev/null >>"$LOG" 2>&1 &
    fi
    disown || true
  else
    say "A run is already going, so attaching to it"
  fi
  attach
}

# Tests source this file to reach its functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
