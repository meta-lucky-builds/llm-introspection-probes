#!/usr/bin/env bash
# One command for experiment D1 on a rented machine with one or more GPUs:
#
#     bash scripts/run_d1_gpu.sh
#
# It checks the machine, then the pins, then installs what is missing, checks the data, and only then downloads models,
# as the ladder needs them. It runs the dry-run checks for each model, runs every stage across all GPUs and writes the
# results folder. Everything that can fail cheaply comes first. It keeps running if the terminal closes. Running it again
# attaches to a live run, resumes an interrupted one, or prints the final state of one that ended, without restarting
# anything. One 24 GB GPU is enough up to Qwen3-8B. Qwen3-14B needs 36 GB. The first step prints how far this machine's disk
# and RAM reach.
#
# Every outcome the prereg names ends as DONE, with the outcome in the report: G1 failed through the ladder, G2 failed
# after its retry, or a verdict, inconclusive and probe unusable included. FAILED means the run itself broke, and that says
# nothing about D1.
#
# Optional environment:
#   D1_ON_FINISH        Shell command to run after the final state is saved, for example to stop the machine or
#                       send a message. It sees D1_FINAL_STATUS, either DONE or FAILED. Default: nothing
#   D1_RETRY            Set to 1 to resume a run that ended as FAILED. Finished work is kept
#   D1_REANALYZE        Set to 1 on a run that ended as DONE, after a fix to analysis code or the analysis config:
#                       recomputes gates and the report from stored numbers, with no GPU work, and rewrites the results
#   D1_STALL_MINUTES    Kill a loaded worker after this long with no progress. Default 30
#   D1_CONFIG           Config file. Default: experiments/d1_double_down/configs/qwen3_1_7b.json. full.json has the whole ladder
#   D1_STATE_DIR        Checkpoints, venv, model cache and results. Default: /workspace/d1_double_down if
#                       /workspace is writable, otherwise .d1_state/ inside the clone
#   D1_RESULTS_DIR      A second copy of the results. Default: results/d1 inside the clone
#   D1_PYTHON           The Python to build the venv from
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)"
CONFIG="${D1_CONFIG:-$REPO_ROOT/experiments/d1_double_down/configs/qwen3_1_7b.json}"
RESULTS_COPY="${D1_RESULTS_DIR:-$REPO_ROOT/results/d1}"
STALL_MINUTES="${D1_STALL_MINUTES:-30}"
ON_FINISH="${D1_ON_FINISH:-}"
if [[ -n "${D1_STATE_DIR:-}" ]]; then
  STATE="$D1_STATE_DIR"
elif [[ -d /workspace && -w /workspace ]]; then
  STATE=/workspace/d1_double_down
else
  STATE="$REPO_ROOT/.d1_state/double_down"
fi
RUN_DIR="$STATE/run"
VENV="$STATE/venv"
LOG="$STATE/run.log"
LOCK="$STATE/run.lock"
PID_FILE="$STATE/run.pid"
STATUS_FILE="$STATE/status" # First line: the state. The rest: the banner to show again.
ENV_REPORT="$STATE/environment.txt"
PROBE_OUTPUT="$STATE/probe.txt"
# On the persistent disk, so that results survive a machine stopped by D1_ON_FINISH.
RESULTS_DIR="$STATE/results"
SUMMARY="$RESULTS_DIR/summary.txt"
FAILURE_REASON_FILE="$RUN_DIR/failure_reason.txt"
MIN_GPU_FREE_MIB=14336
MIN_GPU_FREE_TEXT="14 GB"
NVIDIA_SMI_SECONDS=60
PROBE_SECONDS=900
PIP_SECONDS=1800
APT_SECONDS=900
PYTHON_INSTALL_SECONDS=900
DOWNLOAD_ATTEMPT_SECONDS=7200
FINALIZE_SECONDS=1800
ON_FINISH_SECONDS=600
RUN_MAX_SECONDS=$((168 * 3600))
STOP_WAIT=30
ACTIVE_PID=""
WATCHER_PID=""
INTERRUPTED=0
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
  [[ "$state" == "DONE" || "$state" == "FAILED" ]]
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

# step LIMIT "what it does" command...
# Runs a bounded command and dies with a plain message on any failure.
step() {
  local limit="$1" what="$2" code=0
  shift 2
  bounded "$limit" "$@" || code=$?
  check_code "$code" "$limit" "$what"
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
##  The run broke. This is not a result about D1. Finished work is safe on disk.
##  To try again and keep it:  D1_RETRY=1 bash scripts/run_d1_gpu.sh
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

run_on_finish() {
  [[ -n "$ON_FINISH" ]] || return 0
  say "Running D1_ON_FINISH with D1_FINAL_STATUS=$1"
  D1_FINAL_STATUS="$1" bounded "$ON_FINISH_SECONDS" bash -c "$ON_FINISH" </dev/null \
    || say "D1_ON_FINISH failed or took too long. The final state is unchanged. THE MACHINE MAY STILL BE BILLING."
}

on_exit() {
  local code=$?
  trap - EXIT ERR TERM
  if [[ -n "$WATCHER_PID" ]]; then kill "$WATCHER_PID" 2>/dev/null || true; fi
  if [[ -n "$ACTIVE_PID" ]]; then kill -TERM "$ACTIVE_PID" 2>/dev/null || true; fi
  ACTIVE_PID=""
  if [[ "$code" -eq 0 ]]; then
    write_status DONE "$(banner_done)"
    run_on_finish DONE
  elif [[ "$INTERRUPTED" -eq 1 ]]; then
    write_status INTERRUPTED
    say "Interrupted from outside. Finished work is safe. Run the same command to resume."
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
  GPU_FREE_MIB=0
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
      say "    busy: Qwen3-1.7B needs about $MIN_GPU_FREE_TEXT free to train through it"
    else
      GPU_LIST="${GPU_LIST:+$GPU_LIST,}$index"
      N_GPUS=$((N_GPUS + 1))
      if ((GPU_FREE_MIB == 0 || free < GPU_FREE_MIB)); then GPU_FREE_MIB="$free"; fi
    fi
  done <"$listing"
}

check_machine() {
  say "Step 1 of 7: checking this machine"
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi was not found, so this does not look like an NVIDIA GPU machine."
  step "$NVIDIA_SMI_SECONDS" "nvidia-smi" nvidia-smi </dev/null

  scan_gpus
  if ((N_GPUS < N_SEEN)); then
    # Workers from a run stopped a moment ago may still be letting go of their memory.
    say "  Some GPUs are busy, so looking again in 20 seconds"
    sleep 20
    scan_gpus
  fi
  ((N_SEEN > 0)) || die "nvidia-smi lists no GPUs that this run may use."
  ((N_GPUS >= 1)) || die "No GPU has $MIN_GPU_FREE_TEXT free. See 'nvidia-smi' for what is using them."
  say "  Using $N_GPUS GPU(s): $GPU_LIST. The smallest has $((GPU_FREE_MIB / 1024)) GB free, which decides how far the ladder can go."
  FIRST_GPU="${GPU_LIST%%,*}"

  FREE_DISK_GB="$(df -Pk "$STATE" | awk 'NR==2 {print int($4 / 1000000)}')"
  say "  Free disk at $STATE: ${FREE_DISK_GB} GB"

  # A container's memory limit can be far below what the host reports as available.
  local ram_kb limit_bytes limit_kb
  ram_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  limit_bytes="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)"
  if [[ "$limit_bytes" =~ ^[0-9]+$ ]]; then
    limit_kb=$((limit_bytes / 1024))
    ((limit_kb >= ram_kb)) || ram_kb="$limit_kb"
  fi
  RAM_GB=$((ram_kb / 1048576))
  say "  Available RAM: ${RAM_GB} GB"
  # Weights go straight from the file to the GPU, so a load costs a few GB of RAM and not a whole model. Step 2 judges all this.
  LOAD_SLOTS=$(((RAM_GB - 4) / 6))
  ((LOAD_SLOTS >= 1)) || LOAD_SLOTS=1
  ((LOAD_SLOTS <= N_GPUS)) || LOAD_SLOTS="$N_GPUS"
  say "  Workers will load the model at most $LOAD_SLOTS at a time"

  local cpus
  cpus="$(nproc 2>/dev/null || echo 4)"
  THREADS_PER_WORKER=$((cpus / N_GPUS))
  ((THREADS_PER_WORKER >= 1)) || THREADS_PER_WORKER=1
  ((THREADS_PER_WORKER <= 8)) || THREADS_PER_WORKER=8
  say "  $cpus CPU(s), so $THREADS_PER_WORKER thread(s) per worker"
}

# Standard library only and seconds long, so these come before anything is installed. A resume has its software and data in
# place, so a remote it cannot reach is only a warning there, and disk is left to the orchestrator, which knows what is stored.
early_checks() {
  say "Step 2 of 7: how far this machine reaches, and whether the pins still hold"
  local python="${VENV}/bin/python" code=0 resuming=()
  if [[ -f "$VENV/.d1_ready" && -x "$python" ]]; then
    :
  else
    find_python
    python="$BASE_PYTHON"
  fi
  if [[ -f "$VENV/.d1_ready" && -f "$STATE/.data_ready" ]]; then
    resuming=(--resuming)
  fi
  "$python" -m mechinterp.d1.cli capacity --config "$CONFIG" --free-disk-gb "$FREE_DISK_GB" --ram-gb "$RAM_GB" \
    --gpu-free-gb "$((GPU_FREE_MIB / 1024))" --n-gpus "$N_GPUS" "${resuming[@]}" </dev/null \
    || die "This machine cannot run the first rung. See the lines above. Free some disk, set D1_STATE_DIR to a bigger disk, or use a larger machine."
  bounded 300 "$python" -m mechinterp.d1.cli check-pins --config "$CONFIG" </dev/null || code=$?
  case "$code" in
    0) ;;
    3)
      if ((${#resuming[@]} > 0)); then
        say "  Carrying on, since everything needed is already here"
      else
        die "Hugging Face cannot be reached from this machine, so nothing can be downloaded. See the line above."
      fi
      ;;
    1) die "A pinned model or dataset is not what the config says. See the line above. This is not a machine problem." ;;
    *) check_code "$code" 300 "Checking the pins" ;;
  esac
}

python_is_usable() {
  # transformers 5.22 needs 3.11 or newer, and torch wheels lag the newest Python.
  "$1" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null
}

find_python() {
  local candidate
  [[ -z "${BASE_PYTHON:-}" ]] || return 0
  for candidate in "${D1_PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if python_is_usable "$candidate"; then
      BASE_PYTHON="$candidate"
      say "  Base Python: $(command -v "$candidate") ($("$candidate" --version 2>&1))"
      return 0
    fi
  done
  install_python
}

fetch_url() {
  local url="$1" target="$2"
  if command -v curl >/dev/null 2>&1; then
    bounded 300 curl -fsSL --connect-timeout 30 --max-time 280 -o "$target" "$url" </dev/null
  elif command -v wget >/dev/null 2>&1; then
    bounded 300 wget -q -T 60 -O "$target" "$url" </dev/null
  else
    local python
    for python in python3 python; do
      if command -v "$python" >/dev/null 2>&1; then
        bounded 300 "$python" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$url" "$target" </dev/null
        return
      fi
    done
    return 1
  fi
}

install_python() {
  say "  No Python 3.11 to 3.13 is on this machine, so installing one. This needs no sudo."
  local uv_dir="$STATE/uv" found url version
  mkdir -p "$uv_dir"
  export UV_PYTHON_INSTALL_DIR="$STATE/uv_python"
  export UV_PYTHON_PREFERENCE=only-managed
  url="https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz"
  say "  Downloading uv from $url"
  if fetch_url "$url" "$uv_dir/uv.tar.gz" && tar -xzf "$uv_dir/uv.tar.gz" -C "$uv_dir" --strip-components=1; then
    if bounded "$PYTHON_INSTALL_SECONDS" "$uv_dir/uv" python install 3.12 </dev/null; then
      found="$("$uv_dir/uv" python find 3.12 2>/dev/null || true)"
      if [[ -n "$found" ]] && python_is_usable "$found"; then
        BASE_PYTHON="$found"
        say "  Base Python: $found, installed with uv"
        return 0
      fi
    fi
  fi
  say "  uv did not work"
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    say "  Trying apt-get, since this is root"
    export DEBIAN_FRONTEND=noninteractive
    bounded "$APT_SECONDS" apt-get update -qq </dev/null || true
    for version in 3.13 3.12 3.11; do
      if bounded "$APT_SECONDS" apt-get install -y -qq "python$version" "python$version-venv" </dev/null \
        && python_is_usable "python$version"; then
        BASE_PYTHON="python$version"
        say "  Base Python: python$version, installed with apt-get"
        return 0
      fi
    done
  fi
  die "Python 3.11 to 3.13 is required and could not be found or installed. uv $([[ "$(id -u)" -eq 0 ]] && echo 'and apt-get' || echo 'alone, not being root,') did not work. Install Python 3.12, then run: D1_RETRY=1 bash scripts/run_d1_gpu.sh"
}

make_venv() {
  # System site packages stay visible so that a preinstalled CUDA torch can be reused.
  if bounded 300 "$BASE_PYTHON" -m venv --system-site-packages "$VENV" </dev/null; then
    return 0
  fi
  rm -rf "$VENV"
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    say "  venv support is missing, so installing it with apt-get"
    local version
    version="$("$BASE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    export DEBIAN_FRONTEND=noninteractive
    bounded "$APT_SECONDS" apt-get update -qq </dev/null || true
    bounded "$APT_SECONDS" apt-get install -y -qq "python${version}-venv" </dev/null \
      || bounded "$APT_SECONDS" apt-get install -y -qq python3-venv </dev/null || true
    if bounded 300 "$BASE_PYTHON" -m venv --system-site-packages "$VENV" </dev/null; then
      return 0
    fi
    rm -rf "$VENV"
  fi
  say "  Trying virtualenv instead"
  bounded "$PIP_SECONDS" "$BASE_PYTHON" -m pip install --quiet --target "$STATE/virtualenv_pkg" virtualenv </dev/null \
    && PYTHONPATH="$STATE/virtualenv_pkg" bounded 300 "$BASE_PYTHON" -m virtualenv --system-site-packages "$VENV" </dev/null \
    || die "Could not create a virtual environment with $BASE_PYTHON. Install python3-venv, then run: D1_RETRY=1 bash scripts/run_d1_gpu.sh"
}

probe_environment() {
  local code=0
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" bounded "$PROBE_SECONDS" "$PY" - >"$PROBE_OUTPUT" <<'PROBE' || code=$?
import sys

import accelerate  # noqa: F401  (a device_map, which loads weights straight to the GPU, needs it)
import huggingface_hub
import jinja2  # noqa: F401
import numpy
import pyarrow
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM  # noqa: F401

major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
problems = []
if (major, minor) < (2, 6):
    problems.append(f"torch {torch.__version__} is older than 2.6")
if not transformers.__version__.startswith("5.22."):
    problems.append(f"transformers is {transformers.__version__}, not 5.22.*")
if not torch.cuda.is_available():
    problems.append(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) cannot use CUDA")
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
    f"huggingface_hub {huggingface_hub.__version__}, numpy {numpy.__version__}, pyarrow {pyarrow.__version__}, "
    f"GPU {torch.cuda.get_device_name(0)}"
)
PROBE
  cat "$PROBE_OUTPUT"
  if [[ "$code" -eq 124 ]]; then
    die "A small test on GPU $FIRST_GPU did not finish in $((PROBE_SECONDS / 60)) minutes, so the GPU or driver looks hung."
  fi
  return "$code"
}

install_torch() {
  local driver_cuda driver_major driver_minor index number smi_output="$STATE/nvidia_smi.txt"
  bounded "$NVIDIA_SMI_SECONDS" nvidia-smi >"$smi_output" </dev/null || die "nvidia-smi stopped answering."
  driver_cuda="$(sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' "$smi_output" | sed -n 1p)"
  [[ -n "$driver_cuda" ]] || die "Could not read the CUDA version from nvidia-smi."
  driver_major="${driver_cuda%%.*}"
  driver_minor="${driver_cuda##*.}"
  say "  The driver supports CUDA up to $driver_cuda"
  for index in cu134 cu133 cu132 cu131 cu130 cu129 cu128 cu127 cu126 cu125 cu124 cu122 cu121 cu118; do
    number="${index#cu}"
    if ((10#${number:0:2} * 100 + 10#${number:2} > 10#$driver_major * 100 + 10#$driver_minor)); then
      continue
    fi
    say "  Trying a torch build from https://download.pytorch.org/whl/$index"
    # --ignore-installed puts it in the venv even when a system torch would satisfy the version.
    if bounded "$PIP_SECONDS" "$PY" -m pip install --quiet --ignore-installed --index-url "https://download.pytorch.org/whl/$index" "torch>=2.6" </dev/null; then
      if probe_environment; then
        TORCH_SOURCE="installed from whl/$index, for driver CUDA $driver_cuda"
        return 0
      fi
    fi
  done
  die "No torch build worked with this machine's driver (CUDA $driver_cuda). See the pip errors above."
}

install_dependencies() {
  say "Step 3 of 7: Python environment in $VENV"
  PY="$VENV/bin/python"
  TORCH_SOURCE="kept from the existing venv"
  if [[ -f "$VENV/.d1_ready" && -x "$PY" ]] && probe_environment; then
    say "  Reusing the environment from before"
  else
    find_python
    rm -rf "$VENV"
    make_venv
    bounded "$PIP_SECONDS" "$PY" -m pip install --quiet --upgrade pip </dev/null || say "  Could not upgrade pip. Carrying on."
    step "$PIP_SECONDS" "Installing transformers with pip" \
      "$PY" -m pip install --quiet "transformers==5.22.*" "accelerate>=1.1.0" numpy "jinja2>=3.1" huggingface_hub safetensors pyarrow </dev/null
    if probe_environment; then
      TORCH_SOURCE="reused the torch already on this machine"
      say "  Reusing the torch already on this machine"
    else
      say "  The torch on this machine cannot be used, so installing one"
      install_torch
    fi
    touch "$VENV/.d1_ready"
  fi
  {
    printf 'date: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'torch: %s\n' "$TORCH_SOURCE"
    printf 'versions: %s\n' "$(cat "$PROBE_OUTPUT")"
    printf 'git commit: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'gpus used: %s\n' "$GPU_LIST"
    cat "$STATE/gpus.txt"
  } >"$ENV_REPORT"
  say "  Torch: $TORCH_SOURCE"
  say "  Recorded in $ENV_REPORT"
}

download_data() {
  say "Step 4 of 7: downloading FIBS, once, before any model. The first run step checks it against the published counts."
  if [[ -f "$STATE/.data_ready" ]]; then
    say "  Already in $HF_HOME"
    return 0
  fi
  local attempt
  (
    exec 9>&- # So that this helper can never hold the run lock.
    while sleep 60; do
      say "  Cache so far: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
    done
  ) &
  WATCHER_PID=$!
  for attempt in 1 2 3 4 5; do
    say "  Download attempt $attempt of 5, limit $((DOWNLOAD_ATTEMPT_SECONDS / 60)) minutes"
    if HF_HUB_OFFLINE=0 bounded "$DOWNLOAD_ATTEMPT_SECONDS" "$PY" - "$CONFIG" <<'DOWNLOAD'
import json
import sys

from huggingface_hub import snapshot_download

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
dataset = config["dataset"]
# The data is cached on this machine only. It has no license tag, so it is never copied into the repo or the results.
snapshot_download(dataset["repo"], repo_type="dataset", revision=dataset["revision"], allow_patterns=["data/*.parquet"])
print(f"FIBS at {dataset['revision']} is cached")
DOWNLOAD
    then
      kill "$WATCHER_PID" 2>/dev/null || true
      wait "$WATCHER_PID" 2>/dev/null || true
      WATCHER_PID=""
      touch "$STATE/.data_ready"
      return 0
    fi
    sleep $((attempt * 20))
  done
  die "The download failed 5 times. Check the network and free disk."
}

run_stages() {
  say "Step 5 of 7: running every stage on $N_GPUS GPU(s). The data check comes first. Each model starts with its dry-run checks, printed as PASS or FAIL."
  say "  Progress and a stage ETA are printed about once a minute. A worker with no progress for $STALL_MINUTES minutes is killed."
  local code=0 idle
  # The orchestrator prints every minute, and handles wedged workers itself. This covers the orchestrator.
  idle=$((${STALL_MINUTES%%.*} * 120 + 300))
  bounded "$RUN_MAX_SECONDS:$idle" "$PY" -u -m mechinterp.d1.cli run --config "$CONFIG" --out "$RUN_DIR" --devices "$GPU_LIST" \
    --gpu-free-gb "$((GPU_FREE_MIB / 1024))" --load-slots "$LOAD_SLOTS" --threads-per-worker "$THREADS_PER_WORKER" \
    --stall-minutes "$STALL_MINUTES" </dev/null || code=$?
  if [[ "$code" -eq 1 && -f "$FAILURE_REASON_FILE" ]]; then
    die "$(cat "$FAILURE_REASON_FILE")"
  fi
  check_code "$code" "$RUN_MAX_SECONDS" "The run"
}

write_results() {
  say "Step 6 of 7: writing results"
  local code=0
  rm -f "$FAILURE_REASON_FILE"
  mkdir -p "$RESULTS_DIR"
  bounded "$FINALIZE_SECONDS" "$PY" -u -m mechinterp.d1.cli finalize --config "$CONFIG" --out "$RUN_DIR" --results-dir "$RESULTS_DIR" --log "$LOG" </dev/null || code=$?
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
  say "======== D1 Double-Down, started with state in $STATE ========"
  say "Config: $CONFIG"

  mkdir -p "$RUN_DIR" "$STATE/tmp" "$STATE/pip_cache"
  export HF_HOME="$STATE/hf_home"
  export TMPDIR="$STATE/tmp" # A small /tmp is a common way for a torch install to fail.
  export PIP_CACHE_DIR="$STATE/pip_cache"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
  export PIP_ROOT_USER_ACTION=ignore
  export PIP_DEFAULT_TIMEOUT=60
  export PIP_RETRIES=3
  export PYTHONNOUSERSITE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export TOKENIZERS_PARALLELISM=false
  # The models are ungated, the data is public, and a stale token on a rented image would turn a public download into a 401.
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_ENABLE_HF_TRANSFER
  export PYTHONPATH="$REPO_ROOT/src"

  check_machine
  early_checks
  install_dependencies
  download_data
  # From here workers read the cache and need no network. The orchestrator fetches each rung itself, when the ladder gets there.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
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
    if [[ "$status" == "DONE" || "$status" == "FAILED" || "$status" == "INTERRUPTED" ]]; then
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
    DONE)
      print_final_state
      return 0
      ;;
    FAILED)
      return 1
      ;;
    *)
      say "The background run stopped before it finished, perhaps from the system's out-of-memory killer or a preemption."
      say "No final result was written. Finished work is safe. Run the same command to resume."
      say "THE MACHINE KEEPS BILLING UNTIL YOU STOP IT."
      return 1
      ;;
  esac
}

print_final_state() {
  if [[ "$(read_status)" == "DONE" && -f "$SUMMARY" ]]; then
    cat "$SUMMARY"
  fi
  print_status_banner
}

# Runs in the foreground and starts no worker, so it takes no GPU and cannot change anything stored but gates and reports.
reanalyze() {
  local py="$VENV/bin/python"
  [[ -x "$py" ]] || {
    say "No environment at $VENV, so nothing can be re-analyzed here."
    return 1
  }
  export PYTHONPATH="$REPO_ROOT/src"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  say "D1_REANALYZE=1, so recomputing gates and the report from stored numbers. No GPU work is started."
  "$py" -u -m mechinterp.d1.cli reanalyze --config "$CONFIG" --out "$RUN_DIR" </dev/null || return 1
  "$py" -u -m mechinterp.d1.cli finalize --config "$CONFIG" --out "$RUN_DIR" --results-dir "$RESULTS_DIR" --log "$LOG" </dev/null || return 1
  if mkdir -p "$RESULTS_COPY" && cp -f "$RESULTS_DIR"/* "$RESULTS_COPY"/; then
    say "Also copied to $RESULTS_COPY"
  fi
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
      printf 'Usage: bash scripts/run_d1_gpu.sh\nThis command takes no arguments.\n' >&2
      return 2
      ;;
  esac

  mkdir -p "$STATE" || {
    printf 'Cannot create %s\n' "$STATE" >&2
    return 1
  }
  [[ -f "$CONFIG" ]] || {
    printf 'Config not found: %s\n' "$CONFIG" >&2
    return 1
  }
  detect_flock
  if is_final_state; then
    if [[ "$(read_status)" == "DONE" ]]; then
      if [[ "${D1_REANALYZE:-0}" == "1" ]]; then
        reanalyze
        return
      fi
      say "This run already finished. Nothing was started. After a fix to analysis code: D1_REANALYZE=1 bash scripts/run_d1_gpu.sh"
      print_final_state
      return 0
    fi
    if [[ "${D1_RETRY:-0}" != "1" ]]; then
      say "This run already ended as FAILED. Nothing was started."
      print_final_state
      return 1
    fi
    say "D1_RETRY=1, so resuming the failed run. Finished work is kept."
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
