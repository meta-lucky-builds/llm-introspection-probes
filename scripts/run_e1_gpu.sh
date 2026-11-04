#!/usr/bin/env bash
# One command for E1's large tier on a rented multi-GPU machine:
#
#     bash scripts/run_e1_gpu.sh
#
# It checks the machine, installs what is missing, downloads the model once, runs the dry-run checks,
# runs every stage across all GPUs and writes the results folder. It keeps running if the terminal
# closes. Running it again attaches to a live run, resumes a dead one, or reprints a finished one.
#
# Optional environment:
#   E1_CONFIG           Config file. Default: configs/large_4x3090.json
#   E1_STATE_DIR        Checkpoints, venv and model cache. Default: /workspace/e1_large_qwen3_4b if
#                       /workspace is writable, otherwise .e1_state/ inside the clone
#   E1_RESULTS_DIR      Final results. Default: results/e1/large_qwen3_4b inside the clone
#   E1_PYTHON           The Python to build the venv from
#   E1_SHAM_TOLERANCE   Largest |L| accepted at alpha = 0. Default 0, as the prereg says
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)"
CONFIG="${E1_CONFIG:-$REPO_ROOT/experiments/e1_localization_capture/configs/large_4x3090.json}"
RESULTS_DIR="${E1_RESULTS_DIR:-$REPO_ROOT/results/e1/large_qwen3_4b}"
SHAM_TOLERANCE="${E1_SHAM_TOLERANCE:-0}"
if [[ -n "${E1_STATE_DIR:-}" ]]; then
  STATE="$E1_STATE_DIR"
elif [[ -d /workspace && -w /workspace ]]; then
  STATE=/workspace/e1_large_qwen3_4b
else
  STATE="$REPO_ROOT/.e1_state/large_qwen3_4b"
fi
RUN_DIR="$STATE/run"
VENV="$STATE/venv"
LOG="$STATE/run.log"
LOCK="$STATE/run.lock"
PID_FILE="$STATE/run.pid"
STATUS_FILE="$STATE/status"
ENV_REPORT="$STATE/environment.txt"
MIN_DISK_GB=30
MIN_RESUME_DISK_GB=3
MIN_RAM_GB=22
MIN_GPU_FREE_MIB=18000
MIN_GPU_FREE_TEXT="18 GB"
DRY_RUN_JSON="$RUN_DIR/preflight/dry_run.json"
SUMMARY="$RESULTS_DIR/summary.txt"

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

write_status() {
  printf '%s\n' "$1" >"$STATUS_FILE.tmp"
  mv -f "$STATUS_FILE.tmp" "$STATUS_FILE"
}

read_status() { cat "$STATUS_FILE" 2>/dev/null || true; }

lock_is_free() {
  if command -v flock >/dev/null 2>&1; then
    (flock -n 9) 9>"$LOCK"
  else
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null
  fi
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
##  No final result was written. Finished trials are safe on disk: running the
##  same command again continues without repeating them.
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
##  Copy them off this machine now: nothing was pushed anywhere.
##
##  THE MACHINE KEEPS BILLING UNTIL YOU STOP IT. STOP IT WHEN YOU HAVE THE FILES.
################################################################################

EOF
}

on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -eq 0 ]]; then
    write_status DONE
  else
    write_status FAILED
    banner_failed
  fi
  exit "$code"
}

on_error() {
  [[ -n "$FAIL_REASON" ]] || FAIL_REASON="Line $1 failed: $2"
}

check_machine() {
  say "Step 1 of 7: checking this machine"
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi was not found, so this does not look like an NVIDIA GPU machine."
  nvidia-smi || die "nvidia-smi failed. The NVIDIA driver is not working on this machine."

  local line index name total free visible
  local -a all_gpus=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && all_gpus+=("$line")
  done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits)
  [[ "${#all_gpus[@]}" -gt 0 ]] || die "nvidia-smi lists no GPUs."

  visible=",${CUDA_VISIBLE_DEVICES:-},"
  GPU_LIST=""
  N_GPUS=0
  for line in "${all_gpus[@]}"; do
    IFS=',' read -r index name total free <<<"$line"
    index="${index// /}"
    total="${total// /}"
    free="${free// /}"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    say "  GPU $index:${name} total ${total} MiB, free ${free} MiB"
    # Only plain indices can be compared with nvidia-smi's. Anything else, such as UUIDs, is not a filter here.
    if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9,]+$ && "$visible" != *",$index,"* ]]; then
      say "    skipped: not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    elif ((free < MIN_GPU_FREE_MIB)); then
      say "    skipped: float32 weights need about 16 GB plus room to work, so $MIN_GPU_FREE_TEXT must be free"
    else
      GPU_LIST="${GPU_LIST:+$GPU_LIST,}$index"
      N_GPUS=$((N_GPUS + 1))
    fi
  done
  [[ "$N_GPUS" -ge 1 ]] || die "No GPU has $MIN_GPU_FREE_TEXT free. See 'nvidia-smi' for what is using them."
  if [[ "$N_GPUS" -lt 4 ]]; then
    say "  WARNING: expected 4 GPUs, found $N_GPUS usable. Continuing on $N_GPUS, which takes proportionally longer."
  fi
  say "  Using $N_GPUS GPU(s): $GPU_LIST"
  FIRST_GPU="${GPU_LIST%%,*}"

  local free_disk_gb needed_disk_gb
  free_disk_gb="$(df -Pk "$STATE" | awk 'NR==2 {print int($4 / 1048576)}')"
  needed_disk_gb="$MIN_DISK_GB"
  if [[ -f "$VENV/.e1_ready" && -f "$STATE/.model_ready" ]]; then
    needed_disk_gb="$MIN_RESUME_DISK_GB"
  fi
  say "  Free disk at $STATE: ${free_disk_gb} GB (need ${needed_disk_gb} GB)"
  ((free_disk_gb >= needed_disk_gb)) \
    || die "Only ${free_disk_gb} GB of disk is free at $STATE. This needs ${needed_disk_gb} GB. Free some, or set E1_STATE_DIR to a bigger disk."

  # A container's memory limit can be far below what the host reports as available.
  local ram_kb limit_bytes limit_kb ram_gb
  ram_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  limit_bytes="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)"
  if [[ "$limit_bytes" =~ ^[0-9]+$ ]]; then
    limit_kb=$((limit_bytes / 1024))
    ((limit_kb >= ram_kb)) || ram_kb="$limit_kb"
  fi
  ram_gb=$((ram_kb / 1048576))
  say "  Available RAM: ${ram_gb} GB (need ${MIN_RAM_GB} GB)"
  ((ram_gb >= MIN_RAM_GB)) || die "Only ${ram_gb} GB of RAM is available. Loading the weights needs about ${MIN_RAM_GB} GB."
  # Each load briefly holds about 16 GB of weights in RAM, and loaded workers keep about 3 GB each.
  LOAD_SLOTS=$(((ram_gb - 6 - 3 * N_GPUS) / 20))
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

find_python() {
  local candidate
  for candidate in "${E1_PYTHON:-}" python3 python3.13 python3.12 python3.11 python3.10 python; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      BASE_PYTHON="$candidate"
      say "  Base Python: $(command -v "$candidate") ($("$candidate" --version 2>&1))"
      return 0
    fi
  done
  die "Python 3.10 to 3.13 was not found. Install one, or set E1_PYTHON."
}

make_venv() {
  # System site packages stay visible so that a preinstalled CUDA torch can be reused.
  if "$BASE_PYTHON" -m venv --system-site-packages "$VENV"; then
    return 0
  fi
  rm -rf "$VENV"
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    say "  venv support is missing, so installing it with apt-get"
    local version
    version="$("$BASE_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    apt-get update -qq || true
    apt-get install -y -qq "python${version}-venv" || apt-get install -y -qq python3-venv || true
    if "$BASE_PYTHON" -m venv --system-site-packages "$VENV"; then
      return 0
    fi
    rm -rf "$VENV"
  fi
  say "  Trying virtualenv instead"
  "$BASE_PYTHON" -m pip install --quiet --target "$STATE/virtualenv_pkg" virtualenv \
    && PYTHONPATH="$STATE/virtualenv_pkg" "$BASE_PYTHON" -m virtualenv --system-site-packages "$VENV" \
    || die "Could not create a virtual environment with $BASE_PYTHON. Install python3-venv and run this again."
}

probe_environment() {
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PY" - <<'PROBE'
import sys

import huggingface_hub
import jinja2  # noqa: F401
import numpy
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
    f"huggingface_hub {huggingface_hub.__version__}, numpy {numpy.__version__}, "
    f"GPU {torch.cuda.get_device_name(0)}"
)
PROBE
}

install_torch() {
  local driver_cuda driver_major driver_minor index number
  driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
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
    if "$PY" -m pip install --quiet --ignore-installed --index-url "https://download.pytorch.org/whl/$index" "torch>=2.6"; then
      if probe_environment; then
        TORCH_SOURCE="installed from whl/$index, for driver CUDA $driver_cuda"
        return 0
      fi
    fi
  done
  die "No torch build worked with this machine's driver (CUDA $driver_cuda). See the pip errors above."
}

install_dependencies() {
  say "Step 2 of 7: Python environment in $VENV"
  find_python
  PY="$VENV/bin/python"
  TORCH_SOURCE="kept from the existing venv"
  if [[ -f "$VENV/.e1_ready" && -x "$PY" ]] && probe_environment; then
    say "  Reusing the environment from before"
  else
    rm -rf "$VENV"
    make_venv
    "$PY" -m pip install --quiet --upgrade pip || say "  Could not upgrade pip. Carrying on."
    "$PY" -m pip install --quiet "transformers==5.22.*" numpy jinja2 huggingface_hub safetensors \
      || die "Could not install transformers. Check the network and the pip errors above."
    if probe_environment; then
      TORCH_SOURCE="reused the torch already on this machine"
      say "  Reusing the torch already on this machine"
    else
      say "  The torch on this machine cannot be used, so installing one"
      install_torch
    fi
    touch "$VENV/.e1_ready"
  fi
  {
    printf 'date: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'torch: %s\n' "$TORCH_SOURCE"
    printf 'versions: %s\n' "$(probe_environment)"
    printf 'git commit: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'gpus used: %s\n' "$GPU_LIST"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  } >"$ENV_REPORT"
  say "  $(sed -n 's/^versions: //p' "$ENV_REPORT")"
  say "  Torch: $TORCH_SOURCE"
  say "  Recorded in $ENV_REPORT"
}

download_model() {
  say "Step 3 of 7: model download, once, before any worker starts"
  if [[ -f "$STATE/.model_ready" ]]; then
    say "  Already in $HF_HOME"
    return 0
  fi
  local attempt watcher
  (
    while sleep 60; do
      say "  Model cache so far: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
    done
  ) &
  watcher=$!
  for attempt in 1 2 3 4 5; do
    say "  Download attempt $attempt of 5"
    if HF_HUB_OFFLINE=0 "$PY" - "$CONFIG" <<'DOWNLOAD'
import json
import sys

from huggingface_hub import snapshot_download

with open(sys.argv[1], encoding="utf-8") as handle:
    model = json.load(handle)["model"]
path = snapshot_download(
    model["name"],
    revision=model["revision"],
    ignore_patterns=["*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.ot", "*.onnx"],
)
print(f"Model at {path}")
DOWNLOAD
    then
      kill "$watcher" 2>/dev/null || true
      wait "$watcher" 2>/dev/null || true
      touch "$STATE/.model_ready"
      return 0
    fi
    sleep $((attempt * 20))
  done
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true
  die "The model download failed 5 times. Check the network and free disk."
}

run_preflight() {
  say "Step 4 of 7: dry-run checks on GPU $FIRST_GPU, before spending hours"
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" OMP_NUM_THREADS="$THREADS_PER_WORKER" CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "$PY" -u -m mechinterp.cli preflight --config "$CONFIG" --out "$RUN_DIR" \
    --n-workers "$N_GPUS" --sham-tolerance "$SHAM_TOLERANCE" \
    || die "The dry-run checks failed, so the run stopped before spending hours. See PASS and FAIL above and $DRY_RUN_JSON"
}

run_stages() {
  say "Step 5 of 7: running all stages on $N_GPUS GPU(s). Progress is printed about once a minute."
  "$PY" -u -m mechinterp.cli shard-run --config "$CONFIG" --out "$RUN_DIR" --devices "$GPU_LIST" \
    --load-slots "$LOAD_SLOTS" --threads-per-worker "$THREADS_PER_WORKER" --sham-tolerance "$SHAM_TOLERANCE" \
    || die "The run did not finish. See the reason above. Running the same command again resumes it."
}

write_results() {
  say "Step 6 of 7: writing results"
  mkdir -p "$RESULTS_DIR"
  "$PY" -u -m mechinterp.cli finalize --out "$RUN_DIR" --results-dir "$RESULTS_DIR" --log "$LOG" \
    || die "The results could not be written, so there is no final result."
  # The copied log ends here, and its hash is of that copy. Later lines reach only the live log.
}

inner_main() {
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK"
    flock -n 9 || exit 0 # Another copy won the race. It owns the status file, so leave it alone.
  fi
  printf '%s\n' "$$" >"$PID_FILE"
  trap on_exit EXIT
  trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
  trap 'exit 143' TERM
  trap '' HUP
  write_status RUNNING
  say "======== E1 large tier, started with state in $STATE ========"
  say "Config: $CONFIG"

  mkdir -p "$RUN_DIR" "$STATE/tmp" "$STATE/pip_cache"
  export HF_HOME="$STATE/hf_home"
  export TMPDIR="$STATE/tmp" # A small /tmp is a common way for a torch install to fail.
  export PIP_CACHE_DIR="$STATE/pip_cache"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_INPUT=1
  export PIP_ROOT_USER_ACTION=ignore
  export PYTHONNOUSERSITE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export TOKENIZERS_PARALLELISM=false
  # This model is ungated, and a stale token on a rented image would turn a public download into a 401.
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_ENABLE_HF_TRANSFER
  export PYTHONPATH="$REPO_ROOT/src"

  check_machine
  install_dependencies
  download_model
  # From here the cache is read-only, so workers cannot race on its locks or need the network.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
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
    sleep 5
    waited=$((waited + 5))
    status="$(read_status)"
    if [[ "$status" == "DONE" || "$status" == "FAILED" ]]; then
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
      print_finished
      return 0
      ;;
    FAILED)
      return 1
      ;;
    *)
      say "The background run stopped without reporting, perhaps killed by the system's out-of-memory killer."
      say "No final result was written. Run the same command again to resume."
      say "THE MACHINE KEEPS BILLING UNTIL YOU STOP IT."
      return 1
      ;;
  esac
}

print_finished() {
  if [[ -f "$SUMMARY" ]]; then
    cat "$SUMMARY"
  fi
  banner_done
}

main() {
  case "${1:-}" in
    --inner)
      shift
      mkdir -p "$STATE"
      inner_main "$@"
      return
      ;;
    "")
      ;;
    *)
      printf 'Usage: bash scripts/run_e1_gpu.sh\nThis command takes no arguments.\n' >&2
      return 2
      ;;
  esac

  mkdir -p "$STATE" || { printf 'Cannot create %s\n' "$STATE" >&2; return 1; }
  [[ -f "$CONFIG" ]] || { printf 'Config not found: %s\n' "$CONFIG" >&2; return 1; }
  touch "$LOG"
  if [[ "$(read_status)" == "DONE" && -f "$SUMMARY" ]]; then
    say "This run already finished. Nothing was started."
    print_finished
    return 0
  fi
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

main "$@"
