# Benchmark System Baseline

Snapshot time (UTC): `2026-04-14T11:41:40Z`  
Scope: lightweight environment/system probes only (no inference workloads run).

## 1) Machine / OS

| Item | Command | Result |
|---|---|---|
| Model name | `system_profiler SPHardwareDataType` | `MacBook Pro` |
| Model identifier | `system_profiler SPHardwareDataType` | `Mac17,2` |
| Apple Silicon model string | `sysctl -n machdep.cpu.brand_string` | `Apple M5` |
| Chip (hardware report) | `system_profiler SPHardwareDataType` | `Chip: Apple M5` |
| macOS version | `sw_vers` | `ProductVersion: 26.3` |
| macOS build | `sw_vers` | `BuildVersion: 25D5087f` |

## 2) RAM / Unified Memory Evidence

| Evidence | Command | Result |
|---|---|---|
| Reported memory | `system_profiler SPHardwareDataType` | `Memory: 32 GB` |
| Physical memory bytes | `sysctl -n hw.memsize` | `34359738368` bytes |
| Physical memory GiB (derived) | derived from `hw.memsize / 1024^3` | `32.0 GiB` |

## 3) Python (`.venv`) and Package Versions

| Item | Command | Result |
|---|---|---|
| `.venv` Python version | `.venv/bin/python --version` | `Python 3.14.4` |

| Package | Version source (`importlib.metadata.version`) | Installed version |
|---|---|---|
| `mlx` | `.venv/bin/python` | `0.31.1` |
| `mlx-lm` | `.venv/bin/python` | `0.31.2` |
| `sentencepiece` | `.venv/bin/python` | `0.2.1` |
| `safetensors` | `.venv/bin/python` | `0.7.0` |
| `dflash` | `.venv/bin/python` | `0.1.0` |

## 4) Storage Baseline

| Item | Command | Result |
|---|---|---|
| Current filesystem free space (KiB) | `df -k .` | `61102420` KiB available |
| Current filesystem free space (human) | `df -h .` | `58Gi` available on `/System/Volumes/Data` |
| `HF_HOME` env var | `echo ${HF_HOME:-<unset>}` | `<unset>` |
| Hugging Face cache size (KiB) | `du -sk ~/.cache/huggingface` | `32098968` KiB |
| Hugging Face cache size (human) | `du -sh ~/.cache/huggingface` | `31G` |

## 5) MLX-Relevant Runtime Knobs (`iogpu` sysctl)

From: `sysctl -a | grep -E '^iogpu'`

| Key | Value |
|---|---|
| `iogpu.wired_lwm_mb` | `0` |
| `iogpu.dynamic_lwm` | `1` |
| `iogpu.wired_limit_mb` | `0` |
| `iogpu.debug_flags` | `0` |
| `iogpu.disable_wired_collector` | `0` |
