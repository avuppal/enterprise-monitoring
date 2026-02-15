# GPU Pulse: Enterprise-Grade AI Infrastructure Monitoring

**Project 1: The Observer**

This repository contains the source code for `gpu-pulse`, a lightweight CLI and exporter for monitoring NVIDIA Data Center GPUs (A100/H100) on RHEL/CentOS systems.

## Why this matters
Most monitoring tools (e.g., `nvidia-smi`) focus on memory *capacity* (VRAM usage). For AI workloads, the true bottlenecks are:
1.  **SM Utilization:** Are the CUDA cores actually busy?
2.  **Memory Bandwidth:** Is the HBM saturated?
3.  **Power State:** Is the GPU throttling?
4.  **PCIe Throughput:** Is the CPU-to-GPU link the bottleneck?

## Architecture
- **Language:** Python 3.9+ (using `pynvml` bindings for NVML)
- **Target OS:** RHEL 8/9, CentOS Stream
- **Metrics:** Prometheus-compatible exposition format

## Getting Started (RHEL)

### Prerequisites
```bash
sudo dnf install python39 python3-pip
sudo pip3 install pynvml prometheus_client
```

### Running the CLI
```bash
python3 src/pulse_cli.py
```
