import time
import sys
import random
import math
from prometheus_client import start_http_server, Gauge

# Prometheus Metrics
GPU_UTIL = Gauge('gpu_utilization_percent', 'GPU SM Utilization', ['gpu_index', 'gpu_name'])
MEM_UTIL = Gauge('gpu_memory_utilization_percent', 'GPU Memory Controller Utilization', ['gpu_index', 'gpu_name'])
MEM_USED = Gauge('gpu_memory_used_gb', 'GPU Memory Used (GB)', ['gpu_index', 'gpu_name'])
POWER_WATTS = Gauge('gpu_power_watts', 'GPU Power Usage (W)', ['gpu_index', 'gpu_name'])
PCIE_TX = Gauge('gpu_pcie_tx_mb', 'GPU PCIe Transmit (MB/s)', ['gpu_index', 'gpu_name'])
PCIE_RX = Gauge('gpu_pcie_rx_mb', 'GPU PCIe Receive (MB/s)', ['gpu_index', 'gpu_name'])

# Try importing pynvml; if missing, we'll use Mock Mode
try:
    from pynvml import *
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def check_nvidia_driver():
    """Verify that NVML can be initialized. Returns False if mock mode needed."""
    if not HAS_NVML:
        return False
    try:
        nvmlInit()
        return True
    except NVMLError:
        return False

def get_real_metrics(handle):
    """Fetch real metrics from physical GPU."""
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        
        # PCIe throughput (TX/RX)
        tx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES) / 1024.0
        rx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES) / 1024.0

        return {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "mem_used": mem_info.used / 1024**3,
            "mem_total": mem_info.total / 1024**3,
            "power_watts": power,
            "pcie_tx": tx,
            "pcie_rx": rx
        }
    except NVMLError:
        return None

def get_mock_metrics(index, tick):
    """Generate realistic fake metrics for testing without a GPU."""
    # Simulate a training workload (sine wave)
    load = (math.sin(tick * 0.5) + 1) / 2  # 0.0 to 1.0
    
    return {
        "gpu_util": int(load * 95) + random.randint(-2, 2),
        "mem_util": int(load * 80) + random.randint(-5, 5),
        "mem_used": 40 + (load * 20), # 40-60GB used
        "mem_total": 80.0,            # H100 80GB
        "power_watts": 100 + (load * 600), # 100-700W
        "pcie_tx": 2000 + (load * 1000),
        "pcie_rx": 4000 + (load * 2000)
    }

def monitor_loop(mock_mode=False, port=8000):
    """Main loop: Update Prometheus metrics + Print to CLI."""
    
    # Start Prometheus Server
    start_http_server(port)
    print(f"🚀 Prometheus Metrics available at http://localhost:{port}/metrics")
    
    device_count = 1 if mock_mode else nvmlDeviceGetCount()
    tick = 0

    try:
        while True:
            # Clear screen for dashboard feel
            print("\033[2J\033[H", end="")
            print(f"--- GPU Pulse (Mode: {'MOCK' if mock_mode else 'REAL'}) ---")
            
            for i in range(device_count):
                if mock_mode:
                    name = "NVIDIA H100 (Simulated)"
                    metrics = get_mock_metrics(i, tick)
                else:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    name = nvmlDeviceGetName(handle).decode('utf-8')
                    metrics = get_real_metrics(handle)

                if metrics:
                    # Update Prometheus
                    GPU_UTIL.labels(i, name).set(metrics['gpu_util'])
                    MEM_UTIL.labels(i, name).set(metrics['mem_util'])
                    MEM_USED.labels(i, name).set(metrics['mem_used'])
                    POWER_WATTS.labels(i, name).set(metrics['power_watts'])
                    PCIE_TX.labels(i, name).set(metrics['pcie_tx'])
                    PCIE_RX.labels(i, name).set(metrics['pcie_rx'])

                    # CLI Output
                    print(f"\nGPU [{i}]: {name}")
                    print(f"  ├── 🧠 SM Util:      {metrics['gpu_util']:>3}%")
                    print(f"  ├── 💾 Mem Util:     {metrics['mem_util']:>3}%  ({metrics['mem_used']:.1f} / {metrics['mem_total']:.1f} GB)")
                    print(f"  ├── ⚡ Power:        {metrics['power_watts']:.1f} W")
                    print(f"  └── ↔️  PCIe (TX/RX): {metrics['pcie_tx']:.0f} / {metrics['pcie_rx']:.0f} MB/s")

            tick += 0.5
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

if __name__ == "__main__":
    # Auto-detect mode
    if check_nvidia_driver():
        print("✅ NVIDIA Driver detected. Running in REAL mode.")
        monitor_loop(mock_mode=False)
        nvmlShutdown()
    else:
        print("⚠️  No NVIDIA GPU detected. Running in MOCK mode (Simulation).")
        print("   (This allows testing logic on CPU-only servers)")
        time.sleep(2)
        monitor_loop(mock_mode=True)
