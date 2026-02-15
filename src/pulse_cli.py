import time
import sys
from pynvml import *

def check_nvidia_driver():
    """Verify that NVML can be initialized (driver is loaded)."""
    try:
        nvmlInit()
        version = nvmlSystemGetDriverVersion()
        print(f"✅ NVIDIA Driver Detected: {version.decode('utf-8')}")
        return True
    except NVMLError as error:
        print(f"❌ Failed to initialize NVML: {error}")
        print("   - Is the NVIDIA driver installed?")
        print("   - Try: sudo dnf install cuda-drivers")
        return False

def get_gpu_metrics(handle):
    """Fetch critical metrics for a single GPU."""
    
    # 1. Utilization (Compute & Memory Controller)
    util = nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu       # SM Utilization (%)
    mem_util = util.memory    # Memory Controller Utilization (%)
    
    # 2. Memory Usage (Capacity)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    mem_used_gb = mem_info.used / 1024**3
    mem_total_gb = mem_info.total / 1024**3
    
    # 3. Power Usage
    power_watts = nvmlDeviceGetPowerUsage(handle) / 1000.0
    try:
        power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    except NVMLError:
        power_limit = 0.0 # Default if unknown

    # 4. PCIe Throughput (TX = Transmit, RX = Receive)
    # throughput is in KB/s
    tx_mb = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES) / 1024.0
    rx_mb = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES) / 1024.0
    
    # 5. Clock Speeds
    graphics_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
    sm_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
    
    return {
        "gpu_util": gpu_util,
        "mem_util": mem_util,
        "mem_used": mem_used_gb,
        "mem_total": mem_total_gb,
        "power_watts": power_watts,
        "power_limit": power_limit,
        "pcie_tx": tx_mb,
        "pcie_rx": rx_mb,
        "clock_graphics": graphics_clock,
        "clock_sm": sm_clock
    }

def monitor_loop(interval=1.0):
    """Continuously monitor and print metrics."""
    device_count = nvmlDeviceGetCount()
    print(f"🔍 Monitoring {device_count} GPU(s) - Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Clear screen (ANSI escape code)
            print("\033[2J\033[H", end="")
            print(f"--- GPU Pulse (Interval: {interval}s) ---")
            
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle).decode('utf-8')
                metrics = get_gpu_metrics(handle)
                
                print(f"\nGPU [{i}]: {name}")
                print(f"  ├── 🧠 Compute (SM):    {metrics['gpu_util']:>3}%  (Clock: {metrics['clock_sm']} MHz)")
                print(f"  ├── 💾 Memory Ctrl:     {metrics['mem_util']:>3}%  (Used: {metrics['mem_used']:.1f}/{metrics['mem_total']:.1f} GB)")
                print(f"  ├── ⚡ Power:           {metrics['power_watts']:.1f}W / {metrics['power_limit']:.1f}W")
                print(f"  └── ↔️  PCIe (TX/RX):    {metrics['pcie_tx']:.1f} / {metrics['pcie_rx']:.1f} MB/s")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")

if __name__ == "__main__":
    if check_nvidia_driver():
        monitor_loop()
        nvmlShutdown()
    else:
        sys.exit(1)
