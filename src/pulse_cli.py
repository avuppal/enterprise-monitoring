from pynvml import *
import sys

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

def list_gpus():
    """List detected GPU devices."""
    device_count = nvmlDeviceGetCount()
    print(f"\n🔍 Found {device_count} GPU(s):")
    
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        pci_info = nvmlDeviceGetPciInfo(handle)
        
        # Get basic memory info
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        mem_used_gb = mem_info.used / 1024**3
        mem_total_gb = mem_info.total / 1024**3
        
        print(f"   [{i}] {name.decode('utf-8')} (PCIe: {pci_info.busId.decode('utf-8')})")
        print(f"       VRAM: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB")

    return device_count

if __name__ == "__main__":
    if check_nvidia_driver():
        list_gpus()
        nvmlShutdown()
    else:
        sys.exit(1)
