import time
import sys
import random
import math
import threading
from prometheus_client import start_http_server, Gauge, generate_latest
from http.server import HTTPServer, BaseHTTPRequestHandler

# Prometheus Metrics
GPU_UTIL = Gauge('gpu_utilization_percent', 'GPU SM Utilization', ['gpu_index', 'gpu_name'])
MEM_UTIL = Gauge('gpu_memory_utilization_percent', 'GPU Memory Controller Utilization', ['gpu_index', 'gpu_name'])
MEM_USED = Gauge('gpu_memory_used_gb', 'GPU Memory Used (GB)', ['gpu_index', 'gpu_name'])
POWER_WATTS = Gauge('gpu_power_watts', 'GPU Power Usage (W)', ['gpu_index', 'gpu_name'])
PCIE_TX = Gauge('gpu_pcie_tx_mb', 'GPU PCIe Transmit (MB/s)', ['gpu_index', 'gpu_name'])
PCIE_RX = Gauge('gpu_pcie_rx_mb', 'GPU PCIe Receive (MB/s)', ['gpu_index', 'gpu_name'])

# Global state for the simple UI
LATEST_METRICS = []

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

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(generate_latest())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def get_html(self):
        # Simple HTML Dashboard with auto-refresh
        rows = ""
        for m in LATEST_METRICS:
            # Color coding for utilization
            util_color = "red" if m['gpu_util'] > 90 else "green" if m['gpu_util'] < 50 else "orange"
            
            rows += f"""
            <div class="gpu-card">
                <h2>{m['name']} <span class="index">#{m['index']}</span></h2>
                <div class="metric-row">
                    <div class="metric">
                        <span class="label">Compute (SM)</span>
                        <div class="bar-container">
                            <div class="bar" style="width: {m['gpu_util']}%; background-color: {util_color};"></div>
                        </div>
                        <span class="value">{m['gpu_util']}%</span>
                    </div>
                    <div class="metric">
                        <span class="label">Memory BW</span>
                        <div class="bar-container">
                            <div class="bar" style="width: {m['mem_util']}%; background-color: #3498db;"></div>
                        </div>
                        <span class="value">{m['mem_util']}%</span>
                    </div>
                </div>
                <div class="details">
                    <p>💾 VRAM: <b>{m['mem_used']:.1f}</b> / {m['mem_total']:.1f} GB</p>
                    <p>⚡ Power: <b>{m['power_watts']:.0f} W</b></p>
                    <p>↔️ PCIe: TX {m['pcie_tx']:.0f} MB/s | RX {m['pcie_rx']:.0f} MB/s</p>
                </div>
            </div>
            """
        
        return f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>GPU Pulse Dashboard</title>
            <meta http-equiv="refresh" content="1">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #1a1a1a; color: #ecf0f1; padding: 20px; }}
                h1 {{ text-align: center; color: #e74c3c; }}
                .container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
                .gpu-card {{ background: #2c3e50; border-radius: 8px; padding: 20px; width: 300px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
                h2 {{ margin-top: 0; font-size: 1.2em; border-bottom: 1px solid #34495e; padding-bottom: 10px; }}
                .index {{ float: right; color: #7f8c8d; font-size: 0.8em; }}
                .metric-row {{ margin: 15px 0; }}
                .metric {{ margin-bottom: 10px; }}
                .label {{ display: block; font-size: 0.8em; margin-bottom: 3px; color: #bdc3c7; }}
                .bar-container {{ background: #34495e; height: 10px; border-radius: 5px; overflow: hidden; }}
                .bar {{ height: 100%; transition: width 0.5s ease; }}
                .value {{ float: right; font-size: 0.9em; font-weight: bold; margin-top: -14px; }}
                .details p {{ margin: 5px 0; font-size: 0.9em; color: #bdc3c7; }}
                .details b {{ color: #fff; }}
            </style>
        </head>
        <body>
            <h1>🔥 GPU Pulse Monitor</h1>
            <div class="container">
                {rows}
            </div>
            <p style="text-align: center; margin-top: 20px; color: #7f8c8d;">Auto-refreshing every 1s • <a href="/metrics" style="color: #3498db;">Prometheus Metrics</a></p>
        </body>
        </html>
        """

def monitor_loop(mock_mode=False, port=8000):
    """Main loop: Update metrics & global state."""
    
    # Start Custom HTTP Server (Handles both UI and Metrics)
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    
    print(f"🚀 Dashboard running at http://localhost:{port}")
    print(f"📊 Prometheus metrics at http://localhost:{port}/metrics")
    
    device_count = 1 if mock_mode else nvmlDeviceGetCount()
    tick = 0
    global LATEST_METRICS

    try:
        while True:
            # Clear CLI screen
            print("\033[2J\033[H", end="")
            print(f"--- GPU Pulse (Mode: {'MOCK' if mock_mode else 'REAL'}) ---")
            
            current_batch = []
            
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
                    
                    # Store for UI
                    metrics['name'] = name
                    metrics['index'] = i
                    current_batch.append(metrics)

                    # CLI Output
                    print(f"\nGPU [{i}]: {name}")
                    print(f"  ├── 🧠 SM Util:      {metrics['gpu_util']:>3}%")
                    print(f"  ├── 💾 Mem Util:     {metrics['mem_util']:>3}%")
                    print(f"  └── ⚡ Power:        {metrics['power_watts']:.0f} W")

            LATEST_METRICS = current_batch
            tick += 0.5
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        server.shutdown()

if __name__ == "__main__":
    if check_nvidia_driver():
        print("✅ NVIDIA Driver detected. Running in REAL mode.")
        monitor_loop(mock_mode=False)
        nvmlShutdown()
    else:
        print("⚠️  No NVIDIA GPU detected. Running in MOCK mode.")
        time.sleep(2)
        monitor_loop(mock_mode=True)
