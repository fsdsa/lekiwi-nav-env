"""Quick test: VLM + VLA simultaneous API test."""
import requests, base64, io, time, subprocess
import numpy as np
from PIL import Image

img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()

# VLM
t0 = time.perf_counter()
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 10,
})
choices = r.json()["choices"]
print(f"[VLM] {choices[0]['message']['content']} ({(time.perf_counter()-t0)*1000:.0f}ms)")

# VLA
t0 = time.perf_counter()
r = requests.post("http://localhost:8002/act", json={
    "base_image_b64": b64,
    "state": [0.0] * 9,
    "instruction": "pick up the red cup",
}, timeout=120)
dt = (time.perf_counter() - t0) * 1000
print(f"[VLA] status={r.status_code} ({dt:.0f}ms)")
if r.status_code == 200:
    d = r.json()
    n = len(d["actions"])
    dim = len(d["actions"][0]) if n > 0 else 0
    print(f"  {n} actions x {dim}D, server={d['inference_time_ms']:.0f}ms")
    if n > 0:
        print(f"  first: {[round(x, 3) for x in d['actions'][0][:10]]}")
else:
    print(f"  error: {r.text[:400]}")

# GPU
g = subprocess.check_output(
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
    shell=True,
).decode().strip()
print(f"[GPU] {g}")

# Health
h = requests.get("http://localhost:8002/health").json()
print(f"[VLA alive] {h['status']}, {h['gpu_memory_mb']:.0f}MB")
