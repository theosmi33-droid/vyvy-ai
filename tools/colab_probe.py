"""Smoke test on a Colab machine: GPU, CUDA, disk and 4-bit bitsandbytes (see colab-probe.yml)."""

import shutil, subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print("python", sys.version.split()[0])
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("disk /content free GB", shutil.disk_usage("/content").free // 2**30, "/tmp", shutil.disk_usage("/tmp").free // 2**30)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2", "transformers==4.57.6", "peft==0.17.1"], check=True)
import bitsandbytes as bnb, torch
x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
q, s = bnb.functional.quantize_4bit(x, quant_type="nf4")
print("bnb", bnb.__version__, "4bit ok", tuple(bnb.functional.dequantize_4bit(q, s).shape))
