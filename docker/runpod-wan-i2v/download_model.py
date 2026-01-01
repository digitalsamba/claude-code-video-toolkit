#!/usr/bin/env python3
"""Download Wan2.2-I2V-A14B model with optional HF authentication."""
import os
from huggingface_hub import snapshot_download

os.makedirs('/root/.cache/huggingface', exist_ok=True)

token = os.environ.get('HF_TOKEN') or None
print(f'Downloading Wan2.2-I2V-A14B model (~12GB)...')
print(f'Using HF token: {"yes" if token else "no (anonymous)"}')
snapshot_download('Wan-AI/Wan2.2-I2V-A14B', cache_dir='/root/.cache/huggingface', token=token)
print('Model downloaded successfully')
