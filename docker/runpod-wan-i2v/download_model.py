#!/usr/bin/env python3
"""Download Wan2.2-I2V-A14B model with optional HF authentication."""
import os
from huggingface_hub import snapshot_download, login

os.makedirs('/root/.cache/huggingface', exist_ok=True)

token = os.environ.get('HF_TOKEN', '')
if token:
    print('Authenticating with HuggingFace...')
    login(token=token)

print('Downloading Wan2.2-I2V-A14B model (~12GB)...')
snapshot_download('Wan-AI/Wan2.2-I2V-A14B', cache_dir='/root/.cache/huggingface')
print('Model downloaded successfully')
