#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt

python - <<'PY'
import sys
import torch
import numpy
import cv2
import imageio
import matplotlib
import tqdm

print('Python:', sys.version.replace('\n', ' '))
print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Check the rented GPU and PyTorch image.')
print('GPU:', torch.cuda.get_device_name(0))
print('GPU memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print('NumPy:', numpy.__version__)
print('OpenCV:', cv2.__version__)
print('ImageIO:', imageio.__version__)
print('Matplotlib:', matplotlib.__version__)
print('tqdm:', tqdm.__version__)
PY