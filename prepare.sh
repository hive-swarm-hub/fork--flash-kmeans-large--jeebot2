#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Python dev headers (needed for Triton) ==="
if ! python3 -c "import sysconfig; assert sysconfig.get_path('include')" 2>/dev/null || \
   ! [ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')/Python.h" ]; then
    apt-get install -y python3-dev 2>/dev/null || {
        echo "No sudo — installing headers from deb..."
        mkdir -p /tmp/pydev && cd /tmp/pydev
        apt-get download libpython3.10-dev 2>/dev/null && \
            dpkg-deb -x libpython3.10-dev*.deb /tmp/pydev/extracted
        cd - >/dev/null
        echo "Set C_INCLUDE_PATH before running eval:"
        echo '  export C_INCLUDE_PATH="/tmp/pydev/extracted/usr/include/python3.10:/tmp/pydev/extracted/usr/include/x86_64-linux-gnu:/tmp/pydev/extracted/usr/include"'
        echo '  export CPATH="$C_INCLUDE_PATH"'
    }
else
    echo "Python dev headers: OK"
fi

echo ""
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Verifying CUDA + Triton ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import triton
print(f'Triton: {triton.__version__}')
print('OK: CUDA + Triton ready')
"
