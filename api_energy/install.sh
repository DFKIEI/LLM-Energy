#!/usr/bin/env bash
# filepath: /home/banwari/llm_energy/LLM-Energy/api_energy/install.sh

set -euo pipefail

echo "=== Task Prolog: Installing dependencies on $(hostname) ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Setup HF cache in user's netscratch
export HF_HOME="/netscratch/$USER/hf"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"
echo "HF_HOME: $HF_HOME"

# Fix pip - handle if already working
echo "Checking pip..."
if python -m pip --version &>/dev/null; then
    echo "Pip version: $(python -m pip --version)"
else
    echo "Reinstalling pip..."
    if ! python -m ensurepip --upgrade 2>/dev/null; then
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        python /tmp/get-pip.py --force-reinstall
        rm -f /tmp/get-pip.py
    fi
    echo "Pip version: $(python -m pip --version)"
fi

# Upgrade pip quietly
echo "Upgrading pip..."
python -m pip install --upgrade pip --no-warn-script-location --quiet || true

# Find requirements file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: Requirements not found: $REQ_FILE"
    echo "SCRIPT_DIR: $SCRIPT_DIR"
    echo "Contents of SCRIPT_DIR:"
    ls -la "$SCRIPT_DIR" || true
    exit 1
fi

echo "Installing packages from: $REQ_FILE"

# Install with retry and better error handling
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
    echo "Installation attempt $i/$MAX_RETRIES..."
    if python -m pip install --upgrade --no-cache-dir -r "$REQ_FILE" 2>&1 | tee /tmp/pip_install.log; then
        echo "✓ Packages installed successfully"
        break
    else
        if [ $i -eq $MAX_RETRIES ]; then
            echo "ERROR: Failed to install packages after $MAX_RETRIES attempts"
            echo "Last error:"
            tail -20 /tmp/pip_install.log || true
            exit 1
        fi
        echo "Installation failed, waiting 5 seconds before retry..."
        sleep 5
    fi
done

# HuggingFace login (optional, don't fail if it doesn't work)
if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
    echo "Logging in to HuggingFace..."
    python -m pip install -q --no-cache-dir "huggingface_hub>=0.23" || true
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}', add_to_git_credential=True)" 2>/dev/null || echo "HF login skipped"
fi

# Verify critical packages
echo "Verifying installations..."
if ! python -c "
import sys
try:
    import transformers, torch
    print('✓ transformers:', transformers.__version__)
    print('✓ torch:', torch.__version__)
    print('✓ CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✓ CUDA devices:', torch.cuda.device_count())
    
    # Try carbontracker but don't fail if it doesn't work
    try:
        import carbontracker
        print('✓ carbontracker: installed')
    except:
        print('⚠ carbontracker: not available (non-critical)')
    
    sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: Package verification failed"
    exit 1
fi

echo "=== Installation complete on $(hostname) ==="
exit 0