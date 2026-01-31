#!/usr/bin/env bash
# =============================================================================
# Setup script for a fresh VM (Ubuntu 22.04 + NVIDIA GPU).
# Target model: ai-sage/GigaChat3-10B-A1.8B (MoE, ~12 GB VRAM in bf16)
#
# Installs: NVIDIA drivers, Docker, nvidia-container-toolkit, project deps.
#
# Usage:
#   chmod +x scripts/setup_vm.sh
#   sudo bash scripts/setup_vm.sh
# =============================================================================

set -euo pipefail

echo "============================================"
echo " VM Setup: Entropy-Reward Experiments"
echo " Model: GigaChat3-10B-A1.8B (MoE)"
echo "============================================"

# --- 1. System update ---
echo "[1/7] Updating system packages..."
apt-get update && apt-get upgrade -y

# --- 2. NVIDIA drivers (if not present) ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[2/7] Installing NVIDIA drivers..."
    apt-get install -y linux-headers-$(uname -r)
    apt-get install -y nvidia-driver-550
    echo "  NVIDIA drivers installed. Reboot may be required."
else
    echo "[2/7] NVIDIA drivers already present:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# --- 3. Docker ---
if ! command -v docker &>/dev/null; then
    echo "[3/7] Installing Docker..."
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker && systemctl start docker
else
    echo "[3/7] Docker already installed."
fi

# --- 4. NVIDIA Container Toolkit ---
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "[4/7] Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    echo "[4/7] NVIDIA Container Toolkit already installed."
fi

# --- 5. Add current user to docker group ---
echo "[5/7] Adding user to docker group..."
if [ -n "${SUDO_USER:-}" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "  Added $SUDO_USER to docker group (re-login to apply)."
fi

# --- 6. Python environment (for non-Docker usage) ---
echo "[6/7] Setting up Python environment..."
if ! command -v python3.11 &>/dev/null; then
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
fi

# --- 7. Verify ---
echo "[7/7] Verification..."
docker --version
nvidia-smi || echo "  WARNING: nvidia-smi failed. Reboot may be needed."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>/dev/null \
    && echo "  GPU-in-Docker: OK" \
    || echo "  GPU-in-Docker: FAILED (reboot and re-run)"

echo ""
echo "============================================"
echo " Setup complete."
echo ""
echo " Model: ai-sage/GigaChat3-10B-A1.8B"
echo " VRAM:  ~12 GB (bf16, 1.8B active params)"
echo " License: MIT (no token needed)"
echo ""
echo " Next steps:"
echo "   1. Reboot if NVIDIA drivers were just installed"
echo "   2. Re-login for docker group membership"
echo "   3. cd /path/to/agentic-mid-training"
echo "   4. cp .env.example .env && nano .env"
echo "   5. docker compose build"
echo "   6. docker compose run --rm base"
echo "============================================"
