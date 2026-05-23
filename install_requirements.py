#!/usr/bin/env python3
"""
Installation and verification script for PandaPush RL project dependencies.
Installs and verifies: gymnasium, numpy, panda_gym, stable_baselines3, torch (with CUDA GPU support)
"""

import subprocess
import sys


def run_command(cmd_list, description):
    """Run a command (as list) and return success status."""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🚀 PandaPush RL - Dependency Installer with GPU Support")
    print("="*60)
    
    # Step 1: Upgrade pip
    print("\n[1/4] Upgrading pip, setuptools, wheel...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    if not run_command(cmd, "Upgrading pip"):
        print("⚠️  pip upgrade failed, continuing anyway...")
    
    # Step 2: Install CPU dependencies
    print("\n[2/4] Installing core dependencies (gymnasium, numpy, panda-gym, stable-baselines3)...")
    packages = [
        "gymnasium",
        "numpy",
        "panda-gym",
        "stable-baselines3",
    ]
    
    for package in packages:
        cmd = [sys.executable, "-m", "pip", "install", "-U", package]
        if not run_command(cmd, f"Installing {package}"):
            print(f"⚠️  Failed to install {package}")
    
    # Step 3: Install PyTorch with CUDA support
    print("\n[3/4] Installing PyTorch 1.13.1+cu117 with CUDA GPU support...")
    print("Installing: torch==1.13.1, torchvision, torchaudio...")
    cmd = [sys.executable, "-m", "pip", "install", "torch==1.13.1+cu117", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu117"]
    if not run_command(cmd, "Installing PyTorch 1.13.1+cu117"):
        print("⚠️  PyTorch installation may have issues")
    
    # Step 4: Verify installation
    print("\n[4/4] Verifying installation...")
    print("="*60)
    
    verify_script = """
import sys
try:
    import gymnasium
    print("✅ gymnasium imported successfully")
except ImportError as e:
    print(f"❌ gymnasium import failed: {e}")
    sys.exit(1)

try:
    import numpy
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    sys.exit(1)

try:
    import panda_gym
    print("✅ panda_gym imported successfully")
except ImportError as e:
    print(f"❌ panda_gym import failed: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import SAC, PPO
    print("✅ stable_baselines3 imported successfully")
except ImportError as e:
    print(f"❌ stable_baselines3 import failed: {e}")
    sys.exit(1)

try:
    import torch
    print("✅ torch imported successfully")
    print(f"   PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  No GPU detected (CPU mode)")
except ImportError as e:
    print(f"❌ torch import failed: {e}")
    sys.exit(1)
"""
    
    cmd = [sys.executable, "-c", verify_script]
    run_command(cmd, "Verifying all imports")
    
    print("\n" + "="*60)
    print("✅ Installation complete!")
    print("="*60)
    print("\n🎯 Next steps:")
    print("   1. Navigate to: cd 'd:\\MLDL project\\FAIML-RL-26\\part2'")
    print("   2. Train SAC:   python train_sb3.py --algo sac --sampling-strategy none --env-type source")
    print("   3. Train PPO:   python train_sb3.py --algo ppo --sampling-strategy none --env-type source")
    print("\n")


if __name__ == "__main__":
    main()
