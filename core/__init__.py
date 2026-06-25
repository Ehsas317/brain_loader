#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: core/__init__.py                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Core package initializer for Port. Exports main classes.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Core Package

Exposes the main orchestration classes:
- PortOrchestrator: Main execution loop
- ModelManager: MLX + Ollama backends
- Coordinator: Pure Python state machine
"""

from core.orchestrator import PortOrchestrator
from core.model_manager import ModelConfig, create_model_manager, BaseModelManager
from core.coordinator import Coordinator

__all__ = [
    "PortOrchestrator",
    "ModelConfig",
    "create_model_manager",
    "BaseModelManager",
    "Coordinator",
]
