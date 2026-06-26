"""
Forge Model Manager v3 — Dual Backend (MLX + Ollama)

Manages the lifecycle of LLM models with support for:
- MLX backend (Apple Silicon via Metal GPU)
- Ollama backend (any OS, CPU/GPU)

Single model slot: only one model loaded at a time.
RAM-aware: estimates usage before loading.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("forge.model_manager")

# ─────────────────────────────────────────────────────────────────
# MLX availability check (Apple Silicon only)
# ─────────────────────────────────────────────────────────────────
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    logger.info("MLX not available — using Ollama backend only")


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    description: str
    path: str
    ram_estimate_gb: float
    max_tokens: int = 4096
    temperature: float = 0.7
    backend: str = "mlx"  # "mlx" or "ollama"


# ─────────────────────────────────────────────────────────────────
# Abstract Base Model Manager
# ─────────────────────────────────────────────────────────────────
class BaseModelManager(ABC):
    """Abstract base for all model managers."""

    @abstractmethod
    def load(self, model_key: str, config: ModelConfig) -> bool:
        """Load a model. Returns True on success."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload current model and free memory."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text from prompt."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        ...

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage stats."""
        ...


# ─────────────────────────────────────────────────────────────────
# MLX Backend — Apple Silicon
# ─────────────────────────────────────────────────────────────────
class MLXModelManager(BaseModelManager):
    """MLX model manager for Apple Silicon."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self._currently_loaded = None

    def load(self, model_key: str, config: ModelConfig) -> bool:
        if not MLX_AVAILABLE:
            logger.error("MLX not available — cannot load %s", model_key)
            return False

        if self._currently_loaded == model_key:
            logger.info("[MLX] Model %s already loaded", model_key)
            return True

        self.unload()

        if not Path(config.path).exists():
            logger.error("[MLX] Model path not found: %s", config.path)
            return False

        logger.info("[MLX] Loading %s (%.1f GB est.)...", model_key, config.ram_estimate_gb)
        try:
            self.model, self.tokenizer = mlx_load(config.path)
            self.config = config
            self._currently_loaded = model_key
            logger.info("[MLX] %s loaded successfully", model_key)
            return True
        except Exception as e:
            logger.error("[MLX] Failed to load %s: %s", model_key, e)
            return False

    def unload(self) -> None:
        if self._currently_loaded is None:
            return

        logger.info("[MLX] Unloading %s...", self._currently_loaded)
        self.model = None
        self.tokenizer = None
        self.config = None
        self._currently_loaded = None
        gc.collect()
        if MLX_AVAILABLE:
            try:
                mx.synchronize()
            except Exception as e:
                logger.warning("[MLX] mx.synchronize() failed during unload: %s", e)
            try:
                if hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception as e:
                logger.warning("[MLX] clear_cache failed during unload: %s", e)
        time.sleep(5)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load() first.")

        max_tokens = max_tokens or self.config.max_tokens

        try:
            result = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=self.config.temperature,
                verbose=False,
            )
            return result
        except Exception as e:
            logger.error("[MLX] Generation failed: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._currently_loaded is not None

    def get_memory_usage(self) -> Dict[str, float]:
        if not MLX_AVAILABLE:
            return {"active_gb": 0, "peak_gb": 0}
        try:
            active = mx.metal.get_active_memory() / 1e9
            peak = mx.metal.get_peak_memory() / 1e9
            return {"active_gb": active, "peak_gb": peak}
        except Exception as e:
            logger.warning("[MLX] Failed to get memory usage: %s", e)
            return {"active_gb": 0, "peak_gb": 0}


# ─────────────────────────────────────────────────────────────────
# Ollama Backend — Any system
# ─────────────────────────────────────────────────────────────────
class OllamaModelManager(BaseModelManager):
    """Ollama model manager for any OS."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.config = None
        self._currently_loaded = None

    def load(self, model_key: str, config: ModelConfig) -> bool:
        # Ollama manages loading internally — just verify it's available
        if self._is_model_available(config.path):
            self.config = config
            self._currently_loaded = model_key
            logger.info("[Ollama] Model %s is available", model_key)
            return True
        logger.error("[Ollama] Model %s not available. Run: ollama pull %s",
                     config.path, config.path)
        return False

    def unload(self) -> None:
        if self._currently_loaded:
            logger.info("[Ollama] Unloading %s...", self._currently_loaded)
            self._currently_loaded = None
            self.config = None

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load() first.")

        import requests

        max_tokens = max_tokens or self.config.max_tokens

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.config.path,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.config.temperature, "num_predict": max_tokens},
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error("[Ollama] Generation failed: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._currently_loaded is not None

    def get_memory_usage(self) -> Dict[str, float]:
        # Ollama manages its own memory
        return {"active_gb": 0, "peak_gb": 0}

    def _is_model_available(self, model_name: str) -> bool:
        import requests
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m.get("name") == model_name for m in models)
        except Exception as e:
            logger.warning("[Ollama] Failed to check model availability: %s", e)
            return False


# ─────────────────────────────────────────────────────────────────
# Unified Model Manager — Single hot-swap slot
# ─────────────────────────────────────────────────────────────────
class ForgeModelManager:
    """
    Unified model manager with single hot-swap slot.
    Only one model in RAM at a time — load/unload automatically.
    """

    def __init__(self, device_ram_gb: float = 32.0, ollama_host: str = "http://localhost:11434"):
        self.device_ram_gb = device_ram_gb
        self.mlx = MLXModelManager()
        self.ollama = OllamaModelManager(host=ollama_host)
        self._active_backend: Optional[BaseModelManager] = None

    def load(self, model_key: str, config: ModelConfig) -> bool:
        """Load a model using the appropriate backend."""
        if config.backend == "mlx" and MLX_AVAILABLE:
            success = self.mlx.load(model_key, config)
            self._active_backend = self.mlx if success else None
            return success
        elif config.backend == "ollama":
            success = self.ollama.load(model_key, config)
            self._active_backend = self.ollama if success else None
            return success
        else:
            logger.error("Unknown backend: %s", config.backend)
            return False

    def unload(self) -> None:
        """Unload any loaded model."""
        if self._active_backend:
            self._active_backend.unload()
            self._active_backend = None

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using the loaded model."""
        if self._active_backend is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._active_backend.generate(prompt, max_tokens)

    def is_loaded(self) -> bool:
        return self._active_backend is not None

    def get_memory_usage(self) -> Dict[str, float]:
        mlx_mem = self.mlx.get_memory_usage()
        return mlx_mem

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
