#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: core/model_manager.py                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Model Manager — dual backend support for MLX (Apple Silicon) and
#   Ollama (any system). Abstract base with concrete implementations.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Model Manager (Dual Backend)

Provides an abstract BaseModelManager with two implementations:
  MLXModelManager   — Apple Silicon (mlx-lm). Precise unified-memory control.
  OllamaModelManager — Any system with Ollama. Ollama manages its own memory pool.

Use the factory:
    manager = create_model_manager("mlx",   gc_sleep=2.0)
    manager = create_model_manager("ollama", host="http://localhost:11434")
"""

import gc
import time
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Try importing MLX — only available on Apple Silicon
# ─────────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Shared config dataclass
# ─────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    path: str
    max_tokens: int
    temperature: float
    description: str
    ram_estimate_gb: float
    role: str  # "brain" or "specialist"


# ─────────────────────────────────────────────────────────────────
# Abstract base — all backends implement this interface
# ─────────────────────────────────────────────────────────────────
class BaseModelManager(ABC):
    """
    Abstract model manager.
    Every backend must implement load(), offload(), generate().
    The coordinator calls exactly these three methods.
    """

    @abstractmethod
    def load(self, config: ModelConfig) -> None:
        """Load a model. Must unload any existing model first."""
        ...

    @abstractmethod
    def offload(self) -> None:
        """Unload current model and free memory."""
        ...

    @abstractmethod
    def generate(self, prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text with the currently loaded model."""
        ...

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": getattr(self, "_currently_loaded", None),
            "total_swaps": getattr(self, "_total_swaps", 0),
        }

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.offload()


# ─────────────────────────────────────────────────────────────────
# MLX Backend — Apple Silicon only
# ─────────────────────────────────────────────────────────────────
class MLXModelManager(BaseModelManager):
    """
    MLX backend for Apple Silicon (M1/M2/M3/M4).

    Loads models via mlx-lm from a HuggingFace repo (locally cached).
    Explicit load/offload gives precise control over unified memory.

    Models: https://huggingface.co/mlx-community
    Download: huggingface-cli download mlx-community/<model-name>
    """

    def __init__(self, gc_sleep: float = 2.0, aggressive_cleanup: bool = True):
        if not MLX_AVAILABLE:
            raise ImportError(
                "mlx and mlx-lm not installed.\n"
                "  pip install mlx mlx-lm\n"
                "Requires Apple Silicon."
            )
        self.gc_sleep = gc_sleep
        self.aggressive_cleanup = aggressive_cleanup

        self.model = None
        self.tokenizer = None
        self.config: Optional[ModelConfig] = None
        self._currently_loaded: Optional[str] = None
        self._total_swaps: int = 0

        logger.info("[MLX] Initialized. Single-slot, Apple Silicon.")

    def load(self, config: ModelConfig) -> None:
        if self.model is not None:
            self.offload()
        logger.info("[MLX] Loading: %s (est. %.1f GB)", config.path, config.ram_estimate_gb)
        try:
            self.model, self.tokenizer = mlx_load(config.path)
            self.config = config
            self._currently_loaded = config.path
            self._total_swaps += 1
            logger.info("[MLX] Loaded: %s", config.path)
        except Exception as e:
            logger.critical("[MLX] Load failed: %s | %s", config.path, e)
            self._emergency_cleanup()
            raise

    def offload(self) -> None:
        if self.model is None:
            return
        name = self.config.path if self.config else "unknown"
        logger.info("[MLX] Offloading: %s", name)

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.config = None
        self._currently_loaded = None

        for _ in range(3):
            gc.collect()

        try:
            mx.synchronize()
        except Exception as e:
            logger.warning("[MLX] mx.synchronize() warning: %s", e)

        if self.aggressive_cleanup:
            try:
                if hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
                    logger.info("[MLX] Metal cache cleared.")
                elif hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception as e:
                logger.debug("[MLX] Cache clear note: %s", e)

        logger.info("[MLX] Sleeping %.1f s for memory settlement...", self.gc_sleep)
        time.sleep(self.gc_sleep)
        logger.info("[MLX] Offload complete.")

    def generate(self, prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        if self.model is None:
            raise RuntimeError("[MLX] No model loaded. Call load() first.")
        tokens = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature
        logger.info("[MLX] Generating: max_tokens=%d, temp=%.2f", tokens, temp)
        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=tokens,
            temp=temp,
            verbose=False,
        )

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup — best-effort memory release when normal offload fails."""
        logger.critical("[MLX] Emergency cleanup!")
        self.model = None
        self.tokenizer = None
        self.config = None
        self._currently_loaded = None
        gc.collect()
        if MLX_AVAILABLE:
            try:
                mx.synchronize()
            except Exception:
                pass
            try:
                if hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
        time.sleep(5)


# ─────────────────────────────────────────────────────────────────
# Ollama Backend — Any system
# ─────────────────────────────────────────────────────────────────
class OllamaModelManager(BaseModelManager):
    """
    Ollama backend. Works on any system with Ollama installed.
    https://ollama.com

    Memory management:
      Ollama manages its own memory pool.
      offload() sends keep_alive=0 which forces the model out of RAM/VRAM.
      generate() sends keep_alive=-1 (keep in memory between calls).

    Tip: run `ollama ps` to see what's currently loaded.
    Tip: run `ollama serve` to start the server if it's not running.
    """

    def __init__(self, host: str = "http://localhost:11434", gc_sleep: float = 2.0):
        self.host = host.rstrip("/")
        self.gc_sleep = gc_sleep
        self.config: Optional[ModelConfig] = None
        self._currently_loaded: Optional[str] = None
        self._total_swaps: int = 0
        logger.info("[Ollama] Initialized. Host: %s", self.host)
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify Ollama is reachable and log available models."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info("[Ollama] Connected. Available models: %s", models)
        except Exception as e:
            logger.warning(
                "[Ollama] Cannot connect to Ollama at %s: %s\n"
                "         Make sure Ollama is running: ollama serve",
                self.host, e,
            )

    def load(self, config: ModelConfig) -> None:
        if self._currently_loaded and self._currently_loaded != config.path:
            self.offload()
        logger.info("[Ollama] Active model set: %s", config.path)
        self.config = config
        self._currently_loaded = config.path
        self._total_swaps += 1

    def offload(self) -> None:
        if not self._currently_loaded:
            return
        logger.info("[Ollama] Unloading: %s", self._currently_loaded)
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self._currently_loaded, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except Exception as e:
            logger.warning("[Ollama] Offload request failed: %s", e)
        self._currently_loaded = None
        self.config = None
        time.sleep(self.gc_sleep)
        logger.info("[Ollama] Offload complete.")

    def generate(self, prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        if not self._currently_loaded:
            raise RuntimeError("[Ollama] No model set. Call load() first.")
        tokens = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature
        logger.info(
            "[Ollama] Generating: model=%s, max_tokens=%d, temp=%.2f",
            self._currently_loaded, tokens, temp,
        )
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self._currently_loaded,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": temp,
                        "num_predict": tokens,
                    },
                },
                timeout=600,
            )
            r.raise_for_status()
            return r.json()["response"]
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "[Ollama] Generation timed out after 600 s. "
                "Model may be too large or system too slow."
            )
        except Exception as e:
            raise RuntimeError(f"[Ollama] Generation failed: {e}")


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────
def create_model_manager(backend: str, **kwargs) -> BaseModelManager:
    """
    Create a model manager for the specified backend.

    Args:
        backend: "mlx" or "ollama"
        **kwargs:
            mlx:   gc_sleep (float, default 2.0)
                   aggressive_cleanup (bool, default True)
            ollama: host (str, default "http://localhost:11434")
                    gc_sleep (float, default 2.0)
    """
    if backend == "mlx":
        return MLXModelManager(
            gc_sleep=kwargs.get("gc_sleep", 2.0),
            aggressive_cleanup=kwargs.get("aggressive_cleanup", True),
        )
    elif backend == "ollama":
        return OllamaModelManager(
            host=kwargs.get("host", "http://localhost:11434"),
            gc_sleep=kwargs.get("gc_sleep", 2.0),
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Valid options: 'mlx', 'ollama'.")
