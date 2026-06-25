#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: main.py                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Entry point for the Port multi-backend agentic AI orchestrator.
#   Supports both MLX (Apple Silicon) and Ollama (any system) backends.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# HARDWARE TARGET: MacBook Pro M1 Max 32GB (MLX) or any system with Ollama
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Main Entry Point

Usage:
    python main.py "Build a React Native fitness app with AI meal planner"
    python main.py "My idea" --constraints "Must use TypeScript"
    python main.py --resume
    python main.py --list-specialists
    python main.py --config path/to/config.yaml "My goal"

Supports: MLX (Apple Silicon) and Ollama (any system)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import PortOrchestrator


def setup_logging() -> Path:
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    import datetime
    log_file = logs_dir / f"port_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Port — Portable Agentic AI")
    parser.add_argument("goal", nargs="?", help="Project goal / idea")
    parser.add_argument("--constraints", default="", help="Hard constraints (e.g. 'Must use TypeScript')")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--list-specialists", action="store_true", help="List available specialists and exit")

    args = parser.parse_args()
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    if not Path(args.config).exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("PORT — Portable Agentic AI")
    logger.info("Config: %s", args.config)
    logger.info("=" * 70)

    orchestrator = PortOrchestrator(config_path=args.config)

    if args.list_specialists:
        print("\nAvailable Specialists:")
        print("-" * 50)
        for key, cfg in orchestrator.specialist_configs.items():
            print(f"  {key:15s} — {cfg.description}")
            print(f"    Model : {cfg.path}")
            if cfg.ram_estimate_gb:
                print(f"    RAM   : ~{cfg.ram_estimate_gb} GB")
            print()
        print(f"Brain: {orchestrator.brain_config.path}")
        sys.exit(0)

    if args.resume:
        logger.info("Resuming from checkpoint...")
        orchestrator.run(goal="", resume=True)
    elif args.goal:
        logger.info("Starting: %s", args.goal)
        orchestrator.run(goal=args.goal, constraints=args.constraints)
    else:
        print("\n🐳 Port — Portable Agentic AI")
        print("=" * 50)
        print("What do you want to build or research?")
        print("Example: 'A SaaS dashboard with real-time analytics'")
        print()
        goal = input("> ").strip()
        if not goal:
            print("No goal provided. Exiting.")
            sys.exit(0)
        print("\nAny hard constraints? (press Enter for none)")
        constraints = input("> ").strip()
        orchestrator.run(goal=goal, constraints=constraints)

    logger.info("Done. Log saved: %s", log_file)
    print(f"\nComplete. Log: {log_file}")


if __name__ == "__main__":
    main()
