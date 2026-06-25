#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: core/orchestrator.py                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Port Orchestrator — wires together ModelManager + Coordinator.
#   Drives the full execution loop. Zero reasoning responsibility.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Orchestrator

Wires together: ModelManager + Coordinator.
Drives the full execution loop.

The orchestrator has zero reasoning responsibility.
All intelligence is in the Brain/Specialist models.
All state logic is in the Coordinator.
The orchestrator only sequences calls.
"""

import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

from core.model_manager import ModelConfig, create_model_manager, BaseModelManager
from core.coordinator import Coordinator

logger = logging.getLogger(__name__)


def _escape_tg_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters in user-provided text."""
    if not text:
        return text
    for char in ["_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}"]:
        text = text.replace(char, f"\\{char}")
    return text


class PortOrchestrator:
    """
    Port Orchestrator

    Wires ModelManager and Coordinator together.
    Handles graceful shutdown, resume, and Telegram notifications.

    Usage:
        orch = PortOrchestrator(config_path="config.yaml")
        orch.run("Build a fitness app")
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.backend: str = self.config.get("backend", "mlx")

        mem_cfg = self.config.get("memory", {})
        if self.backend == "mlx":
            self.model_mgr: BaseModelManager = create_model_manager(
                "mlx",
                gc_sleep=mem_cfg.get("gc_sleep_seconds", 2.0),
                aggressive_cleanup=mem_cfg.get("aggressive_cleanup", True),
            )
        elif self.backend == "ollama":
            self.model_mgr = create_model_manager(
                "ollama",
                host=self.config.get("ollama_host", "http://localhost:11434"),
                gc_sleep=mem_cfg.get("gc_sleep_seconds", 2.0),
            )
        else:
            raise ValueError(f"Unknown backend: '{self.backend}'. Use 'mlx' or 'ollama'.")

        proj = self.config["project"]
        self.coord = Coordinator(
            outputs_dir=proj["outputs_dir"],
            memory_file=proj["memory_file"],
            state_file=proj["state_file"],
        )

        brain_key = f"brain_{self.backend}"
        specialists_key = f"specialists_{self.backend}"

        if brain_key not in self.config:
            raise KeyError(f"Config missing section: '{brain_key}'")
        if specialists_key not in self.config:
            raise KeyError(f"Config missing section: '{specialists_key}'")

        self.brain_config: ModelConfig = self._make_config("brain", self.config[brain_key])
        self.specialist_configs: Dict[str, ModelConfig] = {
            k: self._make_config("specialist", v, k)
            for k, v in self.config[specialists_key].items()
        }

        self._telegram = None
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("[Orchestrator] Port ready. Backend: %s", self.backend)

    def _signal_handler(self, signum, frame) -> None:
        """Handle Ctrl+C / SIGTERM by unloading model and exiting cleanly."""
        sig_name = signal.Signals(signum).name
        logger.warning("[Orchestrator] Received %s — initiating graceful shutdown...", sig_name)
        self._shutdown_requested = True
        try:
            self.model_mgr.shutdown()
            logger.info("[Orchestrator] Model unloaded successfully.")
        except Exception as e:
            logger.warning("[Orchestrator] Shutdown cleanup error: %s", e)
        self._notify(f"⚠️ Port stopped ({sig_name}). Model unloaded.")
        os._exit(128 + signum)

    def run(self, goal: str, constraints: str = "", resume: bool = False) -> None:
        safe_goal = _escape_tg_markdown(goal)
        self._notify(f"🐳 Port Started\nBackend: {self.backend}\nGoal: _{safe_goal}_")

        if resume:
            state = self.coord.load_state()
            if state:
                logger.info("[Orchestrator] Resuming from checkpoint...")
                self._notify("📂 Resuming from checkpoint...")
                self._resume_execution(state)
            else:
                logger.warning("[Orchestrator] No checkpoint found. Starting fresh.")
                self._start_new_project(goal, constraints)
        else:
            self._start_new_project(goal, constraints)

    def _start_new_project(self, goal: str, constraints: str) -> None:
        logger.info("[Orchestrator] New project: %s", goal)
        self._notify("📝 Brain creating master plan...")

        task_names, first_specialist, first_subtasks = self._brain_first_load(goal, constraints)

        if not task_names:
            raise RuntimeError("Brain failed to produce a task list.")

        logger.info("[Orchestrator] %d tasks created.", len(task_names))
        self._notify(f"📋 Master plan: *{len(task_names)}* tasks created.")

        state = self.coord.init_state(
            self.config["project"]["name"],
            goal,
            task_names,
            constraints=constraints,
        )

        self.coord.init_memory(
            goal=goal,
            constraints=constraints,
            task_names=task_names,
            first_specialist=first_specialist,
            first_subtasks=first_subtasks,
        )

        state["status"] = "executing"
        self.coord.save_state(state)
        self._execute_task_loop(state)

    def _execute_task_loop(self, state: Dict) -> None:
        """Main loop. Runs until all tasks complete or shutdown is requested."""
        while True:
            if self._shutdown_requested:
                logger.info("[Orchestrator] Shutdown requested — exiting task loop.")
                break
            next_task = self.coord.get_next_pending_task(state)
            if not next_task:
                self._do_final_synthesis(state)
                break
            self._execute_single_task(state, next_task)

    def _execute_single_task(self, state: Dict, task: Dict) -> None:
        """Full cycle for one task: Specialist executes, Brain reviews."""
        task_id = task["id"]
        task_name = task["name"]

        mem_info = self.coord.get_current_task_info()
        specialist_key = mem_info.get("specialist", "coder").lower()
        subtasks = mem_info.get("subtasks", [])

        if specialist_key not in self.specialist_configs:
            logger.warning(
                "[Orchestrator] Unknown specialist '%s' — falling back to 'coder'.", specialist_key
            )
            specialist_key = "coder"

        logger.info("=" * 60)
        logger.info("[Orchestrator] Task %d/%d: %s | %s", task_id, state["total_tasks"], task_name, specialist_key)
        logger.info("=" * 60)

        self.coord.mark_task_active(state, task_id, specialist_key)
        safe_task_name = _escape_tg_markdown(task_name)
        self._notify(
            f"⚙️ *Task {task_id}/{state['total_tasks']}*\n"
            f"{safe_task_name}\n"
            f"Specialist: `{specialist_key}`"
        )

        # PHASE A: Specialist executes
        specialist_config = self.specialist_configs[specialist_key]
        self.model_mgr.load(specialist_config)

        prompt = self.coord.build_specialist_prompt(task_id, task_name, subtasks, state["goal"])
        try:
            output = self.model_mgr.generate(
                prompt=prompt,
                max_tokens=specialist_config.max_tokens,
                temperature=specialist_config.temperature,
            )
        except Exception as e:
            logger.error("[Orchestrator] Specialist generation failed: %s", e)
            output = f"ERROR: Specialist generation failed.\n\n{e}"

        output_file = self.coord.write_task_output(task_id, task_name, specialist_key, subtasks, output)
        self.coord.mark_task_done(state, task_id, output_file)
        self.model_mgr.offload()

        # PHASE B: Brain reviews and plans next
        self._notify(f"🧠 Brain reviewing Task {task_id}...")

        self.model_mgr.load(self.brain_config)

        memory_content = self.coord.read_memory()
        output_content = Path(output_file).read_text()

        brain_prompt = self.coord.build_brain_review_prompt(
            task_id, task_name, memory_content, output_content, state["total_tasks"]
        )

        brain_cfg = self.config[f"brain_{self.backend}"]
        try:
            brain_output = self.model_mgr.generate(
                prompt=brain_prompt,
                max_tokens=brain_cfg["max_tokens_subsequent"],
                temperature=brain_cfg["temperature"],
            )
        except Exception as e:
            logger.error("[Orchestrator] Brain review failed: %s", e)
            brain_output = (
                "SUMMARY: Task completed.\n"
                "NEXT_SPECIALIST: coder\n"
                "NEXT_SUBTASKS:\n1. Continue\nADAPTATIONS: None"
            )

        summary, next_specialist, next_subtasks, is_final = self.coord.parse_brain_review_output(
            brain_output, task_id, state["total_tasks"]
        )

        has_next = (task_id < state["total_tasks"]) and not is_final
        self.coord.update_memory_after_task(
            task_num=task_id,
            summary=summary,
            next_specialist=next_specialist if has_next else None,
            next_subtasks=next_subtasks if has_next else None,
        )

        if not self.coord.verify_memory_integrity():
            self._notify("🚨 CRITICAL: memory.md integrity check FAILED. Halting.")
            raise RuntimeError("Memory integrity check failed.")

        self.model_mgr.offload()
        safe_summary = _escape_tg_markdown(summary[:100])
        self._notify(f"✅ Task {task_id} done. {safe_summary}")

    def _do_final_synthesis(self, state: Dict) -> None:
        """Final brain load: synthesize all outputs into FINAL_ANSWER.md."""
        logger.info("[Orchestrator] All tasks complete. Final synthesis...")
        state["status"] = "complete"
        self.coord.save_state(state)
        self._notify("🔍 *Final Synthesis* — Brain compiling complete answer...")

        self.model_mgr.load(self.brain_config)

        memory_content = self.coord.read_memory()
        all_outputs = self.coord.get_all_outputs(state)
        final_prompt = self.coord.build_final_prompt(memory_content, all_outputs)

        brain_cfg = self.config[f"brain_{self.backend}"]
        final_answer = self.model_mgr.generate(
            prompt=final_prompt,
            max_tokens=brain_cfg["max_tokens_final"],
            temperature=0.5,
        )

        final_file = self.coord.write_final_answer(final_answer)
        self.model_mgr.shutdown()

        safe_goal = _escape_tg_markdown(state["goal"])
        self._notify(
            f"🎉 *PROJECT COMPLETE!*\n\n"
            f"Goal: _{safe_goal}_\n"
            f"Tasks completed: {state['total_tasks']}"
        )
        logger.info("[Orchestrator] Complete. Final answer: %s", final_file)

    def _brain_first_load(self, goal: str, constraints: str):
        """Load brain, generate task list + Task 1 subtasks, unload."""
        self.model_mgr.load(self.brain_config)

        prompt = self.coord.build_brain_first_prompt(
            goal, constraints, list(self.specialist_configs.keys())
        )

        brain_cfg = self.config[f"brain_{self.backend}"]
        output = self.model_mgr.generate(
            prompt=prompt,
            max_tokens=brain_cfg["max_tokens_first_load"],
            temperature=brain_cfg["temperature"],
        )

        result = self.coord.parse_brain_first_output(output)
        self.model_mgr.offload()
        return result

    def _resume_execution(self, state: Dict) -> None:
        """Resume from state.json + memory.md checkpoint."""
        if not self.coord.verify_memory_integrity():
            raise RuntimeError("Cannot resume — memory.md is corrupt or missing.")
        constraints = state.get("constraints", "")
        if constraints:
            logger.info("[Orchestrator] Restored constraints: %s", constraints)
        self._execute_task_loop(state)

    def _make_config(self, role: str, cfg: dict, key: str = "") -> ModelConfig:
        return ModelConfig(
            path=cfg["model_path"],
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=cfg.get("temperature", 0.7),
            description=cfg.get("description", key),
            ram_estimate_gb=cfg.get("ram_estimate_gb", 0.0),
            role=role,
        )

    def _notify(self, message: str) -> None:
        """Send Telegram notification. Silently skips if not configured."""
        tg = self.config.get("telegram", {})
        if not tg.get("token") or not tg.get("chat_id"):
            return

        if self._telegram is None:
            try:
                from utils.telegram_notify import TelegramNotifier
                self._telegram = TelegramNotifier(tg["token"], tg["chat_id"])
            except Exception as e:
                logger.debug("[Orchestrator] Telegram init failed: %s", e)
                return

        try:
            self._telegram.send(message)
        except Exception as e:
            logger.debug("[Orchestrator] Telegram send failed: %s", e)
