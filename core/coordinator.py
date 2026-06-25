#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: core/coordinator.py                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Coordinator — Pure Python state machine. Zero LLM reasoning.
#   Handles all routing, state, memory, parsing, and crash recovery.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Coordinator (Pure Python State Machine)

The coordinator is 100% deterministic Python. Zero LLM reasoning.

Responsibilities:
  1. Load / unload models (delegates to model manager)
  2. Inject memory.md as first message on every brain reload
  3. Write and maintain state.json
  4. Write specialist outputs to outputs/task_N.md
  5. Atomic memory writes (memory.md.tmp → rename → memory.md)
  6. Parse brain outputs deterministically (regex)
  7. Verify memory.md integrity after every brain unload
  8. Route tasks to correct specialist based on brain assignment
  9. Handle crash recovery

No LLM. No creativity. No ambiguity. Pure code.
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Reasoning tag patterns to strip from model outputs
_REASONING_PATTERNS = [
    (r"<think>.*?</think>", re.DOTALL),
    (r"<thinking>.*?</thinking>", re.DOTALL),
    (r"<thought>.*?</thought>", re.DOTALL),
    (r"<reasoning>.*?</reasoning>", re.DOTALL),
    (r"\[Thinking:.*?\]", re.DOTALL),
]


def _strip_all_reasoning(text: str) -> str:
    """Strip all known reasoning/thinking tags from model output."""
    for pattern, flags in _REASONING_PATTERNS:
        text = re.sub(pattern, "", text, flags=flags)
    return text.strip()


class Coordinator:
    """
    Port Coordinator — Pure Python State Machine

    100% deterministic. Zero LLM reasoning. Handles all routing,
    state management, memory, parsing, and crash recovery.

    Usage:
        coord = Coordinator(outputs_dir="./outputs")
        state = coord.init_state("my_project", "Build an app", task_names)
    """

    def __init__(
        self,
        outputs_dir: str = "./outputs",
        memory_file: str = "./memory.md",
        state_file: str = "./state.json",
    ):
        self.outputs_dir = Path(outputs_dir)
        self.memory_file = Path(memory_file)
        self.tmp_file = self.memory_file.with_suffix(".md.tmp")
        self.state_file = Path(state_file)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── State Management ─────────────────────────────────────────

    def init_state(self, project_name: str, goal: str, task_names: List[str],
                   constraints: str = "") -> Dict:
        """Create and save initial state.json."""
        state = {
            "project_name": project_name,
            "goal": goal,
            "constraints": constraints,
            "current_task_index": 1,
            "total_tasks": len(task_names),
            "status": "planning",
            "tasks": [
                {"id": i + 1, "name": name, "status": "pending",
                 "specialist": None, "output_file": None}
                for i, name in enumerate(task_names)
            ],
        }
        self.save_state(state)
        logger.info("[Coordinator] State initialized: %d tasks", len(task_names))
        return state

    def load_state(self) -> Optional[Dict]:
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("[Coordinator] Failed to load state.json: %s", e)
            return None

    def save_state(self, state: Dict) -> None:
        state["last_checkpoint"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def mark_task_active(self, state: Dict, task_id: int, specialist: str) -> None:
        for task in state["tasks"]:
            if task["id"] == task_id:
                task["status"] = "active"
                task["specialist"] = specialist
                break
        state["current_task_index"] = task_id
        self.save_state(state)

    def mark_task_done(self, state: Dict, task_id: int, output_file: str) -> None:
        for task in state["tasks"]:
            if task["id"] == task_id:
                task["status"] = "done"
                task["output_file"] = output_file
                break
        self.save_state(state)

    def get_next_pending_task(self, state: Dict) -> Optional[Dict]:
        for task in state["tasks"]:
            if task["status"] == "pending":
                return task
        return None

    def get_all_outputs(self, state: Dict) -> List[Dict]:
        outputs = []
        for task in state["tasks"]:
            if task.get("output_file"):
                p = Path(task["output_file"])
                if p.exists():
                    outputs.append({"id": task["id"], "name": task["name"],
                                    "content": p.read_text()[:3000]})
        return outputs

    # ── Memory Management ────────────────────────────────────────

    def init_memory(self, goal: str, constraints: str, task_names: List[str],
                    first_specialist: str, first_subtasks: List[str]) -> None:
        lines = [f"# Port State · Task 1 of {len(task_names)}", ""]
        lines += ["## LOCKED — Never modify", f"Goal: {goal}",
                  f"Hard constraints: {constraints if constraints else 'None specified'}", ""]
        lines += ["## Task List"]
        for i, name in enumerate(task_names, 1):
            status = "x" if i == 1 else " "
            current = " ← CURRENT" if i == 1 else ""
            lines.append(f"- [{status}] Task {i}: {name}{current}")
        lines += ["", "## Current Task", "Task: 1",
                  f"Specialist: {first_specialist}", "Subtasks:"]
        for i, st in enumerate(first_subtasks, 1):
            lines.append(f"  {i}. {st}")
        lines += ["", "## ⚠ MANDATORY LAST ACTION BEFORE UNLOAD",
                  "Delete current subtask block. Append one summary line to completed task. "
                  "Write next task subtasks. Update this file.", ""]
        self._atomic_write("\n".join(lines))
        logger.info("[Coordinator] memory.md initialized: %d tasks", len(task_names))

    def read_memory(self) -> str:
        if not self.memory_file.exists():
            return ""
        with open(self.memory_file, "r") as f:
            return f.read()

    def update_memory_after_task(self, task_num: int, summary: str,
                                  next_specialist: Optional[str] = None,
                                  next_subtasks: Optional[List[str]] = None) -> None:
        current = self.read_memory()
        lines = current.split("\n")
        new_lines = []
        in_task_list = False

        for line in lines:
            if line.strip() == "## Task List":
                in_task_list = True
                new_lines.append(line)
                continue
            if in_task_list and line.startswith("## "):
                in_task_list = False

            if in_task_list:
                done_match = re.match(
                    r"^(\s*- \[)(.)\] Task " + str(task_num) + r": (.+?)(?: — .+)?(\s*← CURRENT)?$",
                    line,
                )
                if done_match:
                    new_lines.append(f"{done_match.group(1)}x] Task {task_num}: {done_match.group(3)} — {summary}")
                    continue
                if next_subtasks:
                    next_match = re.match(
                        r"^(\s*- \[)(.)\] Task " + str(task_num + 1) + r": (.+?)(\s*← CURRENT)?$",
                        line,
                    )
                    if next_match:
                        new_lines.append(f"{next_match.group(1)} ] Task {task_num + 1}: {next_match.group(3)} ← CURRENT")
                        continue
            new_lines.append(line)

        content = "\n".join(new_lines)
        if next_subtasks and next_specialist:
            new_section = f"Task: {task_num + 1}\nSpecialist: {next_specialist}\nSubtasks:\n"
            new_section += "".join(f"  {i}. {st}\n" for i, st in enumerate(next_subtasks, 1))
            content = re.sub(r"(## Current Task\n).+?(?=\n## ⚠|$)", r"\1" + new_section,
                             content, flags=re.DOTALL)
            total = len(re.findall(r"^\s*- \[", content, re.MULTILINE))
            content = re.sub(r"# Port State · Task \d+ of \d+",
                             f"# Port State · Task {task_num + 1} of {total}", content)
        else:
            content = re.sub(r"## Current Task\n.+?(?=\n## ⚠|$)", "", content, flags=re.DOTALL)
            content = re.sub(r"# Port State · Task \d+ of \d+", "# Port State · COMPLETE", content)

        self._atomic_write(content)
        logger.info("[Coordinator] memory.md updated after Task %d", task_num)

    def get_current_task_info(self) -> Dict:
        content = self.read_memory()
        info: Dict = {"task_num": None, "specialist": None, "subtasks": [], "total_tasks": 0}
        m = re.search(r"Task (\d+) of (\d+)", content)
        if m:
            info["task_num"] = int(m.group(1))
            info["total_tasks"] = int(m.group(2))
        m = re.search(r"Specialist:\s*(.+?)(?:\n|$)", content)
        if m:
            info["specialist"] = m.group(1).strip()
        m = re.search(r"Subtasks:\n((?:\s+\d+\..*\n?)+)", content)
        if m:
            raw = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|\n##|$)", m.group(1), re.DOTALL)
            info["subtasks"] = [s.strip() for s in raw if s.strip()]
        return info

    def verify_memory_integrity(self) -> bool:
        if not self.memory_file.exists():
            return False
        content = self.read_memory()
        required = ["## LOCKED", "## Task List", "## ⚠ MANDATORY"]
        ok = all(section in content for section in required)
        if not ok:
            missing = [s for s in required if s not in content]
            logger.critical("[Coordinator] memory.md missing sections: %s", missing)
        return ok

    def _atomic_write(self, content: str) -> None:
        try:
            with open(self.tmp_file, "w") as f:
                f.write(content)
            os.replace(self.tmp_file, self.memory_file)
        except Exception as e:
            logger.error("[Coordinator] Atomic write failed: %s", e)
            raise

    # ── Output Management ────────────────────────────────────────

    def write_task_output(self, task_id: int, task_name: str, specialist: str,
                          subtasks: List[str], output: str) -> str:
        out_file = self.outputs_dir / f"task_{task_id:03d}_{specialist}.md"
        with open(out_file, "w") as f:
            f.write(f"# Task {task_id}: {task_name}\n\n")
            f.write(f"**Specialist:** {specialist}\n\n")
            f.write("**Subtasks executed:**\n")
            for i, st in enumerate(subtasks, 1):
                f.write(f"{i}. {st}\n")
            f.write(f"\n---\n\n{output}\n")
        logger.info("[Coordinator] Output written: %s", out_file)
        return str(out_file)

    def write_final_answer(self, answer: str) -> str:
        final_file = self.outputs_dir / "FINAL_ANSWER.md"
        with open(final_file, "w") as f:
            f.write(f"# Final Answer\n\n{answer}\n")
        logger.info("[Coordinator] Final answer written: %s", final_file)
        return str(final_file)

    # ── Brain Output Parsing ─────────────────────────────────────

    def parse_brain_first_output(self, output: str) -> Tuple[List[str], str, List[str]]:
        """Parse brain's first-load output. Returns: (task_names, first_specialist, first_subtasks)"""
        output = _strip_all_reasoning(output)
        task_names: List[str] = []
        first_specialist = "coder"
        first_subtasks: List[str] = []

        m = re.search(r"TASK_LIST:\n(.+?)(?=TASK_1_SPECIALIST|$)", output, re.DOTALL)
        if m:
            for line in m.group(1).strip().splitlines():
                lm = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if lm:
                    task_names.append(lm.group(1).strip())

        m = re.search(r"TASK_1_SPECIALIST:\s*(\w+)", output, re.IGNORECASE)
        if m:
            first_specialist = m.group(1).strip().lower()

        m = re.search(r"TASK_1_SUBTASKS:\n(.+?)(?=\n\n|$)", output, re.DOTALL | re.IGNORECASE)
        if m:
            for line in m.group(1).strip().splitlines():
                lm = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if lm:
                    first_subtasks.append(lm.group(1).strip())

        if not task_names:
            logger.warning("[Coordinator] Brain produced no task list. Using fallback.")
            task_names = ["Analyze requirements", "Design architecture",
                          "Implement core", "Add features", "Test and finalize"]
        if not first_subtasks:
            first_subtasks = ["Understand the goal", "Plan the approach", "Produce output"]

        return task_names, first_specialist, first_subtasks

    def parse_brain_review_output(self, output: str, task_id: int,
                                   total_tasks: int) -> Tuple[str, str, List[str], bool]:
        """Parse brain's between-task review output."""
        output = _strip_all_reasoning(output)
        summary = "Task completed."
        next_specialist = "coder"
        next_subtasks: List[str] = []
        is_final = task_id >= total_tasks

        if "PROJECT_COMPLETE" in output:
            is_final = True

        m = re.search(r"SUMMARY:\s*(.+?)(?=\n|$)", output, re.IGNORECASE)
        if m:
            summary = m.group(1).strip()

        m = re.search(r"NEXT_SPECIALIST:\s*(\w+)", output, re.IGNORECASE)
        if m:
            next_specialist = m.group(1).strip().lower()

        m = re.search(r"NEXT_SUBTASKS:\n(.+?)(?=ADAPTATIONS|\n\n|$)",
                       output, re.DOTALL | re.IGNORECASE)
        if m:
            for line in m.group(1).strip().splitlines():
                lm = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if lm:
                    next_subtasks.append(lm.group(1).strip())

        if not next_subtasks and not is_final:
            next_subtasks = ["Continue from previous task", "Build on prior output"]

        return summary, next_specialist, next_subtasks, is_final

    # ── Prompt Building ──────────────────────────────────────────

    def build_brain_first_prompt(self, goal: str, constraints: str,
                                  available_specialists: List[str]) -> str:
        specialists_text = "\n".join(f"- {s}" for s in available_specialists)
        return f"""You are the Brain. This is your FIRST load. Your KV cache is empty.

## User Goal
{goal}

## Hard Constraints
{constraints if constraints else "None specified"}

## Available Specialists
{specialists_text}

## Your Job
1. Create a complete task list (task NAMES only — no subtasks yet).
2. Assign the correct specialist to Task 1.
3. Write detailed, actionable subtasks for Task 1 ONLY.

## Output Format
```
TASK_LIST:
1. Task name here
2. Task name here
...

TASK_1_SPECIALIST: <specialist_name>

TASK_1_SUBTASKS:
1. First subtask
2. Second subtask
3. Third subtask
```
"""

    def build_specialist_prompt(self, task_id: int, task_name: str,
                                 subtasks: List[str], goal: str) -> str:
        subtasks_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(subtasks))
        return f"""You are a specialist AI executing a specific task in a larger project.

## Project Goal
{goal}

## Your Task
Task {task_id}: {task_name}

## Subtasks to Complete
{subtasks_text}

## Rules
- Execute every subtask thoroughly.
- Produce complete, production-quality output.
- Use markdown formatting.

## Output
Write your complete response below.
"""

    def build_final_prompt(self, memory_content: str, all_outputs: List[Dict]) -> str:
        outputs_text = "".join(
            f"\n### Task {o['id']}: {o['name']}\n{o['content']}\n" for o in all_outputs)
        return f"""You are the Brain doing FINAL SYNTHESIS.

## Full Project Memory
{memory_content}

## All Task Outputs
{outputs_text}

## Your Job
Synthesize all outputs into a coherent, complete final deliverable.
"""
