"""
Coordinator — Pure Python State Machine

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
# Covers Qwen3 <think>, DeepSeek-R1 <thinking>, and other formats
_REASONING_PATTERNS = [
    (r"<think>.*?</think>", re.DOTALL),           # Qwen3
    (r"<thinking>.*?</thinking>", re.DOTALL),      # DeepSeek-R1 and variants
    (r"<thought>.*?</thought>", re.DOTALL),        # Some model variants
    (r"<reasoning>.*?</reasoning>", re.DOTALL),    # Generic reasoning blocks
    (r"\[Thinking:.*?\]", re.DOTALL),              # Bracket-style reasoning
]


def _strip_all_reasoning(text: str) -> str:
    """Strip all known reasoning/thinking tags from model output."""
    for pattern, flags in _REASONING_PATTERNS:
        text = re.sub(pattern, "", text, flags=flags)
    return text.strip()


class Coordinator:

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

    # ════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT — state.json
    # ════════════════════════════════════════════════════════════════

    def init_state(self, project_name: str, goal: str, task_names: List[str],
                   constraints: str = "") -> Dict:
        """Create and save initial state.json. Returns the state dict.
        
        Args:
            constraints: Hard constraints (stored for resume support).
        """
        state = {
            "project_name": project_name,
            "goal": goal,
            "constraints": constraints,
            "current_task_index": 1,
            "total_tasks": len(task_names),
            "status": "planning",
            "loaded_model": None,
            "last_checkpoint": datetime.now().isoformat(),
            "tasks": [
                {
                    "id": i + 1,
                    "name": name,
                    "status": "pending",
                    "specialist": None,
                    "output_file": None,
                }
                for i, name in enumerate(task_names)
            ],
        }
        self.save_state(state)
        logger.info("[Coordinator] State initialized: %d tasks", len(task_names))
        return state

    def load_state(self) -> Optional[Dict]:
        """Load state.json from disk. Returns None if not found or corrupt."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("[Coordinator] Failed to load state.json: %s", e)
            return None

    def save_state(self, state: Dict) -> None:
        """Save state dict to state.json. Always updates last_checkpoint."""
        state["last_checkpoint"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def mark_task_active(self, state: Dict, task_id: int, specialist: str) -> None:
        """Set task status to 'active' and save checkpoint."""
        for task in state["tasks"]:
            if task["id"] == task_id:
                task["status"] = "active"
                task["specialist"] = specialist
                break
        state["current_task_index"] = task_id
        state["loaded_model"] = specialist
        self.save_state(state)

    def mark_task_done(self, state: Dict, task_id: int, output_file: str) -> None:
        """Set task status to 'done', record output file, save checkpoint."""
        for task in state["tasks"]:
            if task["id"] == task_id:
                task["status"] = "done"
                task["output_file"] = output_file
                break
        state["loaded_model"] = None
        self.save_state(state)

    def get_next_pending_task(self, state: Dict) -> Optional[Dict]:
        """Return the first task with status 'pending', or None if all done."""
        for task in state["tasks"]:
            if task["status"] == "pending":
                return task
        return None

    def all_tasks_done(self, state: Dict) -> bool:
        return all(t["status"] == "done" for t in state["tasks"])

    def get_task_output_path(self, state: Dict, task_id: int) -> Optional[str]:
        for task in state["tasks"]:
            if task["id"] == task_id and task.get("output_file"):
                return task["output_file"]
        return None

    def get_all_outputs(self, state: Dict) -> List[Dict]:
        """Return all completed task outputs (first 3000 chars each)."""
        outputs = []
        for task in state["tasks"]:
            if task.get("output_file"):
                p = Path(task["output_file"])
                if p.exists():
                    outputs.append({
                        "id": task["id"],
                        "name": task["name"],
                        "content": p.read_text()[:3000],
                    })
        return outputs

    # ════════════════════════════════════════════════════════════════
    # MEMORY MANAGEMENT — memory.md
    # ════════════════════════════════════════════════════════════════

    def init_memory(
        self,
        goal: str,
        constraints: str,
        task_names: List[str],
        first_specialist: str,
        first_subtasks: List[str],
    ) -> None:
        """Write initial memory.md from brain's first-load output."""
        lines = [f"# Brain State · Task 1 of {len(task_names)}", ""]
        lines += ["## LOCKED — Never modify", f"Goal: {goal}",
                  f"Hard constraints: {constraints if constraints else 'None specified'}", ""]
        lines += ["## Task List"]
        for i, name in enumerate(task_names, 1):
            status = "x" if i == 1 else " "
            current = " ← CURRENT" if i == 1 else ""
            lines.append(f"- [{status}] Task {i}: {name}{current}")
        lines += [
            "",
            "## Current Task",
            "Task: 1",
            f"Specialist: {first_specialist}",
            "Subtasks:",
        ]
        for i, st in enumerate(first_subtasks, 1):
            lines.append(f"  {i}. {st}")
        lines += [
            "",
            "## ⚠ MANDATORY LAST ACTION BEFORE UNLOAD",
            "Delete current subtask block. Append one summary line to completed task. "
            "Write next task subtasks. Update this file.",
            "",
        ]
        self._atomic_write("\n".join(lines))
        logger.info("[Coordinator] memory.md initialized: %d tasks", len(task_names))

    def read_memory(self) -> str:
        if not self.memory_file.exists():
            return ""
        with open(self.memory_file, "r") as f:
            return f.read()

    def update_memory_after_task(
        self,
        task_num: int,
        summary: str,
        next_specialist: Optional[str] = None,
        next_subtasks: Optional[List[str]] = None,
    ) -> None:
        """
        Update memory.md after a task completes.
        - Mark completed task as done with one-line summary
        - Add ← CURRENT marker to next task
        - Replace Current Task section with next task's subtasks
        - Keep file size constant
        """
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
                # Mark completed task done, strip old summary, append new one
                done_match = re.match(
                    r"^(\s*- \[)(.)\] Task " + str(task_num) + r": (.+?)(?: — .+)?(\s*← CURRENT)?$",
                    line,
                )
                if done_match:
                    new_lines.append(f"{done_match.group(1)}x] Task {task_num}: {done_match.group(3)} — {summary}")
                    continue

                # Add ← CURRENT marker to next task
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
            # Replace Current Task section
            new_section = f"Task: {task_num + 1}\nSpecialist: {next_specialist}\nSubtasks:\n"
            new_section += "".join(f"  {i}. {st}\n" for i, st in enumerate(next_subtasks, 1))
            content = re.sub(
                r"(## Current Task\n).+?(?=\n## ⚠|$)",
                r"\1" + new_section,
                content,
                flags=re.DOTALL,
            )
            # Update header
            total = len(re.findall(r"^\s*- \[", content, re.MULTILINE))
            content = re.sub(
                r"# Brain State · Task \d+ of \d+",
                f"# Brain State · Task {task_num + 1} of {total}",
                content,
            )
        else:
            # All tasks done — remove Current Task section, mark complete
            content = re.sub(r"## Current Task\n.+?(?=\n## ⚠|$)", "", content, flags=re.DOTALL)
            content = re.sub(
                r"# Brain State · Task \d+ of \d+",
                "# Brain State · COMPLETE",
                content,
            )

        self._atomic_write(content)
        logger.info("[Coordinator] memory.md updated after Task %d", task_num)

    def get_current_task_info(self) -> Dict:
        """Parse current task number, specialist, and subtasks from memory.md."""
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

    def get_task_list(self) -> List[Dict]:
        """Parse all tasks (id, name, done) from memory.md."""
        content = self.read_memory()
        tasks = []
        for status, num, name in re.findall(
            r"^\s*- \[(.?)\] Task (\d+): (.+?)(?:\s*—|\s*←|$)", content, re.MULTILINE
        ):
            tasks.append({"num": int(num), "name": name.strip(), "done": status == "x"})
        return tasks

    def verify_memory_integrity(self) -> bool:
        """Coordinator Rule #5: Halt if required sections are missing."""
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
        """Write to tmp file then rename — prevents partial-write corruption."""
        try:
            with open(self.tmp_file, "w") as f:
                f.write(content)
            os.replace(self.tmp_file, self.memory_file)
        except Exception as e:
            logger.error("[Coordinator] Atomic write failed: %s", e)
            raise

    # ════════════════════════════════════════════════════════════════
    # OUTPUT MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def write_task_output(
        self, task_id: int, task_name: str, specialist: str,
        subtasks: List[str], output: str
    ) -> str:
        """Write specialist output to outputs/task_NNN_<specialist>.md. Returns path string."""
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
        """Write the final synthesized answer to outputs/FINAL_ANSWER.md."""
        final_file = self.outputs_dir / "FINAL_ANSWER.md"
        with open(final_file, "w") as f:
            f.write(f"# Final Answer\n\n{answer}\n")
        logger.info("[Coordinator] Final answer written: %s", final_file)
        return str(final_file)

    # ════════════════════════════════════════════════════════════════
    # BRAIN OUTPUT PARSING — Deterministic (no LLM)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Strip Qwen3 extended thinking tokens (<think>...</think>).
        Kept for backward compat — _strip_all_reasoning is the full version.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def parse_brain_first_output(self, output: str) -> Tuple[List[str], str, List[str]]:
        """
        Parse brain's first-load output.
        Expected format:
            TASK_LIST:
            1. Task name
            2. Task name
            ...
            TASK_1_SPECIALIST: coder
            TASK_1_SUBTASKS:
            1. First subtask
            2. Second subtask

        Returns: (task_names, first_specialist, first_subtasks)
        """
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

        # Fallbacks
        if not task_names:
            logger.warning("[Coordinator] Brain produced no task list. Using fallback.")
            task_names = ["Analyze requirements", "Design architecture",
                          "Implement core", "Add features", "Test and finalize"]
        if not first_subtasks:
            first_subtasks = ["Understand the goal", "Plan the approach", "Produce output"]

        logger.info("[Coordinator] Parsed %d tasks, specialist=%s, %d subtasks",
                    len(task_names), first_specialist, len(first_subtasks))
        return task_names, first_specialist, first_subtasks

    def parse_brain_review_output(
        self, output: str, task_id: int, total_tasks: int
    ) -> Tuple[str, str, List[str], bool]:
        """
        Parse brain's between-task review output.
        Expected format:
            SUMMARY: One-line summary of task outcome
            NEXT_SPECIALIST: writer
            NEXT_SUBTASKS:
            1. First subtask for next task
            2. ...
            ADAPTATIONS: None (or changes made)

        Returns: (summary, next_specialist, next_subtasks, is_final)
        """
        output = _strip_all_reasoning(output)

        summary = "Task completed."
        next_specialist = "coder"
        next_subtasks: List[str] = []
        is_final = task_id >= total_tasks

        # PROJECT_COMPLETE is the only reliable signal — don't use broad FINAL match
        if "PROJECT_COMPLETE" in output:
            is_final = True

        m = re.search(r"SUMMARY:\s*(.+?)(?=\n|$)", output, re.IGNORECASE)
        if m:
            summary = m.group(1).strip()

        m = re.search(r"NEXT_SPECIALIST:\s*(\w+)", output, re.IGNORECASE)
        if m:
            next_specialist = m.group(1).strip().lower()

        m = re.search(
            r"NEXT_SUBTASKS:\n(.+?)(?=ADAPTATIONS|\n\n|$)",
            output, re.DOTALL | re.IGNORECASE,
        )
        if m:
            for line in m.group(1).strip().splitlines():
                lm = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if lm:
                    next_subtasks.append(lm.group(1).strip())

        if not next_subtasks and not is_final:
            next_subtasks = ["Continue from previous task", "Build on prior output"]

        return summary, next_specialist, next_subtasks, is_final

    # ════════════════════════════════════════════════════════════════
    # PROMPT BUILDING — Deterministic string templates (no LLM)
    # ════════════════════════════════════════════════════════════════

    def build_brain_first_prompt(
        self, goal: str, constraints: str, available_specialists: List[str]
    ) -> str:
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
3. Task name here
... (30–80 tasks, be specific)

TASK_1_SPECIALIST: <specialist_name>

TASK_1_SUBTASKS:
1. First subtask (concrete and actionable)
2. Second subtask
3. Third subtask
```

Create 30–80 tasks. Be specific about each task name. Task 1 subtasks should be granular enough for the specialist to execute without ambiguity.
"""

    def build_brain_review_prompt(
        self,
        task_id: int,
        task_name: str,
        memory_content: str,
        output_content: str,
        total_tasks: int,
    ) -> str:
        is_last = task_id >= total_tasks

        if is_last:
            job_text = """## Your Job
This is the FINAL task. Synthesize all work into a coherent summary.
Output a brief project synthesis. Then output: PROJECT_COMPLETE
"""
        else:
            job_text = f"""## Your Job
1. Read your memory above — this is your ONLY state (KV cache was wiped).
2. Read the task output below — assess what was actually produced.
3. Write ONE summary line for Task {task_id} (outcome + any plan change).
4. Assign the next specialist.
5. Write specific subtasks for Task {task_id + 1}.
6. If the output reveals issues or gaps, adapt the remaining task list.

## Output Format
```
SUMMARY: One-line outcome of Task {task_id}

NEXT_SPECIALIST: <specialist_name>

NEXT_SUBTASKS:
1. First subtask for Task {task_id + 1}
2. Second subtask
3. Third subtask

ADAPTATIONS: (describe any changes to remaining tasks, or "None")
```
"""

        return f"""You are the Brain. Your KV cache was wiped — memory.md is your ONLY state.

## Your Memory (read carefully)
{memory_content}

## Task {task_id} Output (just completed by specialist)
{output_content[:6000]}

{job_text}
"""

    def build_specialist_prompt(
        self, task_id: int, task_name: str, subtasks: List[str], goal: str
    ) -> str:
        subtasks_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(subtasks))
        return f"""You are a specialist AI executing a specific task in a larger project.

## Project Goal
{goal}

## Your Task
Task {task_id}: {task_name}

## Subtasks to Complete (execute ALL of them)
{subtasks_text}

## Rules
- Execute every subtask thoroughly.
- Produce complete, production-quality output.
- Use markdown formatting.
- Explain reasoning where relevant.
- List any assumptions made.

## Output
Write your complete response below.
"""

    def build_final_prompt(self, memory_content: str, all_outputs: List[Dict]) -> str:
        outputs_text = "".join(
            f"\n### Task {o['id']}: {o['name']}\n{o['content']}\n" for o in all_outputs
        )
        return f"""You are the Brain doing FINAL SYNTHESIS.

## Full Project Memory
{memory_content}

## All Task Outputs (truncated to first 3000 chars each)
{outputs_text}

## Your Job
Synthesize all outputs into a coherent, complete final deliverable.
Be thorough. This is the document the user will read.
"""

    # ════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════

    def get_project_summary(self, state: Dict) -> str:
        done = sum(1 for t in state["tasks"] if t["status"] == "done")
        total = len(state["tasks"])
        lines = [
            f"Project : {state['project_name']}",
            f"Status  : {state['status']}",
            f"Progress: {done}/{total} tasks",
            f"Current : Task {state['current_task_index']}",
            f"Loaded  : {state.get('loaded_model', 'none')}",
            "",
            "Tasks:",
        ]
        for task in state["tasks"]:
            icon = "✅" if task["status"] == "done" else "⏳" if task["status"] == "pending" else "🔧"
            spec = task.get("specialist", "—")
            lines.append(f"  {icon} {task['id']:3d}: {task['name']} ({spec})")
        return "\n".join(lines)
