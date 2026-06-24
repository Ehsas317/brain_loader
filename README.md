# Brain Loader v3

A portable, hardware-aware agentic AI orchestration system.
Pure Python coordinator · Single hot-swap model slot · Adaptive planning loop.

**Primary target:** Apple Silicon via MLX  
**Included:** Ollama backend for any other system

> **Bugs fixed from original source:**
> 1. `first_task_specialist=` kwarg typo in orchestrator → `first_specialist=`
> 2. Qwen3 `<think>` token stripping missing from both parse methods
> 3. `"FINAL" in output.upper()` check too broad — caused premature termination
> 4. Telegram async used deprecated `get_event_loop()` pattern → `asyncio.run()`
>
> **Bugs fixed in this version (v3.1):**
> 1. **Telegram markdown escaping** — goals with `_`, `*`, `` ` `` characters caused notification failures; all user-provided text is now escaped before sending
> 2. **Constraints lost on resume** — `constraints` were not stored in `state.json`; resume now restores the original constraints
> 3. **MLX emergency cleanup** — `mx.synchronize()` was not wrapped in `try/except`, which could prevent `mx.metal.clear_cache()` from running during emergency cleanup
> 4. **Graceful shutdown** — added `SIGINT`/`SIGTERM` handlers that unload the model before exit (prevents model staying resident on Ctrl+C)
> 5. **Extended reasoning tag stripping** — `_strip_thinking()` now handles `<thinking>`, `<thought>`, `<reasoning>`, and `[Thinking: ...]` blocks from various model families, not just Qwen3 `<think>`

---

## What This Is

Brain Loader runs a long-horizon agentic task loop entirely on local hardware.

- A **Brain** model creates a 30–80 task plan and adapts it after every task.
- **Specialist** models (coder, researcher, writer, critic, math) execute each task.
- A **pure Python coordinator** handles all routing, state, memory, and parsing — zero LLM overhead for coordination.
- Only **one model lives in RAM at a time** (hot-swap). All available memory goes to the active model.

### Why the coordinator is pure Python (not another LLM)

| v1/v2 approach | v3 approach | Gain |
|---|---|---|
| Coordinator LLM (~4 GB) | Pure Python (0 GB) | +4 GB to KV cache |
| LLM parses brain output | Regex parses brain output | Deterministic, never hallucinates |
| LLM builds prompts | String templates | Testable, predictable |
| LLM manages state | Python dicts + JSON | Faster, no generation latency |

---

## Architecture

### RAM Layout

```
┌─────────────────────────────────────────────────────┐
│  Coordinator — Pure Python                          │
│  0 GB LLM RAM. Handles: routing, state, memory,    │
│  parsing, file I/O, crash recovery.                 │
├─────────────────────────────────────────────────────┤
│  Hot-Swap Slot                                      │
│  Brain (planner) OR Specialist (executor).          │
│  NEVER both. Hard constraint.                       │
│                                                     │
│  MLX examples (32 GB system):                       │
│    Qwen3-32B-Q4     ≈ 20 GB weights + 4 GB KV      │
│    Qwen2.5-72B-Q4   ≈ 22 GB weights + 2 GB KV      │
│    Llama-3.3-70B-Q2 ≈ 22 GB weights + 2 GB KV      │
│                                                     │
│  Ollama examples (size depends on your VRAM/RAM):   │
│    qwen3:32b          ≈ 20 GB                       │
│    qwen2.5-coder:32b  ≈ 20 GB                       │
│    qwen2.5:72b        ≈ 22 GB (needs 24+ GB RAM)    │
├─────────────────────────────────────────────────────┤
│  System overhead ≈ 3–4 GB                           │
└─────────────────────────────────────────────────────┘
```

### Execution Flow

```
User Goal
    │
    ▼
[Python] Load Brain
    │
Brain outputs:  TASK_LIST (all task names)
                TASK_1_SPECIALIST
                TASK_1_SUBTASKS
    │
[Python] Coordinator parses (regex)
[Python] Writes memory.md + state.json
[Python] Unload Brain
    │
    ▼
    ╔══════════════ TASK LOOP ═══════════════════════╗
    ║                                                ║
    ║  [Python] Load Specialist                      ║
    ║      │                                         ║
    ║  Specialist executes subtasks                  ║
    ║      │                                         ║
    ║  [Python] Write outputs/task_N.md              ║
    ║  [Python] Unload Specialist                    ║
    ║      │                                         ║
    ║  [Python] Load Brain                           ║
    ║      │                                         ║
    ║  Brain reads memory.md + task_N.md             ║
    ║  Brain outputs: SUMMARY, NEXT_SPECIALIST,      ║
    ║                 NEXT_SUBTASKS, ADAPTATIONS      ║
    ║      │                                         ║
    ║  [Python] Parse + update memory.md (atomic)    ║
    ║  [Python] Verify memory integrity              ║
    ║  [Python] Unload Brain                         ║
    ║      │                                         ║
    ╚══════ repeat until all tasks done ═════════════╝
    │
    ▼
[Python] Load Brain (final)
Brain synthesizes FINAL_ANSWER.md from all outputs
[Python] Send Telegram notification
[Python] Shutdown
```

### memory.md — Constant-Size State

The key design: completed subtasks are **deleted** and replaced with a single summary
line. File size stays bounded regardless of how many tasks have run.

```markdown
# Brain State · Task 3 of 47

## LOCKED — Never modify
Goal: Build a React Native fitness app with AI meal planner
Hard constraints: Must use TypeScript, must work offline

## Task List
- [x] Task 1: Market research — output thorough, proceeded as planned
- [x] Task 2: Architecture design — added 3 extra tasks for offline sync
- [ ] Task 3: Implement auth system ← CURRENT
- [ ] Task 4: Build REST API layer
- [ ] Task 5: Offline data sync
...

## Current Task
Task: 3
Specialist: coder
Subtasks:
  1. Implement JWT authentication with refresh tokens
  2. Add secure token storage (Keychain on iOS, Keystore on Android)
  3. Write auth middleware and route guards

## ⚠ MANDATORY LAST ACTION BEFORE UNLOAD
Delete current subtask block. Append one summary line to completed task.
Write next task subtasks. Update this file.
```

---

## File Structure

```
brain_loader/
├── main.py
├── config.yaml
├── requirements_mlx.txt
├── requirements_ollama.txt
├── core/
│   ├── __init__.py
│   ├── model_manager.py      ← MLX + Ollama backends behind abstract base
│   ├── coordinator.py        ← Pure Python state machine (all logic)
│   └── orchestrator.py       ← Execution loop (wires everything together)
├── utils/
│   ├── __init__.py
│   └── telegram_notify.py    ← Optional phone notifications
├── outputs/                  ← Auto-created. All task outputs live here.
├── logs/                     ← Auto-created. Timestamped log files.
├── memory.md                 ← Brain's persistent state (auto-managed)
└── state.json                ← Crash recovery checkpoint (auto-managed)
```

---

## Setup

### Apple Silicon (MLX backend)

```bash
# 1. Create project directory
mkdir brain_loader_v3 && cd brain_loader_v3
# (copy all files here)

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements_mlx.txt

# 4. Set backend in config.yaml
# backend: "mlx"

# 5. Download models (cached to ~/.cache/huggingface/)
huggingface-cli download mlx-community/Qwen3-32B-4bit
huggingface-cli download mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
# (download whichever specialists you need)

# 6. Run
python main.py "Build a SaaS dashboard with real-time analytics"
python main.py "My idea" --constraints "Must use Next.js and TypeScript"
python main.py --resume
```

### Any System with Ollama

```bash
# 1. Install Ollama (https://ollama.com)
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull qwen3:32b
ollama pull qwen2.5-coder:32b

# 3. Start Ollama server (if not auto-started)
ollama serve

# 4. Create project directory
mkdir brain_loader_v3 && cd brain_loader_v3
# (copy all files here)

# 5. Virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 6. Install dependencies
pip install -r requirements_ollama.txt

# 7. Set backend in config.yaml
# backend: "ollama"

# 8. Run
python main.py "Build a Python CLI tool for batch PDF processing"
python main.py --resume
```

---

## Adapting For Your System

### Changing models

Edit `config.yaml`. For MLX, find models at https://huggingface.co/mlx-community.
For Ollama, run `ollama list` to see what's pulled.

```yaml
# MLX — 32 GB system, all 32B models
brain_mlx:
  model_path: "mlx-community/Qwen3-32B-4bit"

# MLX — 16 GB system, use 8B / 14B models instead
brain_mlx:
  model_path: "mlx-community/Qwen3-14B-4bit"  # ~10 GB
  # or: "mlx-community/Qwen3-8B-4bit"         # ~6 GB

# Ollama — adjust model names to what you have
brain_ollama:
  model_path: "llama3.3:70b"     # if you have 48+ GB
  # or: "qwen3:14b"              # for 16 GB systems
  # or: "gemma3:27b"             # Google's Gemma 3
```

### Adding a custom specialist

In `config.yaml`, add to `specialists_mlx` or `specialists_ollama`:

```yaml
specialists_mlx:
  # ... existing ...
  planner:
    model_path: "mlx-community/Qwen3-32B-4bit"
    max_tokens: 8192
    temperature: 0.5
    ram_estimate_gb: 24.0
    description: "Strategic planning, roadmaps, project scoping"
```

The Brain will automatically learn to use `planner` as a specialist — the available specialist list is injected into the first-load prompt.

### Adjusting for smaller RAM

```yaml
# config.yaml — 16 GB system
brain_mlx:
  model_path: "mlx-community/Qwen3-14B-4bit"   # ~10 GB
  max_tokens_first_load: 8192
  max_tokens_subsequent: 6144
  max_tokens_final: 8192
  ram_estimate_gb: 12.0

memory:
  gc_sleep_seconds: 3           # give more time for memory to settle
  aggressive_cleanup: true
```

### Running Ollama on a remote machine

If you have a beefy machine on your LAN running Ollama:

```yaml
# On the host machine, start Ollama with network binding:
# OLLAMA_HOST=0.0.0.0 ollama serve

# In config.yaml on your laptop:
ollama_host: "http://192.168.1.50:11434"   # replace with host IP
```

---

## License

MIT
