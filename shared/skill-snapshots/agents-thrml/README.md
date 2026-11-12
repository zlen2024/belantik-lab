# thrml-skill

A portable agent skill for writing [THRML](https://www.extropic.ai) code — Extropic's JAX simulator for block Gibbs sampling, with a focus on Ising / Boltzmann machine sampling and training.

It ships as a standard `SKILL.md` (plus reference files) in the open [Agent Skills](https://opencode.ai/docs/skills/) format, so the same skill works across Claude Code, opencode, Codex CLI, Gemini CLI, and other tools that read `SKILL.md`.

## Install

The skill must live in a directory named `thrml/` (the folder name has to match the `name:` in `SKILL.md`). Drop that directory into wherever your agent looks for skills:

```bash
# Codex CLI, opencode, Gemini CLI — the shared .agents/ convention
git clone https://github.com/extropic-ai/thrml-skill.git .agents/skills/thrml

# Claude Code
git clone https://github.com/extropic-ai/thrml-skill.git .claude/skills/thrml
```

| Agent | Project (this repo) | Global (every project) |
| --- | --- | --- |
| **Claude Code** | `.claude/skills/thrml/` | `~/.claude/skills/thrml/` |
| **opencode** | `.agents/skills/thrml/`, `.claude/skills/thrml/`, or `.opencode/skills/thrml/` | `~/.config/opencode/skills/thrml/`, `~/.claude/skills/thrml/`, or `~/.agents/skills/thrml/` |
| **Codex CLI** | `.agents/skills/thrml/` (searched from your cwd up to the repo root) | `~/.agents/skills/thrml/` |
| **Gemini CLI & other `SKILL.md` tools** | `.agents/skills/thrml/` | that tool's skills directory |

Project paths scope the skill to one repo; the `~` global paths make it available everywhere. Restart the agent (or let it auto-rescan) after adding the directory.

Since `.agents/skills/` is the most widely read location, a handy setup if you use several tools is to clone there once and symlink `.claude/skills/thrml` to it — both opencode and Codex follow symlinked skill folders.

**Claude.ai** (no local clone): download this repo as a ZIP, then Settings → Capabilities → enable *Code execution and file creation*, and Customize → Skills → **+** → *Upload skill*.

## Use

Ask your agent to write or train a THRML model and the skill loads automatically when your request matches its description (THRML, Extropic, Ising / Boltzmann machines, block Gibbs sampling, energy-based models, etc.). In Claude Code you can also trigger it explicitly with `/thrml`. To confirm it's registered, ask *"What skills are available?"*
