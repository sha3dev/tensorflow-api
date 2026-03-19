Read these files before making any implementation changes:

- `AGENTS.md`
- `SKILLS.md`
- `ai/contract.json`
- `ai/rules.md`
- `prompts/init-contract.md`
- `skills/init-workflow/SKILL.md`
- `skills/feature-shaping/SKILL.md`
- `skills/simplicity-audit/SKILL.md`
- `skills/change-synchronization/SKILL.md`
- the assistant-specific adapter in `ai/`

Your job is to implement the requested behavior in the scaffold under `src/` and `test/` following the rules in `ai/rules.md`, `prompts/init-contract.md`, and `skills/init-workflow/SKILL.md`.
You MUST also load `skills/feature-shaping/SKILL.md`, `skills/simplicity-audit/SKILL.md`, and `skills/change-synchronization/SKILL.md`.
If the task introduces meaningful behavior changes, you MUST load `skills/test-scope-selection/SKILL.md`.
If the task creates or updates `README.md`, you MUST also load `skills/readme-authoring/SKILL.md` before editing it.
If the project is a `node-service` or the task changes HTTP endpoints, you MUST also load `skills/http-api-conventions/SKILL.md`.

Implementation reminders:

- Let Biome decide final layout and wrapping.
- Fix `error` rules first; review `warning` rules carefully instead of overcorrecting them.
- Simplify before introducing abstractions or extra files.
- Rewrite `README.md` last so it matches the final public behavior.

## Package Specification

- Goal:
- Public API:
- Runtime constraints:
- Required dependencies:
- Feature requirements:
