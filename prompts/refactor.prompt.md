Read these files before making any implementation changes:

- `AGENTS.md`
- `SKILLS.md`
- `ai/contract.json`
- `ai/rules.md`
- `prompts/refactor-contract.md`
- `skills/refactor-workflow/SKILL.md`
- `skills/feature-shaping/SKILL.md`
- `skills/simplicity-audit/SKILL.md`
- `skills/change-synchronization/SKILL.md`
- `.code-standards/refactor-source/public-contract.json`
- `.code-standards/refactor-source/preservation.json`
- `.code-standards/refactor-source/analysis-summary.md`

Your job is to refactor the project into the fresh scaffold under `src/` and `test/` following the rules in `ai/rules.md`, `prompts/refactor-contract.md`, and `skills/refactor-workflow/SKILL.md`.
You MUST also load `skills/feature-shaping/SKILL.md`, `skills/simplicity-audit/SKILL.md`, and `skills/change-synchronization/SKILL.md`.
If the refactor changes meaningful behavior, you MUST load `skills/test-scope-selection/SKILL.md`.
If the refactor rebuilds or documents HTTP transport behavior, you MUST also load `skills/http-api-conventions/SKILL.md`.
If the refactor rewrites `README.md`, you MUST also load `skills/readme-authoring/SKILL.md`.

Implementation reminders:

- Let Biome decide final layout and wrapping.
- Fix `error` rules first; review `warning` rules carefully instead of overcorrecting them.
- Simplify before introducing abstractions or extra files.
- Rewrite `README.md` after behavior is stable so it documents the real result.
