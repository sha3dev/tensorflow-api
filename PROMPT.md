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

## Non-Negotiables

- You MUST load `skills/init-workflow/SKILL.md` before implementing the task.
- You MUST also load `skills/feature-shaping/SKILL.md`, `skills/simplicity-audit/SKILL.md`, and `skills/change-synchronization/SKILL.md`.
- If the task introduces meaningful behavior changes, you MUST also load `skills/test-scope-selection/SKILL.md`.
- If the task updates `README.md`, you MUST also load `skills/readme-authoring/SKILL.md`.
- If the project is a `node-service` or the task changes HTTP endpoints, you MUST also load `skills/http-api-conventions/SKILL.md`.
- You MUST execute `npm run check` yourself before finishing.
- If `npm run check` fails, you MUST fix the issues and rerun it until it passes.
- You MUST implement the task without editing managed files unless this is a standards update.

## Implementation Request

Complete this section before sending the prompt to your LLM.
Describe the behavior you want to implement, the expected public API, any runtime constraints, and any non-goals.

Task:

En varios proyectos quiero usar tensorflow, pero me encuentro con que la versión de NodeJS es muy limitada y no permite ciertas cosas que la versión de Python
si permite. Por lo que quiero crear una API que me permita usar la versión de python de tensorflow desde proyectos de NodeJS. Lo que quiero es crear un servicio que
exponga endpoints para hacer todo lo que permite hacer tensorflow, pero mediante una API (http, grpc, o lo que sea mas optimo).

Esta API debe permitir:

    * Crear cualquier modelo: esta operación recibe la configuración del modelo y un "id" de modelo. El "id" de modelo es un identificador único que se usará para referirse al modelo en las demás operaciones.
    * Entrenar un modelo: esta operación recibe el "id" del modelo, los datos de entrenamiento y los parámetros de entrenamiento.
    * Predecir con un modelo: esta operación recibe el "id" del modelo y los datos de entrada para predecir.
    * Obtener información de un modelo: esta operación recibe el "id" del modelo y devuelve información sobre el modelo: cuando se creo, cuantas veces se entreno, etc.

Si lo crees conveniente, podemos usar Tensorflow Python + La versión de tensorflow en C para inferir (me parece haber leido que es mas eficiente)

Es importante que el servicio me permita crear cualquier modelo, entrenarlo y predecir con él.

El servicio debe ser resiliente.

El servicio debe exponer tambien un pequeño dashboard para poder ver los modelos que se han creado y su estado. Por ejemplo, cuantas veces se ha entrenado, 
cuando ha sido la ultima perdicción, ultimo entrenamiento, etc.
