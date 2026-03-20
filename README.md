# @sha3/tensorflow-api

HTTP service that lets Node.js applications create, train, and query TensorFlow models through a Python-backed job runtime.

## TL;DR

```bash
npm install
npm run check
npm run start
```

```bash
curl http://localhost:3000/
curl http://localhost:3000/dashboard
curl http://localhost:3000/api/state
```

## Why

Use this service when your Node.js application needs TensorFlow features that are easier or only available in Python, but you still want a stable HTTP contract, persisted job state, and a lightweight dashboard for operators.

The runtime keeps:
- HTTP transport in Node with `hono`
- TensorFlow execution in Python
- model and job metadata in SQLite
- model definitions, job payloads, and results on local disk

## Main Capabilities

- create declarative Keras models from JSON
- queue training jobs and run prediction on the fast path without file-based job orchestration
- expose model and job metadata through JSON endpoints
- serve a tiny SPA dashboard backed by `/api/state`, including model deletion
- recover interrupted running jobs as failed after restart

## Setup

### Requirements

- Node.js 20+
- internet access on first start so the service can bootstrap `uv`, a managed Python runtime, and TensorFlow wheels automatically

### Install Node dependencies

## Installation

```bash
npm install
```

### Run locally

## Running Locally

```bash
npm run start
```

What `npm run start` does on first run:

- installs `uv` if it is not already available
- creates `.venv/` in the project root
- installs a managed Python interpreter if needed
- syncs TensorFlow dependencies from `requirements/python-runtime.txt`
- verifies that the selected Python can import TensorFlow before starting Node
- recreates `.venv/` and retries once if the Python environment is present but broken
- starts the Node service with `PYTHON_BIN` pointing at `.venv`

If you start the service without `npm run start`, the runtime still prefers the local `.venv` Python automatically when that environment already exists.

Default URLs:

- service root: `http://localhost:3000/`
- dashboard SPA: `http://localhost:3000/dashboard`
- aggregate state API: `http://localhost:3000/api/state`

## Usage

```ts
import { ServiceRuntime } from "@sha3/tensorflow-api";

const serviceRuntime = ServiceRuntime.createDefault();
const server = serviceRuntime.startServer();
```

Use `buildServer()` when another process owner should control the port bind, and `dispose()` when you need an explicit shutdown path.

When you use `npm run start`, the bootstrap script will automatically prepare the Python runtime before calling into `src/main.ts`.

## Model Authoring

`POST /api/models` does not accept arbitrary Python source code. It accepts a declarative Keras payload built from a real Keras model config.

The intended flow is:

1. build a model in Python with Keras
2. call `model.get_config()`
3. send that config as `definition.modelConfig`
4. send the compile parameters you want the worker to apply as `definition.compileConfig`

### Supported formats

- `keras-sequential`
- `keras-functional`

### Payload shape

```json
{
  "modelId": "my-model",
  "definition": {
    "format": "keras-sequential",
    "modelConfig": {},
    "compileConfig": {
      "optimizer": "adam",
      "loss": "binary_crossentropy",
      "metrics": ["accuracy"]
    }
  }
}
```

### How to serialize a model from Python

```python
import json
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

payload = {
    "modelId": "sales-dense",
    "definition": {
        "format": "keras-sequential",
        "modelConfig": model.get_config(),
        "compileConfig": {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
    },
}

print(json.dumps(payload))
```

### Training payload notes

`trainingInput.inputs`, `trainingInput.targets`, `validationInputs`, and `validationTargets` are sent as plain JSON arrays for single-output models or keyed JSON objects for multi-output models. The Python worker converts them to tensors before calling Keras.

`trainingInput.sampleWeights` and `trainingInput.validationSampleWeights` accept either a numeric array or an object keyed by output name. `modelMetadata` replaces the persisted model metadata only after a training job succeeds.

Current `fitConfig` keys supported by the worker:

- `epochs`
- `batchSize`
- `shuffle`
- `validationSplit`

These are translated to the Keras names expected by `model.fit()`.

## Examples

Create a model:

```bash
curl -X POST http://localhost:3000/api/models \
  -H "content-type: application/json" \
  -d '{
    "modelId": "demo-model",
    "metadata": {
      "featureSetVersion": 1,
      "task": "forecast"
    },
    "definition": {
      "format": "keras-sequential",
      "modelConfig": {
        "layers": [
          {
            "class_name": "InputLayer",
            "config": {
              "batch_input_shape": [null, 1],
              "dtype": "float32"
            }
          }
        ]
      }
    }
  }'
```

Queue training:

```bash
curl -X POST http://localhost:3000/api/models/demo-model/training-jobs \
  -H "content-type: application/json" \
  -d '{
    "modelMetadata": {
      "featureSetVersion": 2,
      "trainedFor": "session-a"
    },
    "trainingInput": {
      "inputs": [[1], [2]],
      "targets": {
        "classification": [[1, 0], [0, 1]],
        "regression": [[0.12], [0.08]]
      },
      "sampleWeights": {
        "classification": [2, 0.5],
        "regression": [1, 1]
      }
    },
    "fitConfig": {
      "epochs": 2
    }
  }'
```

Run prediction synchronously on the request fast path:

```bash
curl -X POST http://localhost:3000/api/models/demo-model/prediction-jobs \
  -H "content-type: application/json" \
  -d '{
    "predictionInput": {
      "inputs": [[3], [4]]
    }
  }'
```

Example multi-output prediction response:

```json
{
  "modelId": "demo-model",
  "outputs": {
    "classification": [[1.2, -0.4, 0.1]],
    "regression": [[0.123]]
  },
  "status": "predicted"
}
```

Read the dashboard state:

```bash
curl http://localhost:3000/api/state
```

Create the model payload from Python first:

```python
import json
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print(
    json.dumps(
        {
            "modelId": "demo-model",
            "definition": {
                "format": "keras-sequential",
                "modelConfig": model.get_config(),
                "compileConfig": {
                    "optimizer": "adam",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"],
                },
            },
        }
    )
)
```

Create a functional model:

```python
import json
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,), name="risk_features")
x = tf.keras.layers.Dense(8, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="risk-model")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print(
    json.dumps(
        {
            "modelId": "risk-functional",
            "definition": {
                "format": "keras-functional",
                "modelConfig": model.get_config(),
                "compileConfig": {
                    "optimizer": "adam",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"],
                },
            },
        }
    )
)
```

Create a TCN-style model with dilations:

```python
import json
import tensorflow as tf

sequence_inputs = tf.keras.Input(shape=(32, 4), name="sequence_input")
x = tf.keras.layers.Conv1D(
    filters=16,
    kernel_size=3,
    dilation_rate=1,
    padding="causal",
    activation="relu",
)(sequence_inputs)
x = tf.keras.layers.Conv1D(
    filters=16,
    kernel_size=3,
    dilation_rate=2,
    padding="causal",
    activation="relu",
)(x)
x = tf.keras.layers.Conv1D(
    filters=16,
    kernel_size=3,
    dilation_rate=4,
    padding="causal",
    activation="relu",
)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
sequence_outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=sequence_inputs, outputs=sequence_outputs, name="dilated-tcn")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print(
    json.dumps(
        {
            "modelId": "dilated-tcn",
            "definition": {
                "format": "keras-functional",
                "modelConfig": model.get_config(),
                "compileConfig": {
                    "optimizer": "adam",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"],
                },
            },
        }
    )
)
```

Train the TCN-style model:

```bash
curl -X POST http://localhost:3000/api/models/dilated-tcn/training-jobs \
  -H "content-type: application/json" \
  -d '{
    "trainingInput": {
      "inputs": [
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
        [[0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2], [0.5, 0.4, 0.3, 0.2]]
      ],
      "targets": [[0.0], [1.0]]
    },
    "fitConfig": {
      "epochs": 2,
      "shuffle": false
    }
  }'
```

Inspect the TCN training result:

```bash
curl http://localhost:3000/api/jobs/<tcn-training-job-id>/result
```

## HTTP API

### `GET /`

Returns service metadata.

Status:

- `200`

Response:

```json
{
  "ok": true,
  "serviceName": "@sha3/tensorflow-api",
  "dashboardPath": "/dashboard",
  "statePath": "/api/state"
}
```

### `GET /dashboard`

Serves the dashboard SPA HTML shell.

Status:

- `200`

### `GET /dashboard/app.js`

Serves the dashboard SPA script.

Status:

- `200`

### `GET /api/state`

Returns aggregate dashboard state.

Status:

- `200`

Response shape:

```json
{
  "summary": {
    "modelCount": 1,
    "queuedJobCount": 0,
    "runningJobCount": 0,
    "failedJobCount": 0
  },
  "models": [],
  "recentJobs": []
}
```

### `POST /api/models`

Creates a model record, persists optional opaque `metadata`, and queues Python materialization.

Status:

- `201`
- `400` for invalid request payloads
- `409` when the model id already exists

### `GET /api/models`

Returns model summaries.

Status:

- `200`

### `GET /api/models/:modelId`

Returns one model record.

Status:

- `200`
- `404` when the model does not exist

### `PATCH /api/models/:modelId/metadata`

Replaces the persisted model metadata with the JSON object sent in the request body and returns the updated model record.

Status:

- `200`
- `400` for invalid JSON object payloads
- `404` when the model does not exist

### `DELETE /api/models/:modelId`

Deletes a model, its persisted artifact directory, and its recorded job history. The request is rejected while the model still has queued or running jobs.

Status:

- `204`
- `404` when the model does not exist
- `409` when the model has active jobs

### `POST /api/models/:modelId/training-jobs`

Queues model training. Supports `modelMetadata`, `sampleWeights`, and `validationSampleWeights`.

Status:

- `202`
- `400` for invalid payloads
- `404` when the model does not exist
- `409` when the model is not ready

### `POST /api/models/:modelId/prediction-jobs`

Runs model prediction synchronously and returns the prediction payload in the same response. For multi-output models, `outputs` preserves the real Keras output names.

Status:

- `200`
- `400` for invalid payloads
- `404` when the model does not exist
- `409` when the model is not ready
- `500` when the Python worker fails during prediction

### `GET /api/jobs`

Returns jobs ordered from newest to oldest. Failed training jobs include structured `diagnostics` when available.

Query params:

- `modelId`
- `status`

Status:

- `200`

### `GET /api/jobs/:jobId`

Returns one job record. Failed training jobs include structured `diagnostics` when available.

Status:

- `200`
- `404` when the job does not exist

### `GET /api/jobs/:jobId/result`

Returns the JSON job result for successful jobs. Successful training jobs include `modelId`, `status: "succeeded"`, `trainedAt`, and optional `history`. Failed jobs return `409` with the stored `errorCode`, full `errorMessage`, and structured `diagnostics`, including Python traceback text and model/training shape summaries when available.

Status:

- `200`
- `404` when the job does not exist
- `409` when the job has not finished successfully yet

### Error responses

Errors use this transport shape:

```json
{
  "code": "invalid_request",
  "message": "Human-readable explanation"
}
```

Current error codes:

- `invalid_request`
- `not_found`
- `conflict`
- `internal_error`

## Public API

### `ServiceRuntime`

Main runtime entrypoint for composing or starting the service.

```ts
import { ServiceRuntime } from "@sha3/tensorflow-api";

const serviceRuntime = ServiceRuntime.createDefault();
```

#### `createDefault()`

Purpose:
Creates the default runtime with SQLite storage, local filesystem artifact storage, the Hono HTTP server, and the Python worker integration.

Returns:
- `ServiceRuntime`

Behavior notes:
- initializes the SQLite schema
- marks interrupted running jobs as failed
- wires the default job polling loop configuration

#### `buildServer()`

Purpose:
Builds the Hono-backed Node server without binding a TCP port.

Returns:
- `ServerType`

Behavior notes:
- useful for tests or custom process orchestration
- does not start the background HTTP listener

#### `startServer()`

Purpose:
Builds the server, starts the background job polling loop, and listens on `config.DEFAULT_PORT`.

Returns:
- `ServerType`

Behavior notes:
- starts the local queued-job polling loop
- logs the local listening URL

#### `dispose()`

Purpose:
Stops the background polling loop and closes SQLite resources.

Returns:
- `void`

Behavior notes:
- use it for controlled shutdown in embedded runtimes or tests

### `AppInfoPayload`

Root metadata payload returned by `GET /`.

```ts
type AppInfoPayload = {
  ok: true;
  serviceName: string;
  dashboardPath: string;
  statePath: string;
};
```

### `KerasModelDefinition`

Declarative built-in Keras model definition accepted by `POST /api/models`.

```ts
type KerasModelDefinition = {
  format: "keras-functional" | "keras-sequential";
  modelConfig: Record<string, unknown>;
  compileConfig?: {
    optimizer?: Record<string, unknown> | string;
    loss?: Record<string, unknown> | string | string[];
    metrics?: Array<Record<string, unknown> | string>;
    runEagerly?: boolean;
  };
};
```

### `CreateModelRequest`

Request type for model creation.

```ts
type CreateModelRequest = {
  modelId: string;
  definition: KerasModelDefinition;
  metadata?: Record<string, unknown>;
};
```

### `CreateTrainingJobRequest`

Request type for queuing training.

```ts
type CreateTrainingJobRequest = {
  modelMetadata?: Record<string, unknown>;
  trainingInput: {
    inputs: unknown;
    sampleWeights?: TrainingSampleWeight;
    targets: unknown;
    validationInputs?: unknown;
    validationSampleWeights?: TrainingSampleWeight;
    validationTargets?: unknown;
  };
  fitConfig?: {
    epochs?: number;
    batchSize?: number;
    shuffle?: boolean;
    validationSplit?: number;
  };
};
```

### `TrainingSampleWeight`

Training sample weight shape supported by single-output and multi-output jobs.

```ts
type TrainingSampleWeight = number[] | Record<string, number[]>;
```

### `CreatePredictionJobRequest`

Request type for synchronous prediction.

```ts
type CreatePredictionJobRequest = {
  predictionInput: {
    inputs: unknown;
  };
};
```

### `ModelRecord`

Persisted model metadata returned by model and dashboard endpoints.

```ts
type ModelRecord = {
  modelId: string;
  status: "pending" | "ready" | "failed";
  createdAt: string;
  updatedAt: string;
  trainingCount: number;
  lastTrainingAt: string | null;
  lastTrainingJobId: string | null;
  metadata: Record<string, unknown> | null;
  definitionPath: string;
  artifactPath: string;
};
```

### `JobRecord`

Persisted job metadata returned by job endpoints.

```ts
type JobRecord = {
  jobId: string;
  modelId: string;
  jobType: "create_model" | "train_model";
  status: "queued" | "running" | "succeeded" | "failed";
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  errorCode: string | null;
  errorMessage: string | null;
  requestPath: string;
  resultPath: string;
};
```

### `JobResultPayload`

Opaque JSON result payload returned by `GET /api/jobs/:jobId/result`.

```ts
type JobResultPayload = Record<string, unknown>;
```

### `JobFailureDiagnostics`

Structured diagnostics persisted for failed training jobs.

```ts
type JobFailureDiagnostics = {
  modelOutputNames: string[];
  modelOutputCount: number;
  trainingInputSummary: TrainingInputSummary;
  pythonExceptionType: string;
  traceback: string;
  stderrTail: string;
};
```

### `TrainingInputSummary`

Compact summary of the training request structures captured before `model.fit(...)`.

```ts
type TrainingInputSummary = {
  inputShape: ShapeSummary | null;
  inputTypes: TypeSummary | null;
  targetKeys: string[];
  targetShapes: ShapeSummary | null;
  sampleWeightKeys: string[];
  sampleWeightShapes: ShapeSummary | null;
  validationInputShape: ShapeSummary | null;
  validationInputTypes: TypeSummary | null;
  validationTargetKeys: string[];
  validationTargetShapes: ShapeSummary | null;
  validationSampleWeightKeys: string[];
  validationSampleWeightShapes: ShapeSummary | null;
};
```

### `ShapeSummary`

Compact shape summary used inside diagnostics instead of serializing large payloads.

```ts
type ShapeSummary = number[] | Record<string, ShapeSummary>;
```

### `TypeSummary`

Compact type summary used inside diagnostics for inputs without serializing full payloads.

```ts
type TypeSummary = string | Record<string, TypeSummary>;
```

### `FailedJobResultPayload`

Failed job payload returned by `GET /api/jobs/:jobId/result` with status `409`.

```ts
type FailedJobResultPayload = {
  modelId: string;
  status: "failed";
  errorCode: string;
  errorMessage: string;
  diagnostics?: JobFailureDiagnostics;
};
```

### `PredictionResultPayload`

Prediction response payload returned by `POST /api/models/:modelId/prediction-jobs`.

```ts
type PredictionResultPayload = {
  modelId: string;
  outputs: Record<string, unknown> | unknown;
  status: "predicted";
};
```

### `TrainingJobResultPayload`

Successful training job result returned by `GET /api/jobs/:jobId/result`.

```ts
type TrainingJobResultPayload = {
  modelId: string;
  status: "succeeded";
  trainedAt: string;
  history?: Record<string, unknown>;
};
```

### `DashboardStatePayload`

Aggregate payload returned by `GET /api/state`.

```ts
type DashboardStatePayload = {
  summary: {
    modelCount: number;
    queuedJobCount: number;
    runningJobCount: number;
    failedJobCount: number;
  };
  models: ModelRecord[];
  recentJobs: JobRecord[];
};
```

## Compatibility

- Node.js 20+
- Python 3.9+
- ESM package runtime
- local single-node deployment model for this first release

## Configuration

Configuration lives in [`src/config.ts`](/Users/jc/Documents/GitHub/tensorflow-api/src/config.ts).

- `config.RESPONSE_CONTENT_TYPE`: JSON `content-type` used by API responses.
- `config.DEFAULT_PORT`: local port used by `startServer()`.
- `config.SERVICE_NAME`: service name returned by `GET /`.
- `config.STORAGE_ROOT`: root directory where model and job files are written.
- `config.SQLITE_PATH`: SQLite database path for persisted metadata.
- `config.PYTHON_BIN`: Python executable used to run the TensorFlow worker script. Defaults to `.venv/bin/python` when present, otherwise `python3`.
- `config.PYTHON_WORKER_SCRIPT`: path to the Python worker entrypoint.
- `config.JOB_POLL_INTERVAL_MS`: interval used by the background Node polling loop.

## Scripts

- `npm run start`: start the service with `tsx`
- `npm run build`: compile TypeScript to `dist/`
- `npm run standards:check`: run contract verification
- `npm run typecheck`: run TypeScript checks
- `npm run test`: run the Node test suite
- `npm run check`: run standards, lint, formatting, typecheck, and tests

## Structure

- `src/app/service-runtime.service.ts`: runtime composition and polling lifecycle
- `src/http/http-server.service.ts`: HTTP transport and dashboard asset serving
- `src/model/model.service.ts`: model creation and lookup
- `src/job/job.service.ts`: queued training jobs and fast-path prediction execution
- `src/storage/storage.service.ts`: SQLite and filesystem persistence
- `src/python-runtime/python-runtime.service.ts`: Python process execution
- `python/tensorflow_api_worker.py`: TensorFlow worker script

## Troubleshooting

### Python worker fails immediately

Confirm the configured Python binary can import TensorFlow:

```bash
$PYTHON_BIN -c "import tensorflow as tf; print(tf.__version__)"
```

If that fails, remove `.venv/` and rerun `npm run start`, or set `TENSORFLOW_API_PYTHON_VERSION` to a Python version with compatible TensorFlow wheels.

### Jobs remain queued

Check that the Node process is running with `startServer()` and that `config.JOB_POLL_INTERVAL_MS` is not set to an unexpectedly large value.

### SQLite or filesystem permission errors

Ensure the process can create directories and write files under `config.STORAGE_ROOT`, and that the parent directory for `config.SQLITE_PATH` is writable.

### Bootstrap wants a different Python version

Override the managed Python version during startup:

```bash
TENSORFLOW_API_PYTHON_VERSION=3.11 npm run start
```

## AI Workflow

- Read `AGENTS.md`, `SKILLS.md`, `ai/contract.json`, and the relevant `ai/<assistant>.md` before changing behavior.
- Keep managed files read-only unless the task is explicitly a standards update.
- Update tests, README examples, exported types, and HTTP docs in the same pass as behavior changes.
- Run `npm run standards:check` and `npm run check` before finishing work.
