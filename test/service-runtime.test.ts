import * as assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { AppInfoService } from "../src/app-info/index.ts";
import { DashboardStateService } from "../src/dashboard-state/index.ts";
import { HttpServerService } from "../src/http/index.ts";
import { JobService } from "../src/job/index.ts";
import type { JobRecord } from "../src/job/index.ts";
import { ModelService } from "../src/model/index.ts";
import { PythonRuntimeService } from "../src/python-runtime/index.ts";
import type { PythonJobCommand, PythonJobExecutor, PythonPredictionCommand, PythonPredictionExecutor } from "../src/python-runtime/index.ts";
import { StorageService } from "../src/storage/index.ts";

type TestHarness = {
  baseUrl: string;
  close: () => Promise<void>;
  jobService: JobService;
  storageService: StorageService;
  tempRoot: string;
};

type FakeExecutorOptions = {
  failingAction?: PythonJobCommand["action"] | "predict-model";
  failureResultPayload?: Record<string, unknown>;
  predictionResult?: Record<string, unknown>;
  tempRoot?: string;
};

type StatePayload = {
  models: Array<{
    metadata: Record<string, unknown> | null;
    trainingCount: number;
  }>;
  summary: {
    failedJobCount: number;
    modelCount: number;
    queuedJobCount: number;
  };
};

function createFakeExecutor(options?: FakeExecutorOptions): PythonJobExecutor {
  return async (command) => {
    const payload = JSON.parse(readFileSync(command.requestPath, "utf8")) as Record<string, unknown>;

    if (options?.failingAction === command.action) {
      if (options.failureResultPayload) {
        writeFileSync(command.resultPath, JSON.stringify(options.failureResultPayload), "utf8");
      }
      return { errorMessage: `simulated ${command.action} failure`, isSuccess: false };
    }

    if (command.action === "create-model") {
      writeFileSync(String(payload.artifactPath), JSON.stringify({ created: true }), "utf8");
      writeFileSync(command.resultPath, JSON.stringify({ artifactPath: payload.artifactPath, modelId: payload.modelId, status: "ready" }), "utf8");
    }

    if (command.action === "train-model") {
      writeFileSync(command.resultPath, JSON.stringify({ history: { loss: [0.5, 0.2] }, modelId: payload.modelId, status: "trained" }), "utf8");
    }
    return { errorMessage: null, isSuccess: true };
  };
}

function createFakePredictionExecutor(options?: FakeExecutorOptions): PythonPredictionExecutor {
  return async (command: PythonPredictionCommand) => {
    if (options?.failingAction === "predict-model") {
      return { errorMessage: "simulated predict-model failure", isSuccess: false };
    }

    const result = options?.predictionResult || {
      modelId: command.modelId,
      outputs: command.predictionInput.inputs,
      status: "predicted",
    };

    return {
      isSuccess: true,
      result,
    };
  };
}

async function createHarness(options?: FakeExecutorOptions): Promise<TestHarness> {
  const tempRoot = options?.tempRoot || mkdtempSync(join(tmpdir(), "tensorflow-api-"));
  const storageRoot = join(tempRoot, "storage");
  const sqlitePath = join(tempRoot, "storage", "runtime.sqlite");
  const storageService = new StorageService(sqlitePath, storageRoot);
  const modelService = ModelService.create(storageService);
  const pythonRuntimeService = new PythonRuntimeService(
    "python3",
    "python/tensorflow_api_worker.py",
    createFakeExecutor(options),
    createFakePredictionExecutor(options),
  );
  const jobService = JobService.create(storageService, modelService, pythonRuntimeService);
  const dashboardStateService = DashboardStateService.create(storageService);
  const appInfoService = AppInfoService.createDefault();
  const httpServerService = HttpServerService.create(appInfoService, modelService, jobService, dashboardStateService);
  const server = httpServerService.buildServer();
  storageService.initialize();
  jobService.recoverInterruptedJobs();

  await new Promise<void>((resolve) => {
    server.listen(0, () => {
      resolve();
    });
  });

  const address = server.address();

  if (!address || typeof address === "string") {
    throw new Error("failed to bind test server");
  }

  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: async () => {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        });
      });
      storageService.close();
      rmSync(tempRoot, { force: true, recursive: true });
    },
    jobService,
    storageService,
    tempRoot,
  };
}

async function createReadyModel(harness: TestHarness, modelId: string, metadata?: Record<string, unknown>): Promise<void> {
  await fetch(`${harness.baseUrl}/api/models`, {
    body: JSON.stringify({
      definition: { format: "keras-sequential", modelConfig: { layers: [] } },
      metadata,
      modelId,
    }),
    headers: { "content-type": "application/json" },
    method: "POST",
  });

  await harness.jobService.processNextQueuedJob();
}

test("service exposes root metadata, dashboard shell, dashboard asset, and empty state", async () => {
  const harness = await createHarness();

  try {
    const rootResponse = await fetch(`${harness.baseUrl}/`);
    const dashboardResponse = await fetch(`${harness.baseUrl}/dashboard`);
    const assetResponse = await fetch(`${harness.baseUrl}/dashboard/app.js`);
    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const rootPayload = await rootResponse.json();
    const dashboardMarkup = await dashboardResponse.text();
    const assetSource = await assetResponse.text();
    const statePayload = await stateResponse.json();

    assert.equal(rootResponse.status, 200);
    assert.deepEqual(rootPayload, {
      dashboardPath: "/dashboard",
      ok: true,
      serviceName: "@sha3/tensorflow-api",
      statePath: "/api/state",
    });
    assert.equal(dashboardResponse.status, 200);
    assert.match(dashboardMarkup, /TensorFlow API Dashboard/);
    assert.equal(assetResponse.status, 200);
    assert.match(assetSource, /\/api\/state/);
    assert.match(assetSource, /delete-model/);
    assert.match(assetSource, /Expand for full traceback/);
    assert.match(assetSource, /error-details/);
    assert.match(assetSource, /renderDateCell/);
    assert.match(assetSource, /renderIdentifierCell/);
    assert.match(assetSource, /break-anywhere/);
    assert.equal(stateResponse.status, 200);
    assert.deepEqual(statePayload, {
      models: [],
      recentJobs: [],
      summary: { failedJobCount: 0, modelCount: 0, queuedJobCount: 0, runningJobCount: 0 },
    });
  } finally {
    await harness.close();
  }
});

test("service creates models, exposes state, and rejects duplicates", async () => {
  const harness = await createHarness();

  try {
    const createResponse = await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({
        definition: {
          format: "keras-sequential",
          modelConfig: { layers: [{ class_name: "InputLayer", config: { batch_input_shape: [null, 1], dtype: "float32" } }] },
        },
        metadata: { market: "alpha", version: 1 },
        modelId: "alpha-model",
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const createPayload = await createResponse.json();
    const duplicateResponse = await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({
        definition: { format: "keras-sequential", modelConfig: { layers: [] } },
        modelId: "alpha-model",
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const duplicatePayload = await duplicateResponse.json();
    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const statePayload = await stateResponse.json();
    const modelResponse = await fetch(`${harness.baseUrl}/api/models/alpha-model`);
    const modelPayload = await modelResponse.json();

    assert.equal(createResponse.status, 201);
    assert.equal(createPayload.kind, "created");
    assert.equal(createPayload.model.status, "pending");
    assert.deepEqual(createPayload.model.metadata, { market: "alpha", version: 1 });
    assert.equal(duplicateResponse.status, 409);
    assert.deepEqual(duplicatePayload, {
      code: "conflict",
      message: "model 'alpha-model' already exists",
    });
    assert.equal(modelResponse.status, 200);
    assert.deepEqual(modelPayload.metadata, { market: "alpha", version: 1 });
    assert.equal(statePayload.models.length, 1);
    assert.deepEqual(statePayload.models[0].metadata, { market: "alpha", version: 1 });
    assert.equal(statePayload.summary.modelCount, 1);
    assert.equal(statePayload.summary.queuedJobCount, 1);
  } finally {
    await harness.close();
  }
});

test("service processes create and train jobs while prediction stays synchronous", async () => {
  const harness = await createHarness();

  try {
    const createResponse = await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({
        definition: { format: "keras-sequential", modelConfig: { layers: [] } },
        modelId: "beta-model",
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const createPayload = await createResponse.json();
    const creationJobId = createPayload.job.jobId as string;

    await harness.jobService.processNextQueuedJob();

    const createdModelResponse = await fetch(`${harness.baseUrl}/api/models/beta-model`);
    const createdModelPayload = await createdModelResponse.json();
    const trainResponse = await fetch(`${harness.baseUrl}/api/models/beta-model/training-jobs`, {
      body: JSON.stringify({ fitConfig: { epochs: 2 }, trainingInput: { inputs: [[1], [2]], targets: [[1], [0]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const trainPayload = await trainResponse.json();
    const predictionResponse = await fetch(`${harness.baseUrl}/api/models/beta-model/prediction-jobs`, {
      body: JSON.stringify({ predictionInput: { inputs: [[3], [4]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const predictionPayload = await predictionResponse.json();

    await harness.jobService.processNextQueuedJob();

    const jobsResponse = await fetch(`${harness.baseUrl}/api/jobs?modelId=beta-model`);
    const jobsPayload = (await jobsResponse.json()) as JobRecord[];
    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const statePayload = (await stateResponse.json()) as StatePayload;
    const creationJobResponse = await fetch(`${harness.baseUrl}/api/jobs/${creationJobId}`);
    const creationJobPayload = await creationJobResponse.json();

    assert.equal(createdModelResponse.status, 200);
    assert.equal(createdModelPayload.status, "ready");
    assert.equal(trainResponse.status, 202);
    assert.equal(trainPayload.status, "queued");
    assert.equal(predictionResponse.status, 200);
    assert.deepEqual(predictionPayload, {
      modelId: "beta-model",
      outputs: [[3], [4]],
      status: "predicted",
    });
    assert.equal(jobsResponse.status, 200);
    assert.equal(jobsPayload.length, 2);
    assert.equal(creationJobResponse.status, 200);
    assert.equal(creationJobPayload.status, "succeeded");
    assert.equal(statePayload.summary.modelCount, 1);
    assert.equal(statePayload.summary.queuedJobCount, 0);
    assert.equal(statePayload.models.length, 1);
    assert.equal(statePayload.models[0]?.trainingCount, 1);
  } finally {
    await harness.close();
  }
});

test("service updates model metadata only after successful training", async () => {
  const harness = await createHarness();

  try {
    await createReadyModel(harness, "metadata-model", { version: 1 });

    const trainingResponse = await fetch(`${harness.baseUrl}/api/models/metadata-model/training-jobs`, {
      body: JSON.stringify({
        modelMetadata: { version: 2, window: 128 },
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: [1, 0.5],
          targets: [[0], [1]],
        },
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const trainingPayload = await trainingResponse.json();

    await harness.jobService.processNextQueuedJob();

    const modelResponse = await fetch(`${harness.baseUrl}/api/models/metadata-model`);
    const modelPayload = await modelResponse.json();
    const trainingResultResponse = await fetch(`${harness.baseUrl}/api/jobs/${trainingPayload.jobId}/result`);
    const trainingResultPayload = await trainingResultResponse.json();

    assert.equal(trainingResponse.status, 202);
    assert.equal(modelResponse.status, 200);
    assert.deepEqual(modelPayload.metadata, { version: 2, window: 128 });
    assert.equal(trainingResultResponse.status, 200);
    assert.equal(trainingResultPayload.modelId, "metadata-model");
    assert.equal(trainingResultPayload.status, "succeeded");
    assert.equal(typeof trainingResultPayload.trainedAt, "string");
    assert.deepEqual(trainingResultPayload.history, { loss: [0.5, 0.2] });
  } finally {
    await harness.close();
  }
});

test("service deletes a model and removes it from the dashboard state", async () => {
  const harness = await createHarness();

  try {
    await createReadyModel(harness, "delete-model", { family: "forecast" });

    const deleteResponse = await fetch(`${harness.baseUrl}/api/models/delete-model`, {
      method: "DELETE",
    });
    const modelResponse = await fetch(`${harness.baseUrl}/api/models/delete-model`);
    const modelPayload = await modelResponse.json();
    const jobsResponse = await fetch(`${harness.baseUrl}/api/jobs?modelId=delete-model`);
    const jobsPayload = await jobsResponse.json();
    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const statePayload = (await stateResponse.json()) as StatePayload;

    assert.equal(deleteResponse.status, 204);
    assert.equal(modelResponse.status, 404);
    assert.deepEqual(modelPayload, {
      code: "not_found",
      message: "model 'delete-model' was not found",
    });
    assert.deepEqual(jobsPayload, []);
    assert.equal(statePayload.summary.modelCount, 0);
    assert.deepEqual(statePayload.models, []);
  } finally {
    await harness.close();
  }
});

test("service rejects deleting a model with active jobs", async () => {
  const harness = await createHarness();

  try {
    await createReadyModel(harness, "busy-model");

    const trainingResponse = await fetch(`${harness.baseUrl}/api/models/busy-model/training-jobs`, {
      body: JSON.stringify({ trainingInput: { inputs: [[1]], targets: [[1]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const deleteResponse = await fetch(`${harness.baseUrl}/api/models/busy-model`, {
      method: "DELETE",
    });
    const deletePayload = await deleteResponse.json();
    const modelResponse = await fetch(`${harness.baseUrl}/api/models/busy-model`);
    const modelPayload = await modelResponse.json();

    assert.equal(trainingResponse.status, 202);
    assert.equal(deleteResponse.status, 409);
    assert.deepEqual(deletePayload, {
      code: "conflict",
      message: "model 'busy-model' has active jobs and cannot be deleted",
    });
    assert.equal(modelResponse.status, 200);
    assert.equal(modelPayload.modelId, "busy-model");
  } finally {
    await harness.close();
  }
});

test("service keeps prior metadata when training fails", async () => {
  const harness = await createHarness({
    failingAction: "train-model",
    failureResultPayload: {
      diagnostics: {
        modelOutputCount: 2,
        modelOutputNames: ["regression", "classification"],
        pythonExceptionType: "KeyError",
        stderrTail: "",
        traceback: "Traceback (most recent call last):\nKeyError: 0",
        trainingInputSummary: {
          sampleWeightKeys: ["classification", "regression"],
          sampleWeightShapes: {
            classification: [1],
            regression: [1],
          },
          targetKeys: ["classification", "regression"],
          targetShapes: {
            classification: [1, 2],
            regression: [1, 1],
          },
          validationSampleWeightKeys: ["classification", "regression"],
          validationSampleWeightShapes: {
            classification: [1],
            regression: [1],
          },
          validationTargetKeys: ["classification", "regression"],
          validationTargetShapes: {
            classification: [1, 2],
            regression: [1, 1],
          },
        },
      },
      errorCode: "internal_error",
      errorMessage: "Traceback (most recent call last):\nKeyError: 0",
      modelId: "failed-training-model",
      status: "failed",
    },
  });

  try {
    await createReadyModel(harness, "failed-training-model", { version: 1 });

    const trainingResponse = await fetch(`${harness.baseUrl}/api/models/failed-training-model/training-jobs`, {
      body: JSON.stringify({
        modelMetadata: { version: 2 },
        trainingInput: {
          inputs: [[1]],
          sampleWeights: {
            classification: [1],
            regression: [1],
          },
          targets: {
            classification: [[1, 0]],
            regression: [[1]],
          },
          validationInputs: [[1]],
          validationSampleWeights: {
            classification: [1],
            regression: [1],
          },
          validationTargets: {
            classification: [[1, 0]],
            regression: [[1]],
          },
        },
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const trainingPayload = await trainingResponse.json();

    await harness.jobService.processNextQueuedJob();

    const modelResponse = await fetch(`${harness.baseUrl}/api/models/failed-training-model`);
    const modelPayload = await modelResponse.json();
    const failedJobResponse = await fetch(`${harness.baseUrl}/api/jobs/${trainingPayload.jobId}`);
    const failedJobPayload = await failedJobResponse.json();
    const failedJobResultResponse = await fetch(`${harness.baseUrl}/api/jobs/${trainingPayload.jobId}/result`);
    const failedJobResultPayload = await failedJobResultResponse.json();

    assert.equal(trainingResponse.status, 202);
    assert.equal(modelResponse.status, 200);
    assert.deepEqual(modelPayload.metadata, { version: 1 });
    assert.equal(failedJobResponse.status, 200);
    assert.equal(failedJobPayload.status, "failed");
    assert.deepEqual(failedJobPayload.diagnostics.modelOutputNames, ["regression", "classification"]);
    assert.equal(failedJobPayload.diagnostics.modelOutputCount, 2);
    assert.deepEqual(failedJobPayload.diagnostics.trainingInputSummary.targetKeys, ["classification", "regression"]);
    assert.deepEqual(failedJobPayload.diagnostics.trainingInputSummary.sampleWeightKeys, ["classification", "regression"]);
    assert.match(failedJobPayload.diagnostics.traceback, /KeyError: 0/);
    assert.equal(failedJobPayload.diagnostics.stderrTail, "simulated train-model failure");
    assert.equal(JSON.stringify(failedJobPayload.diagnostics).includes("[[1,0]]"), false);
    assert.equal(failedJobResultResponse.status, 409);
    assert.deepEqual(failedJobResultPayload, {
      code: "internal_error",
      diagnostics: {
        modelOutputCount: 2,
        modelOutputNames: ["regression", "classification"],
        pythonExceptionType: "KeyError",
        stderrTail: "simulated train-model failure",
        traceback: "Traceback (most recent call last):\nKeyError: 0",
        trainingInputSummary: {
          sampleWeightKeys: ["classification", "regression"],
          sampleWeightShapes: {
            classification: [1],
            regression: [1],
          },
          targetKeys: ["classification", "regression"],
          targetShapes: {
            classification: [1, 2],
            regression: [1, 1],
          },
          validationSampleWeightKeys: ["classification", "regression"],
          validationSampleWeightShapes: {
            classification: [1],
            regression: [1],
          },
          validationTargetKeys: ["classification", "regression"],
          validationTargetShapes: {
            classification: [1, 2],
            regression: [1, 1],
          },
        },
      },
      errorCode: "internal_error",
      errorMessage: "simulated train-model failure",
      message: "simulated train-model failure",
      modelId: "failed-training-model",
      status: "failed",
    });
  } finally {
    await harness.close();
  }
});

test("service returns named outputs for multi-output prediction", async () => {
  const harness = await createHarness({
    predictionResult: {
      modelId: "multi-output-model",
      outputs: {
        classification: [[1.2, -0.4, 0.1]],
        regression: [[0.123]],
      },
      status: "predicted",
    },
  });

  try {
    await createReadyModel(harness, "multi-output-model");

    const predictionResponse = await fetch(`${harness.baseUrl}/api/models/multi-output-model/prediction-jobs`, {
      body: JSON.stringify({ predictionInput: { inputs: [[3], [4]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const predictionPayload = await predictionResponse.json();

    assert.equal(predictionResponse.status, 200);
    assert.deepEqual(predictionPayload, {
      modelId: "multi-output-model",
      outputs: {
        classification: [[1.2, -0.4, 0.1]],
        regression: [[0.123]],
      },
      status: "predicted",
    });
  } finally {
    await harness.close();
  }
});

test("service returns validation and not-found errors for invalid requests", async () => {
  const harness = await createHarness();

  try {
    const invalidCreateResponse = await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({ definition: {}, modelId: "bad/model" }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const invalidCreatePayload = await invalidCreateResponse.json();
    const missingModelResponse = await fetch(`${harness.baseUrl}/api/models/missing-model/training-jobs`, {
      body: JSON.stringify({ trainingInput: { inputs: [[1]], targets: [[1]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const missingModelPayload = await missingModelResponse.json();

    assert.equal(invalidCreateResponse.status, 400);
    assert.deepEqual(invalidCreatePayload, {
      code: "invalid_request",
      message: "modelId must be a safe non-empty identifier",
    });
    assert.equal(missingModelResponse.status, 404);
    assert.deepEqual(missingModelPayload, {
      code: "not_found",
      message: "model 'missing-model' was not found",
    });
  } finally {
    await harness.close();
  }
});

test("service marks failed jobs when the Python worker reports an error", async () => {
  const harness = await createHarness({ failingAction: "predict-model" });

  try {
    await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({
        definition: { format: "keras-sequential", modelConfig: { layers: [] } },
        modelId: "gamma-model",
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });

    await harness.jobService.processNextQueuedJob();

    const predictionResponse = await fetch(`${harness.baseUrl}/api/models/gamma-model/prediction-jobs`, {
      body: JSON.stringify({ predictionInput: { inputs: [[9]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const predictionPayload = await predictionResponse.json();
    const jobsResponse = await fetch(`${harness.baseUrl}/api/jobs?modelId=gamma-model`);
    const jobsPayload = (await jobsResponse.json()) as JobRecord[];

    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const statePayload = (await stateResponse.json()) as StatePayload;

    assert.equal(predictionResponse.status, 500);
    assert.deepEqual(predictionPayload, {
      code: "internal_error",
      message: "simulated predict-model failure",
    });
    assert.equal(jobsResponse.status, 200);
    assert.equal(jobsPayload.length, 1);
    assert.equal(statePayload.summary.failedJobCount, 0);
  } finally {
    await harness.close();
  }
});

test("service keeps model metadata across restart", async () => {
  const harness = await createHarness();

  try {
    await createReadyModel(harness, "restart-model", { horizon: "1h", version: 1 });

    const restartedStorageService = new StorageService(join(harness.tempRoot, "storage", "runtime.sqlite"), join(harness.tempRoot, "storage"));
    const restartedModelService = ModelService.create(restartedStorageService);
    restartedStorageService.initialize();

    const restartedModel = restartedModelService.getModel("restart-model");

    assert.ok(restartedModel);
    assert.deepEqual(restartedModel?.metadata, { horizon: "1h", version: 1 });

    restartedStorageService.close();
  } finally {
    await harness.close();
  }
});

test("service rejects invalid sample weight shapes", async () => {
  const harness = await createHarness();

  try {
    await createReadyModel(harness, "invalid-shape-model");

    const invalidSingleOutputResponse = await fetch(`${harness.baseUrl}/api/models/invalid-shape-model/training-jobs`, {
      body: JSON.stringify({
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: [1],
          targets: [[0], [1]],
        },
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const invalidSingleOutputPayload = await invalidSingleOutputResponse.json();
    const invalidMultiOutputResponse = await fetch(`${harness.baseUrl}/api/models/invalid-shape-model/training-jobs`, {
      body: JSON.stringify({
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: [1, 1],
          targets: {
            classification: [
              [1, 0],
              [0, 1],
            ],
            regression: [[0.1], [0.2]],
          },
        },
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const invalidMultiOutputPayload = await invalidMultiOutputResponse.json();

    assert.equal(invalidSingleOutputResponse.status, 400);
    assert.deepEqual(invalidSingleOutputPayload, {
      code: "invalid_request",
      message: "trainingInput.sampleWeights length must match target batch size",
    });
    assert.equal(invalidMultiOutputResponse.status, 400);
    assert.deepEqual(invalidMultiOutputPayload, {
      code: "invalid_request",
      message: "trainingInput.sampleWeights must match the target shape",
    });
  } finally {
    await harness.close();
  }
});

test("service recovers running jobs as failed after restart", async () => {
  const harness = await createHarness();

  try {
    await fetch(`${harness.baseUrl}/api/models`, {
      body: JSON.stringify({
        definition: { format: "keras-sequential", modelConfig: { layers: [] } },
        modelId: "delta-model",
      }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });

    await harness.jobService.processNextQueuedJob();

    const trainingResponse = await fetch(`${harness.baseUrl}/api/models/delta-model/training-jobs`, {
      body: JSON.stringify({ trainingInput: { inputs: [[1]], targets: [[1]] } }),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    const trainingPayload = await trainingResponse.json();
    const runningJob = harness.storageService.claimNextQueuedJob(new Date().toISOString());

    assert.ok(runningJob);

    const restartedStorageService = new StorageService(join(harness.tempRoot, "storage", "runtime.sqlite"), join(harness.tempRoot, "storage"));
    const restartedModelService = ModelService.create(restartedStorageService);
    const restartedPythonRuntimeService = new PythonRuntimeService("python3", "python/tensorflow_api_worker.py", createFakeExecutor());
    const restartedJobService = JobService.create(restartedStorageService, restartedModelService, restartedPythonRuntimeService);
    restartedStorageService.initialize();
    restartedJobService.recoverInterruptedJobs();

    const recoveredJob = restartedJobService.getJob(String(trainingPayload.jobId));

    assert.ok(recoveredJob);
    assert.equal(recoveredJob?.status, "failed");
    assert.equal(recoveredJob?.errorMessage, "job was interrupted by a service restart");

    restartedStorageService.close();
  } finally {
    await harness.close();
  }
});
