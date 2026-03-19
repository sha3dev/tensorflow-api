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
  failingAction?: PythonJobCommand["action"];
};

type StatePayload = {
  models: Array<{
    predictionCount: number;
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
      return { errorMessage: `simulated ${command.action} failure`, isSuccess: false };
    }

    if (command.action === "create-model") {
      writeFileSync(String(payload.artifactPath), JSON.stringify({ created: true }), "utf8");
      writeFileSync(command.resultPath, JSON.stringify({ artifactPath: payload.artifactPath, modelId: payload.modelId, status: "ready" }), "utf8");
    }

    if (command.action === "train-model") {
      writeFileSync(command.resultPath, JSON.stringify({ history: { loss: [0.5, 0.2] }, modelId: payload.modelId, status: "trained" }), "utf8");
    }

    if (command.action === "predict-model") {
      const predictionInput = payload.predictionInput as { inputs?: unknown };
      writeFileSync(command.resultPath, JSON.stringify({ modelId: payload.modelId, outputs: predictionInput.inputs || [], status: "predicted" }), "utf8");
    }

    return { errorMessage: null, isSuccess: true };
  };
}

function createFakePredictionExecutor(options?: FakeExecutorOptions): PythonPredictionExecutor {
  return async (command: PythonPredictionCommand) => {
    if (options?.failingAction === "predict-model") {
      return { errorMessage: "simulated predict-model failure", isSuccess: false };
    }

    return {
      isSuccess: true,
      result: {
        modelId: command.modelId,
        outputs: command.predictionInput.inputs,
        status: "predicted",
      },
    };
  };
}

async function waitForPredictionJob(harness: TestHarness, modelId: string): Promise<JobRecord> {
  let predictionJob: JobRecord | undefined;

  for (let attemptIndex = 0; attemptIndex < 20; attemptIndex += 1) {
    const jobsResponse = await fetch(`${harness.baseUrl}/api/jobs?modelId=${encodeURIComponent(modelId)}`);
    const jobsPayload = (await jobsResponse.json()) as JobRecord[];
    predictionJob = jobsPayload.find((jobRecord) => {
      return jobRecord.jobType === "predict_model";
    });

    if (predictionJob) {
      break;
    }

    await new Promise<void>((resolve) => {
      setTimeout(resolve, 10);
    });
  }

  assert.ok(predictionJob);
  return predictionJob;
}

async function createHarness(options?: FakeExecutorOptions): Promise<TestHarness> {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-api-"));
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

    assert.equal(createResponse.status, 201);
    assert.equal(createPayload.kind, "created");
    assert.equal(createPayload.model.status, "pending");
    assert.equal(duplicateResponse.status, 409);
    assert.deepEqual(duplicatePayload, {
      code: "conflict",
      message: "model 'alpha-model' already exists",
    });
    assert.equal(statePayload.models.length, 1);
    assert.equal(statePayload.summary.modelCount, 1);
    assert.equal(statePayload.summary.queuedJobCount, 1);
  } finally {
    await harness.close();
  }
});

test("service processes create, train, and prediction jobs and exposes results", async () => {
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

    const predictionJob = await waitForPredictionJob(harness, "beta-model");
    const predictionResultResponse = await fetch(`${harness.baseUrl}/api/jobs/${predictionJob?.jobId}/result`);
    const predictionResultPayload = await predictionResultResponse.json();
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
    assert.equal(predictionResultResponse.status, 200);
    assert.deepEqual(predictionResultPayload, {
      modelId: "beta-model",
      outputs: [[3], [4]],
      status: "predicted",
    });
    assert.equal(jobsResponse.status, 200);
    assert.equal(jobsPayload.length, 3);
    assert.equal(predictionJob?.status, "succeeded");
    assert.equal(creationJobResponse.status, 200);
    assert.equal(creationJobPayload.status, "succeeded");
    assert.equal(statePayload.summary.modelCount, 1);
    assert.equal(statePayload.summary.queuedJobCount, 0);
    assert.equal(statePayload.models.length, 1);
    assert.equal(statePayload.models[0]?.trainingCount, 1);
    assert.equal(statePayload.models[0]?.predictionCount, 1);
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
    const failedPredictionJob = await waitForPredictionJob(harness, "gamma-model");
    const jobsResponse = await fetch(`${harness.baseUrl}/api/jobs?modelId=gamma-model`);

    const failedJobResponse = await fetch(`${harness.baseUrl}/api/jobs/${failedPredictionJob?.jobId}`);
    const failedJobPayload = await failedJobResponse.json();
    const stateResponse = await fetch(`${harness.baseUrl}/api/state`);
    const statePayload = (await stateResponse.json()) as StatePayload;

    assert.equal(predictionResponse.status, 500);
    assert.deepEqual(predictionPayload, {
      code: "internal_error",
      message: "simulated predict-model failure",
    });
    assert.equal(jobsResponse.status, 200);
    assert.equal(failedJobResponse.status, 200);
    assert.equal(failedJobPayload.status, "failed");
    assert.equal(failedJobPayload.errorCode, "internal_error");
    assert.equal(failedJobPayload.errorMessage, "simulated predict-model failure");
    assert.equal(statePayload.summary.failedJobCount, 1);
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
