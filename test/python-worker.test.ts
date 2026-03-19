import * as assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

const EXEC_FILE_ASYNC = promisify(execFile);

async function executePredictionOverStdio(tempRoot: string, payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  const result = await new Promise<Record<string, unknown>>((resolve, reject) => {
    let standardError = "";
    let standardOutput = "";
    const childProcess = spawn("python3", ["python/tensorflow_api_worker.py", "predict-model-stdio"], {
      cwd: "/Users/jc/Documents/GitHub/tensorflow-api",
      env: { ...process.env, PYTHONPATH: tempRoot },
    });

    childProcess.stdout.on("data", (chunk: Buffer) => {
      standardOutput += String(chunk);
    });
    childProcess.stderr.on("data", (chunk: Buffer) => {
      standardError += String(chunk);
    });
    childProcess.on("close", (exitCode: number | null) => {
      if (exitCode !== 0) {
        reject(new Error(standardError.trim() || "python worker execution failed"));
      } else {
        resolve(JSON.parse(standardOutput) as Record<string, unknown>);
      }
    });
    childProcess.on("error", (error: Error) => {
      reject(error);
    });
    childProcess.stdin.write(JSON.stringify(payload));
    childProcess.stdin.end();
  });

  return result;
}

function createFakeTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `import json
from pathlib import Path

class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def convert_to_tensor(value):
    return {"converted": True, "value": value}


class FakeHistory:
    def __init__(self, history):
        self.history = history


class FakeModel:
    def __init__(self, model_kind, config):
        self.model_kind = model_kind
        self.config = config
        self.compile_config = {}

    def compile(self, **kwargs):
        self.compile_config = kwargs

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "compile_config": self.compile_config,
                    "config": self.config,
                    "model_kind": self.model_kind,
                }
            ),
            encoding="utf-8",
        )

    def fit(self, inputs, targets, validation_data=None, **kwargs):
        return FakeHistory(
            {
                "batch_size": [kwargs.get("batch_size")],
                "input_is_converted": [inputs.get("converted")],
                "target_is_converted": [targets.get("converted")],
                "validation_split": [kwargs.get("validation_split")],
            }
        )

    def predict(self, inputs):
        return FakeArray([inputs.get("converted"), inputs.get("value")])


class Sequential:
    @staticmethod
    def from_config(config):
        return FakeModel("sequential", config)


class Model:
    @staticmethod
    def from_config(config):
        return FakeModel("functional", config)


class ModelsApi:
    @staticmethod
    def load_model(artifact_path):
        saved_model = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        model = FakeModel(saved_model["model_kind"], saved_model["config"])
        model.compile_config = saved_model.get("compile_config", {})
        return model


class KerasApi:
    Sequential = Sequential
    Model = Model
    models = ModelsApi()


keras = KerasApi()
`;
  writeFileSync(join(moduleRootPath, "tensorflow.py"), moduleSource, "utf8");
}

test("python worker normalizes fit config keys and converts training arrays", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFakeTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ compile_config: {}, config: { layers: [] }, model_kind: "sequential" }), "utf8");
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        fitConfig: { batchSize: 4, epochs: 2, validationSplit: 0.5 },
        modelId: "worker-model",
        trainingInput: {
          inputs: [
            [1, 2],
            [3, 4],
          ],
          targets: [[0], [1]],
        },
      }),
      "utf8",
    );

    await EXEC_FILE_ASYNC("python3", ["python/tensorflow_api_worker.py", "train-model", requestPath, resultPath], {
      cwd: "/Users/jc/Documents/GitHub/tensorflow-api",
      env: { ...process.env, PYTHONPATH: tempRoot },
    });

    const resultPayload = JSON.parse(readFileSync(resultPath, "utf8")) as {
      history: Record<string, unknown[]>;
      status: string;
    };

    assert.equal(resultPayload.status, "trained");
    assert.deepEqual(resultPayload.history.batch_size, [4]);
    assert.deepEqual(resultPayload.history.validation_split, [0.5]);
    assert.deepEqual(resultPayload.history.input_is_converted, [true]);
    assert.deepEqual(resultPayload.history.target_is_converted, [true]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker converts prediction arrays before calling predict over stdio", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFakeTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ compile_config: {}, config: { layers: [] }, model_kind: "sequential" }), "utf8");
    const resultPayload = (await executePredictionOverStdio(tempRoot, {
      artifactPath,
      modelId: "worker-model",
      predictionInput: {
        inputs: [
          [5, 6],
          [7, 8],
        ],
      },
    })) as {
      outputs: unknown[];
      status: string;
    };

    assert.equal(resultPayload.status, "predicted");
    assert.deepEqual(resultPayload.outputs, [
      true,
      [
        [5, 6],
        [7, 8],
      ],
    ]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker keeps file-based prediction mode for queued jobs", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "predict-request.json");
  const resultPath = join(tempRoot, "predict-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFakeTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ compile_config: {}, config: { layers: [] }, model_kind: "sequential" }), "utf8");
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        modelId: "worker-model",
        predictionInput: {
          inputs: [
            [5, 6],
            [7, 8],
          ],
        },
      }),
      "utf8",
    );

    await EXEC_FILE_ASYNC("python3", ["python/tensorflow_api_worker.py", "predict-model", requestPath, resultPath], {
      cwd: "/Users/jc/Documents/GitHub/tensorflow-api",
      env: { ...process.env, PYTHONPATH: tempRoot },
    });

    const resultPayload = JSON.parse(readFileSync(resultPath, "utf8")) as {
      outputs: unknown[];
      status: string;
    };

    assert.equal(resultPayload.status, "predicted");
    assert.deepEqual(resultPayload.outputs, [
      true,
      [
        [5, 6],
        [7, 8],
      ],
    ]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});
