import * as assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { PythonRuntimeService } from "../src/python-runtime/index.ts";

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

async function executeTrainingFailure(tempRoot: string, payload: Record<string, unknown>): Promise<string> {
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  let standardError = "";

  writeFileSync(requestPath, JSON.stringify(payload), "utf8");

  await new Promise<void>((resolve) => {
    const childProcess = spawn("python3", ["python/tensorflow_api_worker.py", "train-model", requestPath, resultPath], {
      cwd: "/Users/jc/Documents/GitHub/tensorflow-api",
      env: { ...process.env, PYTHONPATH: tempRoot },
    });

    childProcess.stderr.on("data", (chunk: Buffer) => {
      standardError += String(chunk);
    });
    childProcess.on("close", () => {
      resolve();
    });
  });

  return standardError.trim();
}

function createFakeTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `__version__ = "fake-tf-1.0.0"

import json
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
        self.output_names = config.get("output_names", ["output"])

    def compile(self, **kwargs):
        self.compile_config = kwargs

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "compile_config": self.compile_config,
                    "config": self.config,
                    "model_kind": self.model_kind,
                    "output_names": self.output_names,
                }
            ),
            encoding="utf-8",
        )

    def fit(self, inputs, targets, validation_data=None, sample_weight=None, **kwargs):
        validation_sample_weight = (
            validation_data[2] if validation_data is not None and len(validation_data) == 3 else None
        )
        validation_targets = (
            validation_data[1] if validation_data is not None and len(validation_data) >= 2 else None
        )
        target_is_converted = targets.get("converted") if isinstance(targets, dict) else None
        return FakeHistory(
            {
                "batch_size": [kwargs.get("batch_size")],
                "input_is_converted": [inputs.get("converted")],
                "sample_weight": [sample_weight],
                "target_is_converted": [target_is_converted],
                "targets": [targets],
                "validation_targets": [validation_targets],
                "validation_sample_weight": [validation_sample_weight],
                "validation_split": [kwargs.get("validation_split")],
            }
        )

    def predict(self, inputs, verbose=1):
        if len(self.output_names) > 1:
            return [
                FakeArray([output_name, inputs.get("value")])
                for output_name in self.output_names
            ]
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
        model.output_names = saved_model.get("output_names", ["output"])
        return model


class KerasApi:
    Sequential = Sequential
    Model = Model
    models = ModelsApi()


keras = KerasApi()
`;
  writeFileSync(join(moduleRootPath, "tensorflow.py"), moduleSource, "utf8");
}

function createFailingTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `__version__ = "fake-tf-1.0.0"

import json
from pathlib import Path


def convert_to_tensor(value):
    return value


class FakeModel:
    def __init__(self, model_kind, config):
        self.model_kind = model_kind
        self.config = config
        self.output_names = config.get("output_names", ["output"])

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "config": self.config,
                    "model_kind": self.model_kind,
                    "output_names": self.output_names,
                }
            ),
            encoding="utf-8",
        )

    def fit(self, inputs, targets, validation_data=None, sample_weight=None, **kwargs):
        raise KeyError(0)


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
        model.output_names = saved_model.get("output_names", ["output"])
        return model


class KerasApi:
    Sequential = Sequential
    Model = Model
    models = ModelsApi()


keras = KerasApi()
`;
  writeFileSync(join(moduleRootPath, "tensorflow.py"), moduleSource, "utf8");
}

function createCompileAwareTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `__version__ = "fake-tf-1.0.0"

import json
from pathlib import Path


def convert_to_tensor(value):
    return {"converted": True, "value": value}


class FakeModel:
    def __init__(self, model_kind, config):
        self.model_kind = model_kind
        self.config = config
        self.compile_config = {}
        self.output_names = config.get("output_names", ["output"])

    def compile(self, **kwargs):
        metrics = kwargs.get("metrics")

        if len(self.output_names) > 1 and isinstance(metrics, list) and len(metrics) == 0:
            raise ValueError("For a model with multiple outputs, when providing the metrics argument as a list, it should have as many entries as the model has outputs")

        self.compile_config = kwargs

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "compile_config": self.compile_config,
                    "config": self.config,
                    "model_kind": self.model_kind,
                    "output_names": self.output_names,
                }
            ),
            encoding="utf-8",
        )


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
        model.output_names = saved_model.get("output_names", ["output"])
        return model


class KerasApi:
    Sequential = Sequential
    Model = Model
    models = ModelsApi()


keras = KerasApi()
`;
  writeFileSync(join(moduleRootPath, "tensorflow.py"), moduleSource, "utf8");
}

function createNoisyPredictionTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `__version__ = "fake-tf-1.0.0"

import json
import sys
from pathlib import Path


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def convert_to_tensor(value):
    return {"converted": True, "value": value}


class FakeModel:
    def __init__(self, model_kind, config):
        self.model_kind = model_kind
        self.config = config
        self.output_names = config.get("output_names", ["output"])

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "config": self.config,
                    "model_kind": self.model_kind,
                    "output_names": self.output_names,
                }
            ),
            encoding="utf-8",
        )

    def predict(self, inputs, verbose=1):
        sys.stdout.write("\\u001b[1m1/1\\u001b[0m fake-progress\\n")
        sys.stdout.flush()
        return FakeArray([inputs.get("converted"), inputs.get("value"), verbose])


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
        model.output_names = saved_model.get("output_names", ["output"])
        return model


class KerasApi:
    Sequential = Sequential
    Model = Model
    models = ModelsApi()


keras = KerasApi()
`;
  writeFileSync(join(moduleRootPath, "tensorflow.py"), moduleSource, "utf8");
}

function createStrictInputTensorflowModule(moduleRootPath: string): void {
  const moduleSource = `__version__ = "fake-tf-1.0.0"

import json
from pathlib import Path


def _flatten_scalars(value):
    if isinstance(value, (list, tuple)):
        flattened = []
        for nested_value in value:
            flattened.extend(_flatten_scalars(nested_value))
        return flattened
    return [value]


def convert_to_tensor(value):
    flattened = _flatten_scalars(value)
    scalar_types = {type(nested_value).__name__ for nested_value in flattened}

    if "str" in scalar_types and len(scalar_types) > 1:
        raise ValueError("Can't convert Python sequence with mixed types to Tensor")

    if "int" in scalar_types and "float" in scalar_types:
        raise ValueError("Can't convert Python sequence with mixed types to Tensor")

    return {"converted": True, "value": value}


class FakeHistory:
    def __init__(self, history):
        self.history = history


class FakeModel:
    def __init__(self, model_kind, config):
        self.model_kind = model_kind
        self.config = config
        self.output_names = config.get("output_names", ["output"])

    def save(self, artifact_path):
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "config": self.config,
                    "model_kind": self.model_kind,
                    "output_names": self.output_names,
                }
            ),
            encoding="utf-8",
        )

    def fit(self, inputs, targets, validation_data=None, sample_weight=None, **kwargs):
        return FakeHistory(
            {
                "inputs": [inputs],
                "targets": [targets],
            }
        )


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
        model.output_names = saved_model.get("output_names", ["output"])
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
    assert.deepEqual(resultPayload.history.sample_weight, [null]);
    assert.deepEqual(resultPayload.history.target_is_converted, [true]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker normalizes mixed numeric training inputs before tensor conversion", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createStrictInputTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ config: { layers: [] }, model_kind: "functional", output_names: ["regression", "classification"] }), "utf8");
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        modelId: "worker-model",
        trainingInput: {
          inputs: [
            [1, 2.5],
            [3, 4.25],
          ],
          sampleWeights: {
            classification: [2, 0.5],
            regression: [1, 1],
          },
          targets: {
            classification: [
              [1, 0],
              [0, 1],
            ],
            regression: [[0.1], [0.2]],
          },
          validationInputs: [[5, 6.5]],
          validationSampleWeights: {
            classification: [0.75],
            regression: [0.25],
          },
          validationTargets: {
            classification: [[1, 0]],
            regression: [[0.3]],
          },
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
    assert.deepEqual(resultPayload.history.inputs, [
      {
        converted: true,
        value: [
          [1.0, 2.5],
          [3.0, 4.25],
        ],
      },
    ]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker omits empty metrics list for multi-output model creation", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "create-request.json");
  const resultPath = join(tempRoot, "create-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createCompileAwareTensorflowModule(tempRoot);
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        definition: {
          compileConfig: {
            loss: {
              classification: "categorical_crossentropy",
              regression: "mse",
            },
            metrics: [],
            optimizer: "adam",
          },
          format: "keras-functional",
          modelConfig: {
            layers: [],
            output_names: ["regression", "classification"],
          },
        },
        modelId: "worker-model",
      }),
      "utf8",
    );

    await EXEC_FILE_ASYNC("python3", ["python/tensorflow_api_worker.py", "create-model", requestPath, resultPath], {
      cwd: "/Users/jc/Documents/GitHub/tensorflow-api",
      env: { ...process.env, PYTHONPATH: tempRoot },
    });

    const resultPayload = JSON.parse(readFileSync(resultPath, "utf8")) as {
      status: string;
    };
    const artifactPayload = JSON.parse(readFileSync(artifactPath, "utf8")) as {
      compile_config: Record<string, unknown>;
    };

    assert.equal(resultPayload.status, "ready");
    assert.equal("metrics" in artifactPayload.compile_config, false);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker passes single-output sample weights into fit", async () => {
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
        modelId: "worker-model",
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: [1, 0.5],
          targets: [[0], [1]],
          validationInputs: [[3]],
          validationSampleWeights: [0.25],
          validationTargets: [[1]],
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
    };

    assert.deepEqual(resultPayload.history.sample_weight, [{ converted: true, value: [1, 0.5] }]);
    assert.deepEqual(resultPayload.history.validation_sample_weight, [{ converted: true, value: [0.25] }]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker passes multi-output sample weights into fit", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFakeTensorflowModule(tempRoot);
    writeFileSync(
      artifactPath,
      JSON.stringify({ compile_config: {}, config: { layers: [] }, model_kind: "functional", output_names: ["regression", "classification"] }),
      "utf8",
    );
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        modelId: "worker-model",
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: {
            classification: [2, 0.5],
            regression: [1, 1],
          },
          targets: {
            classification: [
              [1, 0],
              [0, 1],
            ],
            regression: [[0.1], [0.2]],
          },
          validationInputs: [[3]],
          validationSampleWeights: {
            classification: [0.75],
            regression: [0.25],
          },
          validationTargets: {
            classification: [[1, 0]],
            regression: [[0.3]],
          },
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
    };

    assert.deepEqual(resultPayload.history.sample_weight, [
      [
        { converted: true, value: [1, 1] },
        { converted: true, value: [2, 0.5] },
      ],
    ]);
    assert.deepEqual(resultPayload.history.targets, [
      [
        { converted: true, value: [[0.1], [0.2]] },
        {
          converted: true,
          value: [
            [1, 0],
            [0, 1],
          ],
        },
      ],
    ]);
    assert.deepEqual(resultPayload.history.validation_sample_weight, [
      [
        { converted: true, value: [0.25] },
        { converted: true, value: [0.75] },
      ],
    ]);
    assert.deepEqual(resultPayload.history.validation_targets, [
      [
        { converted: true, value: [[0.3]] },
        {
          converted: true,
          value: [[1, 0]],
        },
      ],
    ]);
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

test("python worker maps multi-output predictions by output name", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFakeTensorflowModule(tempRoot);
    writeFileSync(
      artifactPath,
      JSON.stringify({ compile_config: {}, config: { layers: [] }, model_kind: "functional", output_names: ["regression", "classification"] }),
      "utf8",
    );
    const resultPayload = (await executePredictionOverStdio(tempRoot, {
      artifactPath,
      modelId: "worker-model",
      predictionInput: {
        inputs: [[5, 6]],
      },
    })) as {
      outputs: Record<string, unknown>;
      status: string;
    };

    assert.equal(resultPayload.status, "predicted");
    assert.deepEqual(resultPayload.outputs, {
      classification: ["classification", [[5, 6]]],
      regression: ["regression", [[5, 6]]],
    });
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker prediction stdio stays valid JSON when predict writes ANSI progress", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createNoisyPredictionTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ config: { layers: [] }, model_kind: "sequential" }), "utf8");
    const resultPayload = (await executePredictionOverStdio(tempRoot, {
      artifactPath,
      modelId: "worker-model",
      predictionInput: {
        inputs: [[5, 6]],
      },
    })) as {
      outputs: unknown[];
      status: string;
    };

    assert.equal(resultPayload.status, "predicted");
    assert.deepEqual(resultPayload.outputs, [true, [[5, 6]], 0]);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python runtime verification succeeds when tensorflow is importable", () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const previousPythonPath = process.env.PYTHONPATH;

  try {
    createFakeTensorflowModule(tempRoot);
    process.env.PYTHONPATH = tempRoot;
    const pythonRuntimeService = new PythonRuntimeService("python3", "python/tensorflow_api_worker.py");

    pythonRuntimeService.verifyRuntime();
  } finally {
    process.env.PYTHONPATH = previousPythonPath;
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python runtime verification reports the configured python binary on failure", () => {
  const pythonRuntimeService = new PythonRuntimeService("python-does-not-exist", "python/tensorflow_api_worker.py");

  assert.throws(() => {
    pythonRuntimeService.verifyRuntime();
  }, /python runtime check failed.*python-does-not-exist/);
});

test("python worker reports traceback details for training failures", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createFailingTensorflowModule(tempRoot);
    writeFileSync(
      artifactPath,
      JSON.stringify({
        config: { layers: [], output_names: ["regression", "classification"] },
        model_kind: "sequential",
        output_names: ["regression", "classification"],
      }),
      "utf8",
    );
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        modelId: "worker-model",
        trainingInput: {
          inputs: [[1], [2]],
          sampleWeights: {
            classification: [2, 0.5],
            regression: [1, 1],
          },
          targets: {
            classification: [
              [1, 0],
              [0, 1],
            ],
            regression: [[0.1], [0.2]],
          },
          validationSampleWeights: {
            classification: [0.75],
            regression: [0.25],
          },
          validationTargets: {
            classification: [[1, 0]],
            regression: [[0.3]],
          },
        },
      }),
      "utf8",
    );

    const standardError = await executeTrainingFailure(tempRoot, JSON.parse(readFileSync(requestPath, "utf8")) as Record<string, unknown>);
    const resultPayload = JSON.parse(readFileSync(resultPath, "utf8")) as Record<string, unknown>;

    assert.match(standardError, /Traceback \(most recent call last\):/);
    assert.match(standardError, /KeyError: 0/);
    assert.equal(resultPayload.status, "failed");
    assert.equal(resultPayload.modelId, "worker-model");
    assert.equal(resultPayload.errorCode, "internal_error");
    assert.equal((resultPayload.diagnostics as Record<string, unknown>).modelOutputCount, 2);
    assert.deepEqual((resultPayload.diagnostics as Record<string, unknown>).modelOutputNames, ["regression", "classification"]);
    assert.equal((resultPayload.diagnostics as Record<string, unknown>).pythonExceptionType, "KeyError");
    assert.match(String((resultPayload.diagnostics as Record<string, unknown>).traceback), /KeyError: 0/);
    assert.deepEqual(((resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>).inputShape, [2, 1]);
    assert.equal(((resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>).inputTypes, "int");
    assert.deepEqual(((resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>).targetKeys, [
      "classification",
      "regression",
    ]);
    assert.deepEqual(((resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>).sampleWeightKeys, [
      "classification",
      "regression",
    ]);
    assert.deepEqual(((resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>).targetShapes, {
      classification: [2, 2],
      regression: [2, 1],
    });
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});

test("python worker includes input diagnostics when input tensor conversion fails", async () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "tensorflow-worker-test-"));
  const requestPath = join(tempRoot, "train-request.json");
  const resultPath = join(tempRoot, "train-result.json");
  const artifactPath = join(tempRoot, "artifact.keras");

  try {
    createStrictInputTensorflowModule(tempRoot);
    writeFileSync(artifactPath, JSON.stringify({ config: { layers: [] }, model_kind: "sequential" }), "utf8");
    writeFileSync(
      requestPath,
      JSON.stringify({
        artifactPath,
        modelId: "worker-model",
        trainingInput: {
          inputs: [
            [1, "bad"],
            [3, 4],
          ],
          targets: [[0], [1]],
          validationInputs: [[5, "bad"]],
          validationTargets: [[1]],
        },
      }),
      "utf8",
    );

    const standardError = await executeTrainingFailure(tempRoot, JSON.parse(readFileSync(requestPath, "utf8")) as Record<string, unknown>);
    const resultPayload = JSON.parse(readFileSync(resultPath, "utf8")) as Record<string, unknown>;
    const trainingInputSummary = (resultPayload.diagnostics as Record<string, unknown>).trainingInputSummary as Record<string, unknown>;

    assert.match(standardError, /ValueError: Can't convert Python sequence with mixed types to Tensor/);
    assert.equal(resultPayload.status, "failed");
    assert.equal((resultPayload.diagnostics as Record<string, unknown>).pythonExceptionType, "ValueError");
    assert.deepEqual(trainingInputSummary.inputShape, [2, 2]);
    assert.equal(trainingInputSummary.inputTypes, "int|str");
    assert.deepEqual(trainingInputSummary.validationInputShape, [1, 2]);
    assert.equal(trainingInputSummary.validationInputTypes, "int|str");
    assert.equal(JSON.stringify(trainingInputSummary).includes('"bad"'), false);
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
});
