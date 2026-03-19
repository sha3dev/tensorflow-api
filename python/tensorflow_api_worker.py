#!/usr/bin/env python3

import json
import sys
from pathlib import Path


FIT_CONFIG_KEY_ALIASES = {
    "batchSize": "batch_size",
    "validationSplit": "validation_split",
}


def write_result(result_path: str, payload: dict) -> None:
    target_path = Path(result_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_stdout(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def load_request(request_path: str) -> dict:
    content = Path(request_path).read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def load_stdin_request() -> dict:
    payload = json.loads(sys.stdin.read())
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return payload


def require_tensorflow():
    try:
        import tensorflow as tensorflow_module
    except ImportError as import_error:
        raise RuntimeError(
            "tensorflow is not installed in the configured Python environment"
        ) from import_error
    return tensorflow_module


def normalize_fit_config(raw_fit_config: dict) -> dict:
    normalized_fit_config = {}

    for key, value in raw_fit_config.items():
        normalized_key = FIT_CONFIG_KEY_ALIASES.get(key, key)
        normalized_fit_config[normalized_key] = value

    return normalized_fit_config


def to_tensor(tf_module, value):
    tensor_value = tf_module.convert_to_tensor(value)
    return tensor_value


def build_validation_data(tf_module, training_input: dict):
    validation_data = None

    if (
        "validationInputs" in training_input
        and "validationTargets" in training_input
    ):
        validation_data = (
            to_tensor(tf_module, training_input["validationInputs"]),
            to_tensor(tf_module, training_input["validationTargets"]),
        )

    return validation_data


def create_model(payload: dict, result_path: str) -> None:
    tf = require_tensorflow()
    definition = payload["definition"]
    artifact_path = payload["artifactPath"]
    model_format = definition["format"]
    model_config = definition["modelConfig"]
    compile_config = definition.get("compileConfig") or {}

    if model_format == "keras-sequential":
        model = tf.keras.Sequential.from_config(model_config)
    elif model_format == "keras-functional":
        model = tf.keras.Model.from_config(model_config)
    else:
        raise ValueError(f"unsupported model format: {model_format}")

    if compile_config:
        model.compile(**compile_config)

    model.save(artifact_path)
    write_result(
        result_path,
        {
            "artifactPath": artifact_path,
            "modelId": payload["modelId"],
            "status": "ready",
        },
    )


def train_model(payload: dict, result_path: str) -> None:
    tf = require_tensorflow()
    artifact_path = payload["artifactPath"]
    training_input = payload["trainingInput"]
    fit_config = normalize_fit_config(payload.get("fitConfig") or {})
    model = tf.keras.models.load_model(artifact_path)
    history = model.fit(
        to_tensor(tf, training_input["inputs"]),
        to_tensor(tf, training_input["targets"]),
        validation_data=build_validation_data(tf, training_input),
        **fit_config,
    )
    model.save(artifact_path)
    write_result(
        result_path,
        {
            "history": history.history,
            "modelId": payload["modelId"],
            "status": "trained",
        },
    )


def predict_model(payload: dict, result_path: str) -> None:
    tf = require_tensorflow()
    artifact_path = payload["artifactPath"]
    prediction_input = payload["predictionInput"]
    model = tf.keras.models.load_model(artifact_path)
    prediction_output = model.predict(to_tensor(tf, prediction_input["inputs"]))
    if hasattr(prediction_output, "tolist"):
        serializable_output = prediction_output.tolist()
    else:
        serializable_output = prediction_output
    write_result(
        result_path,
        {
            "modelId": payload["modelId"],
            "outputs": serializable_output,
            "status": "predicted",
        },
    )


def predict_model_to_stdout(payload: dict) -> None:
    tf = require_tensorflow()
    artifact_path = payload["artifactPath"]
    prediction_input = payload["predictionInput"]
    model = tf.keras.models.load_model(artifact_path)
    prediction_output = model.predict(to_tensor(tf, prediction_input["inputs"]))
    if hasattr(prediction_output, "tolist"):
        serializable_output = prediction_output.tolist()
    else:
        serializable_output = prediction_output
    write_stdout(
        {
            "modelId": payload["modelId"],
            "outputs": serializable_output,
            "status": "predicted",
        }
    )


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "predict-model-stdio":
        try:
            payload = load_stdin_request()
            predict_model_to_stdout(payload)
        except Exception as runtime_error:  # noqa: BLE001
            print(str(runtime_error), file=sys.stderr)
            return 1
        return 0

    if len(sys.argv) != 4:
        print(
            "usage: tensorflow_api_worker.py <create-model|train-model|predict-model> <request-path> <result-path>",
            file=sys.stderr,
        )
        return 1

    action = sys.argv[1]
    request_path = sys.argv[2]
    result_path = sys.argv[3]

    try:
        payload = load_request(request_path)
        if action == "create-model":
            create_model(payload, result_path)
        elif action == "train-model":
            train_model(payload, result_path)
        elif action == "predict-model":
            predict_model(payload, result_path)
        else:
            raise ValueError(f"unsupported action: {action}")
    except Exception as runtime_error:  # noqa: BLE001
        print(str(runtime_error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
