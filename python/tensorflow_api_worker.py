#!/usr/bin/env python3

import contextlib
import io
import json
import numbers
import sys
import traceback
from pathlib import Path


FIT_CONFIG_KEY_ALIASES = {
    "batchSize": "batch_size",
    "validationSplit": "validation_split",
}

TYPE_SAMPLE_LIMIT = 25


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


def format_runtime_error(runtime_error: Exception) -> str:
    error_lines = traceback.format_exception(
        type(runtime_error), runtime_error, runtime_error.__traceback__
    )
    formatted_error = "".join(error_lines).strip()
    return formatted_error


def normalize_fit_config(raw_fit_config: dict) -> dict:
    normalized_fit_config = {}

    for key, value in raw_fit_config.items():
        normalized_key = FIT_CONFIG_KEY_ALIASES.get(key, key)
        normalized_fit_config[normalized_key] = value

    return normalized_fit_config


def normalize_compile_config(model, raw_compile_config: dict) -> dict:
    normalized_compile_config = dict(raw_compile_config)
    output_names = list(getattr(model, "output_names", []) or [])
    metrics = normalized_compile_config.get("metrics")

    if (
        isinstance(metrics, list)
        and len(metrics) == 0
        and len(output_names) > 1
    ):
        normalized_compile_config.pop("metrics")

    return normalized_compile_config


def to_tensor(tf_module, value):
    tensor_value = tf_module.convert_to_tensor(normalize_numeric_structure(value))
    return tensor_value


def to_tensor_structure(tf_module, value):
    tensor_structure = None

    if isinstance(value, dict):
        tensor_structure = {}

        for key, nested_value in value.items():
            tensor_structure[key] = to_tensor_structure(tf_module, nested_value)
    else:
        tensor_structure = to_tensor(tf_module, value)

    return tensor_structure


def normalize_numeric_structure(value):
    normalized_value = value

    if isinstance(value, dict):
        normalized_value = {
            key: normalize_numeric_structure(nested_value)
            for key, nested_value in value.items()
        }
    elif isinstance(value, (list, tuple)):
        normalized_sequence = [
            normalize_numeric_structure(nested_value) for nested_value in value
        ]
        scalar_types = {
            type(nested_value).__name__
            for nested_value in normalized_sequence
            if isinstance(nested_value, numbers.Real)
        }
        if (
            len(normalized_sequence) > 0
            and len(scalar_types) > 1
            and all(
                isinstance(nested_value, numbers.Real)
                for nested_value in normalized_sequence
            )
        ):
            normalized_value = [float(nested_value) for nested_value in normalized_sequence]
        else:
            normalized_value = normalized_sequence

    return normalized_value


def to_serializable_output(value):
    serializable_value = value

    if hasattr(value, "tolist"):
        serializable_value = value.tolist()
    elif isinstance(value, dict):
        serializable_value = {}

        for key, nested_value in value.items():
            serializable_value[key] = to_serializable_output(nested_value)
    elif isinstance(value, (list, tuple)):
        serializable_value = [to_serializable_output(nested_value) for nested_value in value]

    return serializable_value


def summarize_shape(value):
    shape_summary = []

    if isinstance(value, dict):
        shape_summary = {
            key: summarize_shape(nested_value) for key, nested_value in value.items()
        }
    elif isinstance(value, list):
        if len(value) > 0:
            nested_shape = summarize_shape(value[0])
            shape_summary = [len(value), *nested_shape]
        else:
            shape_summary = [0]

    return shape_summary


def get_structure_keys(value):
    structure_keys = list(value.keys()) if isinstance(value, dict) else []
    return structure_keys


def summarize_scalar_type(value):
    type_name = type(value).__name__

    if isinstance(value, bool):
        type_name = "bool"
    elif isinstance(value, int):
        type_name = "int"
    elif isinstance(value, float):
        type_name = "float"
    elif value is None:
        type_name = "null"

    return type_name


def combine_type_names(type_names):
    flattened_type_names = []

    for type_name in type_names:
        flattened_type_names.extend(type_name.split("|"))

    combined_type_names = list(dict.fromkeys(flattened_type_names))
    type_summary = "|".join(combined_type_names)
    return type_summary


def summarize_types(value):
    type_summary = summarize_scalar_type(value)

    if isinstance(value, dict):
        type_summary = {
            key: summarize_types(nested_value) for key, nested_value in value.items()
        }
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            type_summary = "empty"
        else:
            nested_type_names = [
                summarize_types(nested_value) for nested_value in value[:TYPE_SAMPLE_LIMIT]
            ]
            if all(isinstance(type_name, str) for type_name in nested_type_names):
                type_summary = combine_type_names(nested_type_names)
            else:
                type_summary = combine_type_names(
                    [
                        json.dumps(type_name, sort_keys=True)
                        for type_name in nested_type_names
                    ]
                )

    return type_summary


def build_training_input_summary(training_input: dict):
    input_summary = {
        "inputShape": summarize_shape(training_input.get("inputs"))
        if "inputs" in training_input
        else None,
        "inputTypes": summarize_types(training_input.get("inputs"))
        if "inputs" in training_input
        else None,
        "targetKeys": get_structure_keys(training_input.get("targets")),
        "targetShapes": summarize_shape(training_input.get("targets"))
        if "targets" in training_input
        else None,
        "sampleWeightKeys": get_structure_keys(training_input.get("sampleWeights")),
        "sampleWeightShapes": summarize_shape(training_input.get("sampleWeights"))
        if "sampleWeights" in training_input
        else None,
        "validationTargetKeys": get_structure_keys(training_input.get("validationTargets")),
        "validationTargetShapes": summarize_shape(
            training_input.get("validationTargets")
        )
        if "validationTargets" in training_input
        else None,
        "validationInputShape": summarize_shape(training_input.get("validationInputs"))
        if "validationInputs" in training_input
        else None,
        "validationInputTypes": summarize_types(training_input.get("validationInputs"))
        if "validationInputs" in training_input
        else None,
        "validationSampleWeightKeys": get_structure_keys(
            training_input.get("validationSampleWeights")
        ),
        "validationSampleWeightShapes": summarize_shape(
            training_input.get("validationSampleWeights")
        )
        if "validationSampleWeights" in training_input
        else None,
    }
    return input_summary


def build_training_diagnostics(model, training_input: dict):
    model_output_names = list(getattr(model, "output_names", []) or [])
    diagnostics = {
        "modelOutputCount": len(model_output_names),
        "modelOutputNames": model_output_names,
        "pythonExceptionType": "",
        "stderrTail": "",
        "traceback": "",
        "trainingInputSummary": build_training_input_summary(training_input),
    }
    return diagnostics


def build_prediction_outputs(model, prediction_output):
    outputs = None

    if isinstance(prediction_output, dict):
        outputs = {key: to_serializable_output(value) for key, value in prediction_output.items()}
    else:
        output_names = getattr(model, "output_names", None)

        if (
            output_names
            and isinstance(prediction_output, (list, tuple))
            and len(output_names) == len(prediction_output)
        ):
            outputs = {}

            for index, output_name in enumerate(output_names):
                outputs[output_name] = to_serializable_output(prediction_output[index])
        else:
            outputs = to_serializable_output(prediction_output)

    return outputs


def align_named_output_structure(model, value):
    aligned_value = value

    if isinstance(value, dict):
        output_names = list(getattr(model, "output_names", []) or [])

        if output_names:
            aligned_value = [value[output_name] for output_name in output_names]

    return aligned_value


def build_output_tensor_structure(tf_module, model, value):
    tensor_structure = None
    aligned_value = align_named_output_structure(model, value)

    if isinstance(value, dict):
        tensor_structure = [
            to_tensor_structure(tf_module, nested_value)
            for nested_value in aligned_value
        ]
    else:
        tensor_structure = to_tensor_structure(tf_module, aligned_value)

    return tensor_structure


def build_fit_targets(tf_module, model, training_input: dict):
    tensor_targets = build_output_tensor_structure(
        tf_module, model, training_input["targets"]
    )
    return tensor_targets


def build_sample_weights(tf_module, model, training_input: dict, field_name: str):
    sample_weights = None

    if field_name in training_input:
        sample_weights = build_output_tensor_structure(
            tf_module, model, training_input[field_name]
        )

    return sample_weights


def build_validation_data(tf_module, model, training_input: dict):
    validation_data = None

    if (
        "validationInputs" in training_input
        and "validationTargets" in training_input
    ):
        validation_sample_weights = build_sample_weights(
            tf_module, model, training_input, "validationSampleWeights"
        )
        validation_targets = build_output_tensor_structure(
            tf_module, model, training_input["validationTargets"]
        )
        validation_data = (
            to_tensor_structure(tf_module, training_input["validationInputs"]),
            validation_targets,
            validation_sample_weights,
        ) if validation_sample_weights is not None else (
            to_tensor_structure(tf_module, training_input["validationInputs"]),
            validation_targets,
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

    compile_config = normalize_compile_config(model, compile_config)

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
    diagnostics = build_training_diagnostics(model, training_input)

    try:
        tensor_inputs = to_tensor_structure(tf, training_input["inputs"])
        tensor_targets = build_fit_targets(tf, model, training_input)
        sample_weight = build_sample_weights(tf, model, training_input, "sampleWeights")
        validation_data = build_validation_data(tf, model, training_input)
        history = model.fit(
            tensor_inputs,
            tensor_targets,
            sample_weight=sample_weight,
            validation_data=validation_data,
            **fit_config,
        )
    except Exception as runtime_error:
        traceback_text = format_runtime_error(runtime_error)
        write_result(
            result_path,
            {
                "diagnostics": {
                    **diagnostics,
                    "pythonExceptionType": type(runtime_error).__name__,
                    "traceback": traceback_text,
                },
                "errorCode": "internal_error",
                "errorMessage": traceback_text,
                "modelId": payload["modelId"],
                "status": "failed",
            },
        )
        raise

    model.save(artifact_path)
    write_result(
        result_path,
        {
            "history": history.history,
            "modelId": payload["modelId"],
            "status": "trained",
        },
    )


def predict_model_to_stdout(payload: dict) -> None:
    tf = require_tensorflow()
    artifact_path = payload["artifactPath"]
    prediction_input = payload["predictionInput"]
    model = tf.keras.models.load_model(artifact_path)

    with contextlib.redirect_stdout(io.StringIO()):
        prediction_output = model.predict(
            to_tensor_structure(tf, prediction_input["inputs"]), verbose=0
        )

    write_stdout(
        {
            "modelId": payload["modelId"],
            "outputs": build_prediction_outputs(model, prediction_output),
            "status": "predicted",
        }
    )


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "predict-model-stdio":
        try:
            payload = load_stdin_request()
            predict_model_to_stdout(payload)
        except Exception as runtime_error:  # noqa: BLE001
            print(format_runtime_error(runtime_error), file=sys.stderr)
            return 1
        return 0

    if len(sys.argv) != 4:
        print(
            "usage: tensorflow_api_worker.py <create-model|train-model> <request-path> <result-path>",
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
        else:
            raise ValueError(f"unsupported action: {action}")
    except Exception as runtime_error:  # noqa: BLE001
        print(format_runtime_error(runtime_error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
