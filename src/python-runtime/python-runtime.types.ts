export type PythonJobCommand = {
  action: "create-model" | "predict-model" | "train-model";
  requestPath: string;
  resultPath: string;
};

export type PythonPredictionCommand = {
  artifactPath: string;
  modelId: string;
  predictionInput: {
    inputs: unknown;
  };
};

export type PythonExecutionResult = {
  errorMessage: string | null;
  isSuccess: boolean;
};

export type PythonPredictionResult =
  | {
      isSuccess: true;
      result: Record<string, unknown>;
    }
  | {
      errorMessage: string;
      isSuccess: false;
    };

export type PythonJobExecutor = (command: PythonJobCommand) => Promise<PythonExecutionResult>;

export type PythonPredictionExecutor = (command: PythonPredictionCommand) => Promise<PythonPredictionResult>;
